

from torch import nn
import torch
import copy
import math
import sys

from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertPreTrainedModel
from .classifier import MultiNonLinearClassifier
from .span_layer import SpanLayer
from loader4image import ImageModel
import torch.nn.functional as F
from torch.distributions import Normal, Independent

ANNEALING_RATIO = 0.3
SMALL = 1e-08
SAMPLE_SIZE = 5

# span embedding settings
MAX_SPAN_LEN = 4
MORPH_NUM = 5
MAX_SPAN_NUM = 502
TOKEN_LEN_DIM = 50
SPAN_LEN_DIM = 50
SPAN_MORPH_DIM = 100
BOTTLE_NECK_DIM = 100

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        # self.num_attention_heads=6
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):

        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DistEstimate(nn.Module):
    '''
     refers the implementation of https://github.com/mfederici/Multi-View-Information-Bottleneck/blob/master/utils/modules.py
    '''
    def __init__(self, x_dim, hidden_dim, out_dim):
        super(DistEstimate, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.x_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.out_dim*2)
        )

    def forward(self, x):
        params = self.net(x)

        mu, sigma = params[:, :self.out_dim], params[:, self.out_dim: ]
        sigma = F.softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(mu, sigma), 1)

class IBEstimator(nn.Module):
    def __init__(self, size1, size2, hidden_dim):
        super(IBEstimator,self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )

    # Gradient for JSD mutual information estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -F.softplus(-pos).mean() - F.softplus(neg).mean()

class BertSpanNerBN(BertPreTrainedModel):
    def __init__(
        self,
        config,
        args=None,
        num_labels=None
    ):
        super(BertSpanNerBN, self).__init__(config)

        # ---------------- encoder ------------------
        self.bert = BertModel(config)
        self.image_model=ImageModel()
        self.img2uni=nn.Linear(2048,768)
        self.txt2uni=nn.Linear(1736,768)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.n_class = 5 if not num_labels else num_labels

        # ---------------- span layer ------------------
        self.span_layer = SpanLayer(config.hidden_size, TOKEN_LEN_DIM, SPAN_LEN_DIM,
                                    SPAN_MORPH_DIM, MAX_SPAN_LEN, MORPH_NUM)
        # start + end + token len + span len + morph
        span_dim = config.hidden_size * 2 + TOKEN_LEN_DIM \
                   + SPAN_LEN_DIM + SPAN_MORPH_DIM
        self.txt2img_attention=BertCrossEncoder(config,layer_num=1)
        # self.img2txt_attention = BertCrossEncoder(config, layer_num=1)

        self.dist_t = DistEstimate(768, 768, 768)
        self.dist_v = DistEstimate(768, 768, 768)

        self.tv_ib = IBEstimator(768, 768, 768)

        # self.mu_img=nn.Sequential(nn.Linear(768, 768),
        #               nn.ReLU(),
        #               nn.Linear(768, 768),
        #               # nn.ReLU(),
        #               # nn.Linear(768, 768)
        #               )
        #
        # self.logvar_img = nn.Sequential(nn.Linear(768, 768),
        #                             nn.ReLU(),
        #                             nn.Linear(768, 768),
        #                             # nn.ReLU(),
        #                             # nn.Linear(768, 768)
        #                             )

        # ---------------- classifier ------------------
        # self.span_classifier = MultiNonLinearClassifier(
        #     config.hidden_size, self.n_class, dropout_rate=0.2
        # )
        # self.span_classifier_txt = MultiNonLinearClassifier(
        #      config.hidden_size, self.n_class, dropout_rate=0.2
        # )
        self.span_classifier_img = MultiNonLinearClassifier(
            config.hidden_size*2, self.n_class, dropout_rate=0.2
        )

        # self.span_classifier = MultiNonLinearClassifier(
        #     BOTTLE_NECK_DIM, self.n_class, dropout_rate=0.2
        # )

        self.softmax = torch.nn.Softmax(dim=-1)
        self.beta = args.beta

        self.init_weights()




    def get_mu_logvar_img(self,x):
        mu=self.mu_img(x)
        logvar=self.logvar_img(x)
        logvar=nn.functional.softplus(logvar) + 1e-7

        return mu,logvar

    def reparameterise(self,mu,logvar):
        eps = torch.randn_like(logvar)

        z = mu + torch.exp(0.5 * logvar) * eps

        return z

    def forward(self, ori_feas):
        """
        默认有 labels 为 train, 无 labels 为 test
        """
        #ori_encoding, cont_encoding = {}, {}
        ori_encoding = {}
        ori_encoding['spans_rep'],ori_encoding['txt_rep_g'] = self.span_encoding(**ori_feas)
        ori_encoding['spans_rep']=self.dropout(ori_encoding['spans_rep'])
        span_rep_map=self.txt2uni(ori_encoding['spans_rep'])

        text_dist=self.dist_t(ori_encoding['txt_rep_g'])
        t=text_dist.rsample()

        #print(ori_encoding['spans_rep'].shape)
        batch_size=ori_encoding['spans_rep'].shape[0]
        # cont_encoding['spans_rep'] = self.span_encoding(**cont_feas)
        images=ori_feas['image']
        ori_encoding['imag_rep_g'],ori_encoding['imag_rep_l']=self.image_model(images)
        ori_encoding['imag_rep_g'] = self.img2uni(ori_encoding['imag_rep_g'])
        img_dist = self.dist_v(ori_encoding['imag_rep_g'])
        v = img_dist.rsample()

        # vis_embed_global=ori_encoding['imag_rep_g'].unsqueeze(1)
        vis_embed_global=v.unsqueeze(1)
        vis_embed_local=ori_encoding['imag_rep_l'].view(-1,2048,49).permute(0,2,1)
        # vis_embed=torch.cat([vis_embed_global,vis_embed_local],dim=1)
        # vis_embed_map=self.img2uni(vis_embed)
        vis_embed_local=self.img2uni(vis_embed_local)
        vis_embed_map=torch.cat([vis_embed_global,vis_embed_local],dim=1)


        img_mask=torch.ones(batch_size,50)
        extended_img_mask=img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask=(1.0-extended_img_mask)*-10000.0
        # print(extended_img_mask)
        extended_img_mask=extended_img_mask.to(vis_embed_map.device)
        # print(extended_img_mask.device)
        txt2img_rep=self.txt2img_attention(span_rep_map,vis_embed_map,extended_img_mask)[-1]

        span_fusion_rep=torch.cat([span_rep_map,txt2img_rep],dim=-1)
        ori_encoding['logits'] = self.span_classifier_img(span_fusion_rep)
        # ori_encoding['logits'] = self.span_classifier(span_fusion_rep)

        loss_dict = {}
        outputs = [self.softmax(ori_encoding['logits'])]



        if ori_feas['span_labels'] is not None:
            # TODO, test stop gradient of cont features
            # cont_encoding['spans_rep'].detach()
            loss_dict = self.compute_loss(ori_feas, ori_encoding,t,v,text_dist,img_dist)
            # loss_dict = self.compute_loss(ori_feas, ori_encoding)

        return outputs, loss_dict  # (loss), scores

    def span_encoding(self, input_ids=None, input_mask=None, segment_ids=None,
                      span_token_idxes=None, span_lens=None, morph_idxes=None, **kwargs):
        """
        Encode tokens by Bert, and get span representations.
        """
        # encoder [batch, seq_len, hidden]
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids
        )

        sequence_output_global=sequence_output[1]

        span_rep = self.span_layer(sequence_output[0], span_token_idxes.long(), span_lens, morph_idxes)

        return span_rep,sequence_output_global

    def compute_loss(self, ori_feas, ori_encoding,t=None,v=None,text_dist=None,img_dist=None):
        """
        计算loss, 包括: span分类loss, gi loss, si loss
        """
        # ----------compute span classification loss----------
        # loss_dic = {'c': self.compute_clas_loss(ori_feas, ori_encoding),
        #             # 'cc': self.compute_clas_loss(cont_feas, cont_encoding)
        #             }
        loss_dic = {'c': self.compute_clas_loss(ori_feas, ori_encoding,mode="logits")
                    # 'cc': self.compute_clas_loss(cont_feas, cont_encoding)
                    }
        # loss_dic['c_txt'] = self.compute_clas_loss(ori_feas, ori_encoding, mode="logits_txt")
        # loss_dic['c_img'] = self.compute_clas_loss(ori_feas, ori_encoding, mode="logits_img")
        # priori_loss=self.bn_loss.update(mu_t,logvar_t)
        # loss_dic['pi_t']=0.0001*priori_loss

        #****************T和V之间的信息瓶颈************
        mi_tv = self.tv_ib(v, t)
        t_prob = text_dist.log_prob(t).exp()
        v_prob = img_dist.log_prob(v).exp()
        kl_tv = (text_dist.log_prob(t) - img_dist.log_prob(t)) * t_prob
        kl_vt = (img_dist.log_prob(v) - text_dist.log_prob(v)) * v_prob
        # skl_tv = (kl_tv + kl_vt).mean() / 2.
        skl_tv=kl_vt.mean()

        loss_dic['loss_ib_tv'] = -0.1*mi_tv + 0.001*skl_tv
    



        # ----------compute KL(p(z_o|t), p(z_c|t)) loss----------
        # x_span_cont, y_span_cont = span_select(
        #     ori_encoding['spans_rep'].unsqueeze(1), ori_feas['cont_span_idx'],
        #     cont_encoding['spans_rep'].unsqueeze(1), cont_feas['cont_span_idx']
        # )
        #
        # entity_dist = self.oov_reg.update(x_span_cont, y_span_cont)
        # loss_dic['si'] = self.beta * entity_dist
        #
        # # ----------compute MI(z_o, z_c) loss----------
        # entity_mi = self.gama * self.z_reg(x_span_cont, y_span_cont)
        # loss_dic['gi'] = entity_mi

        # ----------sum loss----------
        # loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])

        # loss_dic['loss']=loss_dic['c']
        loss_dic['loss'] = sum([item[1] for item in loss_dic.items()])

        return loss_dic
    def compute_clas_loss(self, features, encoding,mode='logits'):
        """
        计算分类loss.
        """
        batch_size, n_span = features['span_labels'].size()
        ori_span_labels = features['span_labels'].view(-1)
        if mode=="logits":
            ori_span_rep = encoding['logits'].view(-1, self.n_class)
        elif mode=='logits_txt':
            ori_span_rep = encoding['logits_txt'].view(-1, self.n_class)
        elif mode=='logits_img':
            ori_span_rep = encoding['logits_img'].view(-1, self.n_class)


        clas_loss = self.cross_entropy(ori_span_rep, ori_span_labels)
        clas_loss = clas_loss.view(batch_size, n_span) * features['span_weights']
        clas_loss = torch.masked_select(clas_loss, features['span_masks'].bool()).mean()

        return clas_loss
    def compute_clas_loss_yuan(self, features, encoding):
        """
        计算分类loss.
        """
        batch_size, n_span = features['span_labels'].size()
        ori_span_rep = encoding['logits'].view(-1, self.n_class)
        ori_span_labels = features['span_labels'].view(-1)

        clas_loss = self.cross_entropy(ori_span_rep, ori_span_labels)
        clas_loss = clas_loss.view(batch_size, n_span) * features['span_weights']
        clas_loss = torch.masked_select(clas_loss, features['span_masks'].bool()).mean()

        return clas_loss
