# coding=utf-8

import math
import sys

import torch

sys.path.append('../')

#import fitlog
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AutoConfig, AutoTokenizer
from eval_metric import span_f1_prune

from engine_utils import *
from utils.datasets import collate_fn, get_labels, load_examples, SpanNerDataset

from models.bn_bert_ner_MIB import BertSpanNerBN

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def get_args():
    parser = arg_parse()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        # default="/root/MINER2/MINER/data/WNUT2017/",
        default="data/twitter2017",
        type=str,
        help="The input data dir. Should contain the training files for the "
             "CoNLL-2003 NER task.",
    )

    parser.add_argument(
        "--output_dir",
        default="out/twitter2017-MIB-news/bert_uncase/",
        type=str,
        help="The output directory where the model predictions and "
             "checkpoints will be written.",
    )

    # fitlog debug settings, --debug for True, otherwise for False
    # parser.add_argument("--debug", action="store_true",
    #                     help="Whether record results and params.")

    # Other parameters
    parser.add_argument("--gpu_id", default=1, type=int,
                        help="GPU number id")

    parser.add_argument(
        "--epoch", default=30, type=float,
        help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--beta", default=0.001, type=float,
        help="weights of oov regular"
    )

    parser.add_argument(
        "--gama", default=0.01, type=float,
        help="weights of oov regular"
    )

    parser.add_argument(
        "--beta_txt", default=0.001, type=float,
        help="weights of InfoNCE"
    )

    # 0 means typos， 1 means switch
    # parser.add_argument(
    #     "--switch_ratio", default=0.5, type=float,
    #     help="Entity switch ratio."
    # )

    # training parameters
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--do_train", default=True,type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=True,type=bool,
                        help="Whether to run predictions on the test set.")
    # TODO, try different seed
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed for initialization")
    # parser.add_argument("--do_robustness_eval", action="store_true",
    #                     help="Whether to evaluate robustness")

    args = parser.parse_args()
    # Path to a file containing all labels.
    args.labels = os.path.join(args.data_dir, './labels.txt')
    # Path to a file containing import substring of each category
    # args.pmi_json = os.path.join(args.data_dir, './pmi.json')
    # Path to a file containing entities of each category
    # args.entity_json = os.path.join(args.data_dir, './entity.json')
    if args.gpu_id==-1:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
    args.device = device

    return args


args = get_args()
set_seed(args)




# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.warning("Process device: %s", args.device)


def train(args, model, tokenizer, labels):
    """ Train the model """
    train_examples = load_examples(args.data_dir, mode="train", tokenizer=tokenizer)
    #training_steps = (len(train_examples) - 1 / args.batch_size + 1) * args.epoch
    training_steps=math.ceil(len(train_examples)/args.batch_size)*args.epoch
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = prepare_optimizer_scheduler_MIB(args, model, training_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Total train batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", training_steps)

    global_step = 0
    best_score = 0.0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.epoch), desc="Epoch")
    epoch_num = 0
    n_patience=0

    for _ in train_iterator:
        epoch_num += 1
        train_dataset = SpanNerDataset(train_examples, args=args, tokenizer=tokenizer, labels=labels)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=collate_fn)

        logger.info("Training epoch num {0}".format(epoch_num))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            save_flag = False
            model.train()
            ori_fea_tensors = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(ori_fea_tensors)

            loss_dic = outputs[1]
            loss = loss_dic['loss']
            loss.backward()
            #print(loss_dic)

            #fitlog.add_loss(loss.tolist(), name="Loss", step=global_step)
            tr_loss += loss.item()
            description = "".join(["{0}:{1}, ".format(k, round(v.item(), 5))
                                   for k, v in loss_dic.items()]).strip(', ')
            epoch_iterator.set_description(description)

            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % len(train_dataloader) == 0:
                # default evaluate during training
                results, _ = evaluate(
                    args, model, tokenizer, labels,
                    mode="dev", prefix="{}".format(global_step)
                )


                if best_score < results['span_f1']:
                    save_flag=True
                    n_patience=0
                    best_score = results['span_f1']
                    output_dir = os.path.join(args.output_dir, "best_checkpoint")
                    print("best score f1:",best_score)
                    #fitlog.add_best_metric({"dev-f1": results['span_f1']})
                else:
                    n_patience+=1
                    print("n_patience:",n_patience)
                    #output_dir = args.output_dir
                if save_flag:
                    model_save(args,output_dir,model,tokenizer)
        if args.patience>0 and n_patience==args.patience:
            break
    print("best_score span f1:",best_score)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, mode='test', prefix='', examples=None):
    eval_examples = examples if examples else load_examples(args.data_dir, mode=mode, tokenizer=tokenizer)
    eval_dataset = SpanNerDataset(eval_examples, args=args, tokenizer=tokenizer, labels=labels, dev=True)

    # accelerate evaluation speed
    args.eval_batch_size = 64
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    logger.info("***** Running evaluation {0} {1} *****".format(mode, prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    dev_outputs = []
    model.eval()    # close drop out, batch normalization

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        ori_fea_tensors = {k: v.to(args.device) for k, v in batch.items()}
        #cont_fea_tensors = {k: v.to(args.device) for k, v in batch[1].items()}

        with torch.no_grad():
            # without labels, direct out tags
            predicts, _ = model(ori_fea_tensors)
            # span_f1_prune(all_span_idxs, predicts, span_label_ltoken, real_span_mask_ltoken)
            span_f1s, pred_label_idx = span_f1_prune(
                ori_fea_tensors['span_word_idxes'],
                predicts[0],
                ori_fea_tensors['span_labels'],
                ori_fea_tensors['span_masks']
            )
            outputs = {
                'span_f1s': span_f1s,
                'pred_label_idx': pred_label_idx,
                'all_span_idxs': ori_fea_tensors['span_word_idxes'],
                'span_label_ltoken': ori_fea_tensors['span_labels']
            }
            dev_outputs.append(outputs)

        nb_eval_steps += 1

    all_counts = torch.stack([x[f'span_f1s'] for x in dev_outputs]).sum(0)
    correct_pred, total_pred, total_golden = all_counts
    print('correct_pred, total_pred, total_golden: ', correct_pred, total_pred, total_golden)
    precision = correct_pred / (total_pred + 1e-10)
    recall = correct_pred / (total_golden + 1e-10)
    f1 = precision * recall * 2 / (precision + recall + 1e-10)

    res = {
        'span_precision': round(precision.cpu().numpy().tolist(), 5),
        'span_recall': round(recall.cpu().numpy().tolist(), 5),
        'span_f1': round(f1.cpu().numpy().tolist(), 5)
    }
    logger.info("{0} metric is {1}".format(prefix, res))

    # save metrics result
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {0} {1} *****".format(mode, prefix))
        writer.write("***** Eval results {0} {1} *****\n".format(mode, prefix))
        for key in sorted(res.keys()):
            logger.info("{} = {}".format(key, str(res[key])))
            writer.write("{} = {}\n".format(key, str(res[key])))

    return res, dev_outputs  # dev_outputs is a list of outputs, which is [outputs1, outputs2 ... ]


def fast_evaluate(args, ckpt_dir, config, tokenizer, labels,
                  mode, prefix='', model=None):
    if not model:
        print("The best dev model was saved in path:",ckpt_dir)
        model = BertSpanNerBN.from_pretrained(
            ckpt_dir,
            config=config,
            num_labels=len(labels),
            args=args
        )
        model.to(args.device)

    results, predictions = evaluate(args, model, tokenizer, labels, mode=mode, prefix=prefix)
    output_eval_file = os.path.join(ckpt_dir, "{0}_results.txt".format(mode))

    with open(output_eval_file, "a") as writer:
        writer.write('***** Predict in {0} {1} dataset *****\n'.format(mode, prefix))

    return results, predictions


def main():

    # modified, Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    logger.info('The list of lables is: %s ',labels)
    print('Number of labels is: ',len(labels))

    # ------------config--------------
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None
    )
    args.model_type = config.model_type.lower() #args.model_type: bert

    # ------------tokenizer--------------
    tokenizer_args = {k: v for k, v in vars(args).items()
                      if v is not None and k in TOKENIZER_ARGS}
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=None,
        **tokenizer_args,
    )

    # ------------load pre-trained model--------------
    model = BertSpanNerBN.from_pretrained(args.model_name_or_path, config=config,
                                          num_labels=len(labels), args=args)
    

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    #print(args.do_train)


    # ------------Training------------
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, labels)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")

    # ------------Evaluation------------
    if args.do_eval:
        # eval best checkpoint
        fast_evaluate(args, best_ckpt_dir, config, tokenizer, labels,
                      mode="dev", prefix="best ckpt")

    # ------------Prediction------------
    if args.do_predict:
        # eval best checkpoint
        results, predictions = fast_evaluate(args, best_ckpt_dir, config,  tokenizer,
                                             labels, mode="test", prefix="best ckpt")
        #fitlog.add_metric({"test": {"best_ckpt_f1": results["span_f1"]}}, step=0)

        # Save predictions
        test_file = os.path.join(args.data_dir, "test.txt")
        output_test_predictions = os.path.join(best_ckpt_dir, "test_predictions.txt")
        predictions_save(test_file, predictions, output_test_predictions, labels)


if __name__ == "__main__":
    main()

