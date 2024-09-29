import os
import numpy as np
#import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Tuple
#from tensorflow.data import Dataset
#from tensorflow.keras.preprocessing import image
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,SequentialSampler
#import torchvision.models as models
#img_model=models.resnet152()
from torchvision.models import resnet152,resnet
import torch.nn as nn
import torch.nn.functional as F


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel,self).__init__()
        net=getattr(resnet,'resnet152')()
        net.load_state_dict(torch.load("./resnet/resnet152-b121ed2d.pth"))
        #self.resnet=resnet152(pretrained=True)
        self.resnet=net


    def forward(self, x, att_size=7):
        # x shape batch_size * channels * 224 * 224
        # 64 * 3 * 224 * 224

        # batch_size * channels * 112 * 112
        # 64 * 64 * 112 * 112
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # 64 * 256 * 56 * 56
        x = self.resnet.maxpool(x)

        # 32 * 512 * 56 * 56
        x = self.resnet.layer1(x)
        # 32 * 512 * 28 * 28
        x = self.resnet.layer2(x)
        # 32 * 1024 * 14 * 14
        x = self.resnet.layer3(x)
        # 32 * 2048 * 7 * 7
        x = self.resnet.layer4(x)

        # 32 * 2048
        # fc = x.mean(3).mean(2)
        # 32 * 2048 * 7 * 7
        att = F.adaptive_avg_pool2d(x, [att_size, att_size])

        # 32 * 2048 * 1 * 1
        x = self.resnet.avgpool(x)
        # 32 * 2048 * 2048
        x = x.view(x.size(0), -1)

        return x, att

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image
def read_image (path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # args.crop_size, by default it is set to be 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    file_names = os.listdir(path)
    imgids, imgs = [], []
    dict_features=[]
    prom_num=0
    for file_name in file_names:
        image_path=os.path.join(path, file_name)
        try:
            image = image_process(image_path, transform)
        except:
            print(image_path + " has problem!")
            prom_num+=1
            image_path_fail = os.path.join(path, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)
        # print(type(image))
        # image=image.unsqueeze(0)

        imgs.append(image)
        imgids.append(file_name.split(".")[0])
       # print(imgs.shape)
    print("total num of problem images:", prom_num)
    # imgs = np.vstack(imgs)
    return (imgids, imgs)



class ImageDataset(Dataset):
    def __init__(self,examples):
        self.examples=examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


'''

class ImageLoader(object):
    def __init__(self, path: str) -> None:
        self.__feature_path = f"{path}/dict_feature.pkl"
        self.img_model =ImageModel()
        # imgs = read_image(path)
        # # print(len(imgs[0]))
        # # print(imgs[1].shape)
        # features = self.__parse_data(imgs, self.__feature_path)

        if not os.path.exists(self.__feature_path):
            imgs = read_image(path)
            features = self.__parse_data(imgs, self.__feature_path)
        else:
            features = read_pickle(self.__feature_path)
        self.__dict_features = features
        #print(len(list(self.__dict_features.values())))

        self.__feature_shape = list(self.__dict_features.values())[0].shape
        #print(self.__feature_shape)
        #exit()

    def getFeature(self, imgid: str):
        return self.__dict_features.get(imgid, None).reshape(
            (1, self.__feature_shape[-1])
        )

    def __parse_data(self, data: Tuple, save_path: str) -> Dict:
        imgids, imgs = data
        #print(imgs[0])
        #print(type(imgs))
        img_dataset=ImageDataset(imgs)
        img_sampler=SequentialSampler(img_dataset)
        img_dataloader=DataLoader(img_dataset,shuffle=False,sampler=img_sampler,batch_size=64)
        features = []
        for batch_data in tqdm(img_dataloader,ncols=80,ascii=True):
            img_global_feature,img_local_feature=self.img_model(batch_data)
            # print(type(img_global_feature))
            # print(img_global_feature.shape)
            # print(img_local_feature.shape)
            # print(img_local_feature)
            # exit()
            features.append(img_global_feature)

        features = np.vstack(features)
        # print(features.shape)
        # print(type(features))
        # exit()
        dict_features = {}
        for imgid, feature in zip(imgids, features):
            dict_features[imgid] = feature
        pd.to_pickle(dict_features, save_path)
        return dict_features

'''
if __name__ == "__main__":
    outputs=read_image("./data/twitter2017_images")
    #img_model=ImageModel()
    imgids,images=outputs
    net=getattr(resnet,'resnet152')()
    net.load_state_dict(torch.load("./resnet/resnet152-b121ed2d.pth"))
    img_model=ImageModel(net)
    #imgids, imgs = data
    # print(imgs[0])
    # print(type(imgs))
    img_dataset = ImageDataset(images)
    img_sampler = SequentialSampler(img_dataset)
    img_dataloader = DataLoader(img_dataset, shuffle=False, sampler=img_sampler, batch_size=32)
    features = []
    for batch_data in tqdm(img_dataloader, ncols=80, ascii=True):
        img_global_feature, img_local_feature = img_model(batch_data)
        print(img_local_feature.shape)
        print(img_global_feature.shape)




    # imgLoader = ImageLoader("./data/twitter2017_images")
    # feature=imgLoader.getFeature("9683")
    # print(type(feature))
    # print(feature.shape)
    # print(feature)
