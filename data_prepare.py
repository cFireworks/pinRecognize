import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import io
import sys
import os
import cv2
from PIL import Image

try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

# get object annotation bndbox loc start


def GetAnnotBoxLoc(AnotPath: str) -> dict:
    """
    获取voc格式标注文件中的bndbox
        :param AnotPath: AnotPath VOC标注文件路径
    """
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)
        y1 = int(BndBox.find('ymin').text)
        x2 = int(BndBox.find('xmax').text)
        y2 = int(BndBox.find('ymax').text)
        BndBoxLoc = (x1, y1, x2, y2)
        if ObjName in ObjBndBoxSet:
            # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
            ObjBndBoxSet[ObjName].append(BndBoxLoc)
        else:
            # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
            ObjBndBoxSet[ObjName] = [BndBoxLoc]
    return ObjBndBoxSet


def boxSet2data(objBndSet, img_shape):
    """
    将类型字典转换为向量信息
        :param objBndSet: key为类型， value为bnd box的字典
        return 表示一个电气图元 上、下、左、右分别对应的点存在的相对坐标情况(a_up, a_down, a_left, a_right),不存在为None
    """
    h, w = img_shape[0], img_shape[1]
    a_up = (objBndSet["up"][2]+objBndSet["up"][0]-w) / w if "up" in objBndSet else 1024
    a_down = (objBndSet["down"][2]+objBndSet["down"][0]-w) / w if "down" in objBndSet else 1024
    a_left = (objBndSet["left"][3]+objBndSet["left"][1]-h) / h if "left" in objBndSet else 1024
    a_right = (objBndSet["right"][3]+objBndSet["right"][1]-h) / h if "right" in objBndSet else 1024
    return (a_up, a_down, a_left, a_right)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_dir, anno_dir, list_IDs, batch_size=1, img_size=(224, 224,3),
                 *args, **kwargs):
        """
        self.list_IDs:存放所有需要训练的图片文件名的列表。
        self.labels:记录图片标注的分类信息的pandas.DataFrame数据类型，已经预先给定。
        self.batch_size:每次批量生成，训练的样本大小。
        self.img_size:训练的图片尺寸。
        self.img_dir:图片在电脑中存放的路径。
        """
        self.anno_dir = anno_dir
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        """
            返回生成器的长度，也就是总共分批生成数据的次数。

        """
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        该函数返回每次我们需要的经过处理的数据。
        """
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X, Y_prob, Y_alias = self.__data_generation(list_IDs_temp)
        return X, [Y_prob, Y_alias]

    def on_epoch_end(self):
        """
        该函数将在训练时每一个epoch结束的时候自动执行，在这里是随机打乱索引次序以方便下一batch运行。

        """
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        """
            给定文件名，生成数据。
        """
        X = np.empty((self.batch_size, *self.img_size))
        Y_cls = np.empty((self.batch_size, 4), dtype=np.float32)
        Y_reg = np.empty((self.batch_size, 4), dtype=np.float32)

        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread(self.img_dir+ID+".png")
            X[i, ] = img
            anno_pth = os.path.join(self.anno_dir, ID+".xml")
            objSet = GetAnnotBoxLoc(anno_pth)
            a = boxSet2data(objSet, img.shape)

            Y_cls[i, ] = np.array([1 if n <=1 else 0 for n in a])
            Y_reg[i, ] = np.array([a[0], a[1], a[2], a[3]])

        return X, Y_cls, Y_reg
