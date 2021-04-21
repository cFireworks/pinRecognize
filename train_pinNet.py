'''
# @Descripttion: 
# @Date: 2021-04-20 22:46:54
# @Author: cfireworks
# @LastEditTime: 2021-04-21 23:11:04
'''
import os
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from pinNet import createPinNet, class_loss_regr, class_loss_cls
from data_prepare import DataGenerator


def train():
    input_shape_img = (224,224,3)
    model = createPinNet(input_shape_img)

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss=[class_loss_cls, class_loss_regr])

    num_epochs = 20
    img_dir = "/home/wkx/Workspace/Datasets/pinNetDataset/image/kg0"
    anno_dir =  "/home/wkx/Workspace/Datasets/pinNetDataset/label/kg0"
    fn_list = [os.path.basename(fn).split(".")[0] for fn in os.listdir(anno_dir)]
    data_generator = DataGenerator(img_dir, anno_dir, fn_list, batch_size=32, img_size=(224,224,3), epoch_len=1000)
    model.fit(data_generator,epochs=num_epochs)


if __name__ == "__main__":
    train()