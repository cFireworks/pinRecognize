'''
Author: your name
Date: 2020-12-31 13:25:53
LastEditTime: 2021-01-04 15:49:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /detection_lab/symbol-recognition/test_resnet.py
'''
import os
import os.path as ops
from keras.preprocessing.image import ImageDataGenerator
from resnet import ResNet34, ResNet50

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset_pth = "/home/wkx/Workspace/ee-schematic-analysis/tf-object-detector-model/my_datasets/huabei_data/"
data_dir = ops.join(dataset_pth, "recognition_dataset")

img_size = (224, 224)
train_gen = ImageDataGenerator(rescale=1/255., validation_split=0.2)
train_generator = train_gen.flow_from_directory(data_dir, 
                                                target_size=img_size,
                                                batch_size=32,
                                                class_mode="categorical",
                                                subset='training',
                                               )
valid_generator = train_gen.flow_from_directory(data_dir, 
                                                target_size=img_size,
                                                batch_size=32,
                                                class_mode="categorical",
                                                subset='validation',
                                               )

resnet = ResNet50(n_classes=34)
resnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = resnet.fit_generator(train_generator, 
                              steps_per_epoch=train_generator.n//train_generator.batch_size, 
                              validation_data=valid_generator, 
                              validation_steps=valid_generator.n//valid_generator.batch_size, 
                              epochs=50, 
                             )