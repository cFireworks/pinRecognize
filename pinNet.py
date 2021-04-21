'''
# @Descripttion: 
# @Date: 2021-04-20 21:47:51
# @Author: cfireworks
# @LastEditTime: 2021-04-20 23:03:38
'''
from resnet import ResNet34
from keras.layers import Input, Dense, AveragePooling2D, Flatten
from keras import backend as K
from keras.objectives import categorical_crossentropy

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def createPinNet(input_shape=(224, 224, 3)):
    input_layer = Input(input_shape)
    x = ResNet34(input_layer)
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    out_probs = Dense(4, activation='softmax')(x)
    out_alias =  Dense(4, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=[out_probs, out_alias])
    return model



def class_loss_regr(y_true, y_pred):
    x = y_true[:,:,1:] - y_pred
    x_abs = K.abs(x)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
    return lambda_cls_regr * K.sum(y_true[:,:,0] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:,:,0])


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true, y_pred))