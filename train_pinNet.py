'''
# @Descripttion: 
# @Date: 2021-04-20 22:46:54
# @Author: cfireworks
# @LastEditTime: 2021-04-20 23:07:52
'''
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from pinNet import createPinNet, class_loss_regr, class_loss_cls

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()



def train():
    model = createPinNet(input_shape_img)
    model.load_weights()
    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss=[class_loss_cls, class_loss_regr])
    num_epochs = 20000
    iter_num = 0
    train_step = 0

    start_time = time.time()

    best_loss = np.Inf

    for epoch_num in range(num_epochs):
        X, Y, img_data = next(data_gen_train)
        loss_pin = model.train_on_batch(X, Y)

        train_step += 1
        iter_num += 1

        if iter_num == epoch_length:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if C.verbose:
                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                print('Loss RPN regression: {}'.format(loss_rpn_regr))
                print('Loss Detector classifier: {}'.format(loss_class_cls))
                print('Loss Detector regression: {}'.format(loss_class_regr))
                print('Elapsed time: {}'.format(time.time() - start_time))

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            iter_num = 0
            start_time = time.time()

            write_log(callback,
                      ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                       'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                      [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                       loss_class_cls, loss_class_regr, class_acc, curr_loss],
                      epoch_num)

            if curr_loss < best_loss:
                if C.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                best_loss = curr_loss
                model_all.save_weights(C.model_path)