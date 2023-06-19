import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
import math
import matplotlib.pyplot as plt
import time
import cv2
import os
import csv


def fully_connected(prev_layer, num_units, is_training, use_bias=False):
    """
    num_units参数传递该层神经元的数量，根据prev_layer参数传入值作为该层输入创建全连接神经网络。
   :param prev_layer: Tensor
        该层神经元输入
    :param num_units: int
        该层神经元结点个数
    :param is_training: bool or Tensor
        表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
    :returns Tensor
        一个新的全连接神经网络层
    """
    layer = tf.layers.dense(prev_layer, num_units, use_bias=use_bias, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def conv_layer(prev_layer, filters, is_training, kernel_size=[3, 3], strides=1, padding='same'):
    """
   使用给定的参数作为输入创建卷积层
    :param prev_layer: Tensor
        传入该层神经元作为输入
    :param layer_depth: int
        我们将根据网络中图层的深度设置特征图的步长和数量。
        这不是实践CNN的好方法，但它可以帮助我们用很少的代码创建这个示例。
    :param is_training: bool or Tensor
        表示该网络当前是否正在训练，告知Batch Normalization层是否应该更新或者使用均值或方差的分布信息
    :returns Tensor
        一个新的卷积层
    """
    conv_layer = tf.layers.conv2d(prev_layer, filters, kernel_size, strides, padding, use_bias=False, activation=None)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    conv_layer = tf.nn.relu(conv_layer)
    return conv_layer


def convert_to_one_hot(y, class_qty):
    return np.eye(class_qty)[y]


def Batch_Data(folder, info_array, batch_i, class_qty, height, width, batch_size=64):
    # 定义和训练数据相同大小的索引
    qty = info_array.shape[0]
    data_index = np.arange(qty)
    # 设定每个批次对应的索引
    start_index = (batch_i * batch_size) % qty
    # print('start_index', batch_i, 'batch_size', batch_size, 'qty', qty)
    index = data_index[start_index:min(start_index + batch_size, qty)]
    # print('batch_i',batch_i,'start_index',start_index, 'end_index',min(start_index + batch_size, qty))
    # print(index)

    data_image = np.empty((index.shape[0], height, width, 1), dtype='int32')
    labels = []
    for i, info in enumerate(info_array[index]):
        # print('cv2.imread',folder + info[0].decode())
        img = cv2.imread(folder + info[0].decode().replace('.jpg', '.png'), 0)
        data_image[i, :, :, 0] = img
        labels.append(int(info[2]))
    # print('labels',i,labels[i])
    # plt.imshow(data_image[i, :, :, 0], cmap='gray')
    # plt.show()
    batch_xs = data_image.astype('float32')
    batch_ys = convert_to_one_hot(np.array(labels), class_qty)
    # print('batch_i',batch_i,'batch_xs', batch_xs.shape,'batch_ys',batch_ys.shape)
    return batch_xs, batch_ys


def Train(expand_folder, vali_folder, train_info, vali_info, class_qty, height, width, train_opt, model_loss,
          correct_prediction, softmax_tensor, inputs,
          labels,
          is_training,
          batch_size=64, loop_n=1, batch_freq=0.05, loop_freq=1, pb_file_folder="./model"):
    # 配置GPU显存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:

    # 训练并测试网络模型
    with tf.Session(config=config) as sess:
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        # with tf.device('/gpu:0'):
        sess.run(tf.global_variables_initializer())
        train_qty = train_info.shape[0]
        # 按Batch分批训练N次的循环
        for loop_i in range(loop_n):
            batch_qty = math.ceil(train_qty / batch_size)
            for batch_i in range(batch_qty):
                batch_xs, batch_ys = Batch_Data(expand_folder, train_info, batch_i, class_qty, height, width,
                                                batch_size)
                # 训练样本批次
                _, loss, correct, _ = sess.run([train_opt, model_loss, correct_prediction, softmax_tensor],
                                               {inputs: batch_xs, labels: batch_ys, is_training: True})
                # 定期检查训练集上的loss和精确度
                if batch_i in np.arange(0, batch_qty, math.ceil(batch_qty * batch_freq)) and batch_freq > 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                          'Loop: {:>2}, Batch: {:>2}: Training loss: {:>3.5f}, error: {:>2}, accuracy: {:>3.5f}'.format(
                              loop_i, batch_i, loss, batch_ys.shape[0] - np.sum(correct),
                                                     np.sum(correct) / batch_ys.shape[0]))
            if loop_i % loop_freq == 0 and loop_freq > 0:
                # train_softmax = Predict(folder,train_info, class_qty, sess,train_opt,
                # 						model_loss, correct_prediction, softmax_tensor, inputs, labels, is_training,
                # 						batch_size=batch_size, loop_i=loop_i, predict_name='Train Predict')
                vali_softmax, accuracy = Predict(vali_folder, vali_info, class_qty, height, width, sess,
                                                 model_loss, correct_prediction, softmax_tensor, inputs, labels,
                                                 is_training,
                                                 batch_size=batch_size, predict_name='Vali Predict')
                # 在训练结束之后，保持神经网络模型
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                           output_node_names=['model_loss',
                                                                                              'correct_prediction',
                                                                                              'softmax_tensor'])
                pb_file_path = pb_file_folder + "_" + str(loop_i) + "_" + str(accuracy) + ".pb"
                print(pb_file_path, os.path.exists(pb_file_path))
                if os.path.isfile(pb_file_path):  # 如果文件存在
                    os.remove(pb_file_path)
                with tf.gfile.GFile(pb_file_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

    return vali_softmax


def Predict(folder, predict_info, class_qty, height, width, sess,
            model_loss, correct_prediction, softmax_tensor, inputs, labels, is_training,
            batch_size=64, predict_name='Predict'):
    qty = predict_info.shape[0]
    correct_sum = 0
    loss_list = []
    softmax = np.empty(dtype=np.float32, shape=[0, class_qty])  # 定义numpy空数组
    if qty > 0:
        for batch_i in range(math.ceil(qty / batch_size)):
            batch_xs, batch_ys = Batch_Data(folder, predict_info, batch_i, class_qty, height, width, batch_size)
            loss, correct, softmax_batch = sess.run([model_loss, correct_prediction, softmax_tensor],
                                                    {inputs: batch_xs, labels: batch_ys, is_training: False})
            correct_sum += np.sum(correct)
            loss_list.append(loss)
            softmax = np.concatenate((softmax, softmax_batch))  # 将每个批次的概率结果追加到softmax_array
        #                 with printoptions(precision=2, suppress=True):  # 格式化打印各个分类的概率（保留2位小数，不使用科学计数法）
        #                     print('softmax_rate:', softmax_array[-1])
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), predict_name,
              'loss: {:>3.5f}, error: {:>2}, accuracy: {:>3.5f}'.format(np.mean(np.array(loss_list)),
                                                                        qty - correct_sum,
                                                                        correct_sum / qty))
        accuracy = correct_sum / qty
    return softmax, accuracy


def Load_Model_Predict(folder, predict_info, class_qty, height, width,
                       batch_size=64, predict_name='Predict', pb_file_path="./model.pb"):
    with tf.gfile.GFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.reset_default_graph()  # import_graph前先清空所有图，防止以前存储的图误计算导致的错误
        tf.import_graph_def(graph_def, name='')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 恢复tensorflow placeholder
        inputs = sess.graph.get_tensor_by_name('inputs:0')
        labels = sess.graph.get_tensor_by_name('labels:0')
        is_training = sess.graph.get_tensor_by_name('is_training:0')
        # 恢复tensorflow输出tensor
        model_loss = sess.graph.get_tensor_by_name('model_loss:0')
        correct_prediction = sess.graph.get_tensor_by_name('correct_prediction:0')
        softmax_tensor = sess.graph.get_tensor_by_name('softmax_tensor:0')
        sess.run(tf.global_variables_initializer())

        softmax, accuracy = Predict(folder, predict_info, class_qty, height, width, sess,
                                    model_loss, correct_prediction, softmax_tensor, inputs, labels, is_training,
                                    batch_size=batch_size, predict_name=predict_name)
    return softmax


def Show_Result(info, softmax, classes, samples=20, figure_qty=3):
    label_list = info[:, 2].astype('int32')
    argmax_list = np.argmax(softmax, 1)  # 将结果去one hot化
    result_list = np.equal(label_list, argmax_list)  # 对比结果是否正确

    if len(label_list) > 0:
        #     new_images = images.reshape(images.shape[0], 28, 28)  # reshape前是4维数组，reshape后是3维数组
        error_list = np.flatnonzero(result_list == False)  # 挑选全部分类错误的index
        print('Show_Result')
        # 打印每个分类的正确率和错误数量
        for i, c in enumerate(classes):
            index_list = np.flatnonzero(label_list == i)  # flatnonzero取出不为零的位置，挑选当前分类的index
            class_error_list = np.intersect1d(index_list, error_list)  # 挑选当前分类错误的index
            if (len(index_list) > 0):
                print('Class{:>2}:{:>2}, qty: {:>2}, error: {:>2}, accuracy: {:>3.5f}'.format(i, c, len(index_list),
                                                                                              len(class_error_list), (
                                                                                                      len(
                                                                                                          index_list) - len(
                                                                                                  class_error_list)) / len(
                        index_list)))
            else:
                print('Class{:>2}:{:>2}, qty: {:>2}, error: {:>2}, accuracy: {:>3.5f}'.format(i, c, len(index_list),
                                                                                              len(class_error_list), 1))
        print('Total Result qty: {:>2}, error: {:>2}, accuracy: {:>3.5f}'.format(len(label_list), len(error_list), (
                len(label_list) - len(error_list)) / len(label_list)))

    #         finish_list = np.zeros(numClass)  # 是否每个分类的错误都显示完毕
    #         for n in range(math.ceil(len(error_list) / samples)):
    #             if (n > figure_qty - 1):
    #                 break
    #             for i, c in enumerate(classes):
    #                 index_list = np.flatnonzero(label_list == i)  # flatnonzero取出为True的位置，挑选当前分类的index
    #                 class_error_list = np.intersect1d(index_list, error_list)  # 挑选当前分类错误的index
    #                 if (len(class_error_list) > n * samples):  # 判断当前分类的错误是否显示完毕
    #                     choice_indexes = class_error_list[n * samples:(n + 1) * samples]  # 选择N个位置
    #                     for j , ind in enumerate(choice_indexes):
    #                         plt.subplot(numClass, samples , i * samples + j + 1)  # 子图的位置
    #                         plt.imshow(images[ind, :, :, 0], cmap='gray')  # 子图当前位置的内容
    #                         plt.axis('off')  # 子图坐标轴关闭
    #                         plt.title(c + ':' + classes[softmax_list[ind]], fontsize='x-small')  # 每个子图的标题
    #                 else:
    #                     finish_list[i] = 1
    #             if (np.sum(finish_list) == numClass):  #  所有分类的错误都显示完毕后退出
    #                 break
    #             plt.ioff()
    #             plt.show()
    return argmax_list, result_list.astype('int32')


def Save_Predict_Txt(filePath, predict_info, predict_softmax):
    label_list = predict_info[:, 2].astype('int32')
    argmax_list = np.argmax(predict_softmax, 1)  # 将结果去one hot化
    result_list = np.equal(label_list, argmax_list)  # 对比结果是否正确
    predict_info = predict_info.astype('str')
    label_score = predict_softmax[np.arange(predict_softmax.shape[0]), label_list]
    predict_score = np.max(predict_softmax, axis=1)

    with open(filePath, "w", newline="") as datacsv:
        # dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
        csvwriter = csv.writer(datacsv, dialect=("excel"), delimiter="\t")
        csvwriter.writerow(['FileName', 'ClassName', 'Label', 'MMT_Judge', 'AI_Label', 'Equal', 'Comment'])
        for i in range(predict_info.shape[0]):
            # csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
            csvwriter.writerow(
                [predict_info[i, 0], predict_info[i, 1], predict_info[i, 2], predict_info[i, 3], argmax_list[i],
                 result_list[i],
                 '%.3f' % label_score[i] + ',' + '%.3f' % predict_score[i]])


def Show_OK_NG(images, labels, softmax, classes, class_num, OK_NG=False):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)

    label_list = np.argmax(labels, 1)  # 将结果one hot化
    softmax_list = np.argmax(softmax, 1)  # 将结果one hot化
    result_list = np.equal(label_list, softmax_list)  # 对比结果是否正确

    error_list = np.flatnonzero(result_list == OK_NG)  # 挑选全部分类错误的index
    print('Show_Error', classes[class_num], class_num)
    # 打印每个分类的正确率和错误数量
    index_list = np.flatnonzero(label_list == class_num)  # flatnonzero取出为不为零的位置，挑选当前分类的index
    class_error_list = np.intersect1d(index_list, error_list)  # 挑选当前分类错误的index

    for i, error_index in enumerate(class_error_list):
        print(i, classes[class_num] + ':' + classes[softmax_list[error_index]], softmax[error_index, :])
        plt.imshow(images[error_index, :, :, 0], cmap='gray')  # 子图当前位置的内容
        plt.axis('off')  # 子图坐标轴关闭
        plt.title(classes[class_num] + ':' + classes[softmax_list[error_index]])  # 每个子图的标题 , fontsize='x-small'
        plt.ioff()
        plt.show()
