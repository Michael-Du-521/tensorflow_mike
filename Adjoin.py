#导入依赖库
import os
import cv2
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import tensorflow .keras
import Data_Augmentation as da # 定制模块
import Recoba_Tensorflow as rtf # 定制模块
import tensorflow.keras.optimizers

import os
from PIL import Image

#阅读TSV文件，并存入 NumPy binary 文件
def Create_Data(folder, fileName):
    result = pd.read_csv(folder + fileName, sep='\t', header=2) # 表头在该csv文件的第三行
    print(result)
    data_info = np.empty((result.shape[0], 4), dtype='S100') # 创建空NumPy 数组 data_info， result.shape[0] 查找 result array中行的个数.
    #The .shape attribute of a DataFrame returns a tuple 元组 containing the dimensions 维度 of the DataFrame
    for index, row in result.iterrows(): #通过for循环遍历result dateframe 中 的 行序号（index） 与 行数据（row）
        # result.iterrows() returns an iterator that produces pairs of index and row,
        # where the index represents the row index, and the row is a pandas Series object containing the data for that particular row.
        cell_info = np.array(
            ([row['FileName'], row['ClassName'], row['Label'], row['MMT_Judge']],))  # 添加逗号,后，cell_info是1行4列的数组
        data_info[index, :] = cell_info # The index 变量 代表 the row index, and : 代表该行的所有列.
        print(index, data_info[index, :])

    np.save(folder + "Data_Info.npy", data_info)
    return data_info


def Prepare_Data(data_info, label_array, multi_array, folder, expand_folder, resize_folder):
    labels = data_info[:, 2].astype('int32') #将data_info array中的第3列提取出来，并且将它转换为整型数值。
    # 计算每个分类的数据量
    label_array = np.sort(np.unique(labels))  # 取数据中不重复的数据 It scans the array and returns a new array containing only the unique values in sorted order.
    qty_list = [] #"qty_list" 数量列表 may stand for "quantity list"
    for i in range(len(label_array)):
        qty = np.flatnonzero(labels == label_array[i]) # the expression labels == label_array[i] compares each element in the labels array with the current label_array[i] value.
        # The result is a boolean array of the same shape as labels, where each element is True if it matches the value label_array[i], and False otherwise.
        # np.flatnonzero() 取数组作为输入并且返回在这个数组中非零（或者true）的元素 takes an array as input and returns the indices of the non-zero (or True) elements in that array.
        qty_list.append(qty.shape[0]) #qty.shape[0] 返回该qty内元素的数量 In this specific case, qty.shape[0] refers to the size of the first dimension (axis 0) of the qty array. Since qty is a 1-dimensional array, shape[0] gives the length or number of elements in that array.
    qty_array = np.array(qty_list)
    print('Classes Qty', label_array, qty_array)


    # 加载图片
    images = np.empty((labels.shape[0], 256, 256, 1), dtype='int32') #创建一个称作image的numpy数组，（标签的数量，高以像素计，宽以像素计，单通道）
    for i, info in enumerate(data_info):
        img = cv2.imread(folder + info[0].decode(), 0) #.decode（）方法用于将文件名从比特转换为字符串
        #This line uses the OpenCV library's imread function to read an image file.
        resize_image = cv2.resize(img.astype('float32'), (256, 256), cv2.INTER_LINEAR); #opencv method 期待输入图像是浮点数据类型的numpy数组
        #the code converts the input image "img" to a NumPy array of the 'float32' data type using the "astype" method.
        # This is because OpenCV's resize function expects the input image to be a NumPy array of floating point data type.
        cv2.imwrite(resize_folder + info[0].decode().replace('.jpg', '.png'), resize_image)
        images[i, :, :, 0] = resize_image # The "i" index refers to the i-th image in the dataset, and the slice "[:, :, 0]" refers to the first channel of the image.
        print('Load_Image', folder + info[0].decode())

    # 计算膨胀后的数据量
    print(qty_array)
    print(np.array(multi_array) + 1)
    expand_array = qty_array * (np.array(multi_array) + 1)
    print(expand_array)
    expand_qty = np.sum(expand_array)
    print(expand_qty, images.shape)
    # new_images = np.empty((expand_qty, images.shape[1], images.shape[2], images.shape[3]), dtype='int32')
    new_data_info = np.empty((expand_qty, data_info.shape[1]), dtype='S100') # ".shape[1]" is a Python expression that retrieves the second dimension of a NumPy array.

    for class_num in range(label_array.shape[0]):  # 区分3类，所以有3个分类
        # 获取某一分类的Index
        label = label_array[class_num]
        print('label', label)
        index_list = np.flatnonzero(labels == label)
        print('index_list', index_list)
        # 膨胀数据
        expand_images, expand_data_info = da.Data_Grow(images[index_list, :, :, :], data_info[index_list, :], multi_array[class_num])
        print(expand_images.shape, expand_data_info.shape)
        # 将膨胀后的数据填入new_images和new_data_info数组
        start_position = np.sum(expand_array[0:class_num])
        end_position = np.sum(expand_array[0:class_num + 1])
        print('expand index start:', start_position, 'end:', end_position)
        # new_images[start_position:end_position, :, :, :] = expand_images
        new_expand_train_info = expand_data_info.copy()
        j = 0
        last_fileName = ''
        for i, row_info in enumerate(expand_data_info):
            fileName = expand_data_info[i, 0].decode()
            if fileName != last_fileName:
                j = 0
            print('j', j, fileName, last_fileName)
            new_expand_train_info[i, 0] = fileName.replace('.jpg', '_' + str(j) + '.png')
            print('Expand Image Write:', i, expand_folder, new_expand_train_info[i, 0])

            expand_image = expand_images[i, :, :, :].reshape(expand_images.shape[1], expand_images.shape[2]).astype('float32')
            cv2.imwrite(expand_folder + new_expand_train_info[i, 0].decode(), expand_image)
            # if (j == 0):  # 缩小后的原图保存
            #     cv2.imwrite(resize_folder + fileName, resize_image)
            last_fileName = fileName
            j += 1
        new_data_info[start_position:end_position, :] = new_expand_train_info
    # 画一Pattern每个分类最后一张图片
    # plt.imshow(expand_images[end_position - 1, :, :, 0].astype('int32'), cmap='gray')
    # plt.imshow(cv2.imread(expand_folder + new_expand_train_info[-1, 0].decode(), 0), cmap='gray')
    # plt.show()

    # 随机打乱数据
    random_index = np.arange(new_data_info.shape[0])
    np.random.shuffle(random_index)
    print('expand_random_index:', random_index)
    # new_images = new_images[random_index]
    print('data_info_first5:', new_data_info[0:5, :])
    new_data_info = new_data_info[random_index]
    print('expand_info_first5:', new_data_info[0:5, :])
    print('expand raw:', data_info.shape)
    print('expand result:', new_data_info.shape)

    return new_data_info


def VGG_Net_layer(inputs, labels, class_qty, is_training):
    # 定义网络的超参数
    learning_rate = 5e-5
    layer = inputs
    print(layer.shape)
    print(type(layer))

    layer = rtf.conv_layer(layer, 64, is_training)
    layer = rtf.conv_layer(layer, 64, is_training)
    layer = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(layer)
    print(layer.shape)

    layer = rtf.conv_layer(layer, 128, is_training)
    layer = rtf.conv_layer(layer, 128, is_training, strides=2)
    layer = rtf.conv_layer(layer, 128, is_training)
    layer = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(layer)
    print(layer.shape)

    layer = rtf.conv_layer(layer, 256, is_training)
    layer = rtf.conv_layer(layer, 256, is_training, strides=2)
    layer = rtf.conv_layer(layer, 256, is_training)
    layer = rtf.conv_layer(layer, 256, is_training, strides=2)

    print(layer.shape)  # 本例中64,64,128,128,256,256,256,256的训练效果最好


    # 将卷积层输出扁平化处理
    orig_shape = layer.get_shape().as_list()
    layer = tf.reshape(layer, shape=[-1, orig_shape[1] * orig_shape[2] * orig_shape[3]])

    # 添加全连接层
    layer = rtf.fully_connected(layer, 256, is_training)
    layer = tf.keras.layers.Dropout(rate=0.5)(layer, training=is_training)
    layer = rtf.fully_connected(layer, 128, is_training)
    layer = tf.keras.layers.Dropout(rate=0.5)(layer, training=is_training)
    layer = rtf.fully_connected(layer, 64, is_training)

    # 为每一个类别添加一个输出节点
    # logit本身就是是一种函数，它把某个概率p从[0,1]映射到[-inf,+inf]（即正负无穷区间）。这个函数的形式化描述为：logit=ln(p/(1-p))。
    # 我们可以把logist理解为原生态的、未经缩放的，可视为一种未归一化的log 概率，如是[4, 1, -2]
    logits = tf.keras.layers.Dense(class_qty)(layer)
    # 定义loss 函数和训练操作
    # Softmax的工作则是，它把一个系列数从[-inf, +inf] 映射到[0,1]。
    # 除此之外，它还把所有参与映射的值累计之和等于1，变成诸如[0.95, 0.05, 0]的概率向量。


    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1), name='correct_prediction')
    softmax_tensor = tf.nn.softmax(logits, name='softmax_tensor')
    # 通知Tensorflow在训练时要更新均值和方差的分布
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # Use a gradient tape to compute the gradients
    with tf.GradientTape() as tape:
        # Get the TensorFlow tensor from the KerasTensor object
        inputs_tensor = tf.convert_to_tensor(inputs)
        # Compute the loss
        model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),name='model_loss')
        # Convert the KerasTensor objects to TensorFlow tensors
        trainable_vars = [tf.convert_to_tensor(v) for v in tf.compat.v1.trainable_variables()]
        # Compute the gradients using the gradient tape
        grads = tape.gradient(model_loss, trainable_vars)
        # Use the gradients to update the trainable variables
        train_opt = optimizer.apply_gradients(zip(grads, trainable_vars))

    return train_opt, model_loss, correct_prediction, softmax_tensor


# 定义训练必要的参数
expand_folder = "D:/AI_Sample_for_original_06222023/Adjoin/Expand/" #在这里修改目录
folder = "D:/AI_Sample_for_original_06222023/Adjoin/" #在这里修改目录
resize_folder = 'D:/AI_Sample_for_original_06222023/Adjoin/Resize/' #在这里修改目录
fileName = 'Result.txt'
pb_fileName = 'Adjoin_Model'
multi_array = [5, 5, 20]
label_array = np.array([0, 1, 2])
classes = ['Panel', 'Space', 'Abnormal']
class_qty = len(classes)
channels = 1
height = 256
width = 256
#batch_size = 32  # recoba批处理数量
batch_size = 4 # 批处理数量
#loop_n = 20  # recoba 训练次数
loop_n = 10  # miguel 训练次数

# test_info = Create_Data(folder, fileName)
# expand_train_info = Prepare_Data(test_info, label_array, multi_array, folder, expand_folder, resize_folder)
# np.save(folder + "expand_train_info.npy", expand_train_info)

# 加载扩增后的数据
expand_train_info = np.load(folder + "expand_train_info.npy")
test_info = np.load(folder + "Data_Info.npy")

# 创建输入样本和标签的占位符 using tf.keras.Input
inputs = tf.keras.Input(shape=(height, width, channels), dtype=tf.float32, name='inputs')
labels = tf.keras.Input(shape=(class_qty,), dtype=tf.float32, name='labels')
is_training = tf.keras.Input(shape=(), dtype=tf.bool, name='is_training')
# 构建神经网络
train_opt, model_loss, correct_prediction, softmax_tensor = VGG_Net_layer(inputs, labels, class_qty, is_training)
# 进行训练、验证和测试
vali_softmax = rtf.Train(expand_folder, resize_folder, expand_train_info, test_info, class_qty, height, width, train_opt, model_loss,
                         correct_prediction, softmax_tensor,
                         inputs, labels, is_training, batch_size=batch_size, loop_n=loop_n,
                         pb_file_folder=folder + pb_fileName)

rtf.Show_Result(test_info, vali_softmax, classes)

# 使用pb文件进行预测

#Recoba原语句:
# vali_softmax = rtf.Load_Model_Predict(resize_folder, test_info, class_qty, height, width, batch_size=batch_size, predict_name='Load Model Predict',
#                                    pb_file_path=folder + "Adjoin_Model_13_0.9858446052217679.pb")

# Migule修改语句:
# vali_softmax = rtf.Load_Model_Predict(resize_folder, test_info, class_qty, height, width, batch_size=batch_size, predict_name='Load Model Predict',
#                                       pb_file_path=folder + "230422_Adjoin_Model.pb")

# rtf.Show_Result(test_info, vali_softmax, classes)
# rtf.Save_Predict_Txt(folder + "Predict.txt", test_info, vali_softmax)
