import cv2
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt


# from keras.preprocessing.image import ImageDataGenerator


# datagen = ImageDataGenerator(featurewise_center=False,
#                             samplewise_center=False,
#                             featurewise_std_normalization=False,
#                             samplewise_std_normalization=False,
#                             zca_whitening=False,
#                             zca_epsilon=1e-06,
#                             rotation_range=0.0,
#                             width_shift_range=0.0,
#                             height_shift_range=0.0,
#                             brightness_range=None,
#                             shear_range=0.0,
#                             zoom_range=0.0,
#                             channel_shift_range=0.0,
#                             fill_mode='nearest',
#                             cval=0.0,
#                             horizontal_flip=False,
#                             vertical_flip=False,
#                             rescale=None,
#                             preprocessing_function=None,
#                             data_format=None,
#                             validation_split=0.0)

def Contrast_and_Brightness(img, min_contrast, max_contrast, min_brightness, max_brightness):
    blank = np.zeros(img.shape, img.dtype)
    alpha = np.random.uniform(min_contrast, max_contrast)
    beta = np.random.uniform(min_brightness, max_brightness)
    #     print(alpha, beta)
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta).reshape(img.shape)  # .astype('int32')
    dst[dst < 0] = 0
    dst[dst > 255] = 255
    return dst


# def resize_rate_image(raw_img,resize_rate):
#     resize_image=cv2.resize(raw_img, (0, 0), resize_rate, resize_rate, cv2.INTER_AREA);
#     return resize_image

def Resize_Rotate_Image(raw_img, resize_rate=1, offset_degree=0):
    # 通用写法，即使传入的是三通道图片依然不会出错
    height, width = raw_img.shape[:2]
    center = (width // 2, height // 2)
    # 得到旋转矩阵，第一个参数为旋转中心，第二个参数为旋转角度，第三个参数为旋转之前原图像缩放比例
    M = cv2.getRotationMatrix2D(center, -offset_degree, resize_rate)
    # 进行仿射变换，第一个参数图像，第二个参数是旋转矩阵，第三个参数是变换之后的图像大小
    resize_rotate_image = cv2.warpAffine(raw_img.astype('float32'), M, (width, height))
    return resize_rotate_image


def Shift_Image(raw_img, shift_x, shift_y):
    height, width = raw_img.shape[:2]
    translation_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    shift_image = cv2.warpAffine(raw_img, translation_matrix, (width, height))
    return shift_image


def Clip_Image(raw_img, center_row, center_col, half_size):
    clip_image = raw_img[center_row - half_size:center_row + half_size, center_col - half_size:center_col + half_size]
    return clip_image


def Flip_Image(raw_img, is_horizontal_flip, is_vertical_flip, rotate_degree):
    flip_image = raw_img.copy()
    height, width = flip_image.shape[:2]
    center = (width // 2, height // 2)
    if (is_horizontal_flip):
        flip_image = cv2.flip(flip_image, 1)  # 水平翻转
    if (is_vertical_flip):
        flip_image = cv2.flip(flip_image, 0)  # 垂直翻转
    M = cv2.getRotationMatrix2D(center, rotate_degree, 1.0)
    flip_image = cv2.warpAffine(flip_image, M, (width, height))  # 旋转0,90,270度
    return flip_image


def Data_Grow(raw_img_list, data_info, expand_rate):
    half_size = 128
    expand_images = np.empty((raw_img_list.shape[0] * (expand_rate + 1), half_size * 2, half_size * 2, raw_img_list.shape[3]), dtype='int32')
    expand_data_info = np.empty((data_info.shape[0] * (expand_rate + 1), data_info.shape[1]), dtype='S100')
    print('expand_rate:', expand_rate)

    for n in range(raw_img_list.shape[0]):  # images.shape[0]
        print('data_info:', data_info[n])
        expand_images[n * (expand_rate + 1), :, :, :] = raw_img_list[n, raw_img_list.shape[1] // 2 - half_size:raw_img_list.shape[1] // 2 + half_size, raw_img_list.shape[2] // 2 - half_size:raw_img_list.shape[2] // 2 + half_size, :]
        expand_data_info[n * (expand_rate + 1), :] = data_info[n, :]
        print('index=', n * (expand_rate + 1), 'n=', n, 'i=', 0)

        for i in range(1, expand_rate + 1):
            raw_img = raw_img_list[n:n + 1, :, :, :].reshape(raw_img_list.shape[1], raw_img_list.shape[2])
            print("raw_img", raw_img.shape)
            rows, cols = raw_img.shape[:2]
            resize_rate = np.random.uniform(0.8, 1.2)
            offset_degree = np.random.uniform(-10, 10)
            shift_x = cols * np.random.uniform(-0.1, 0.1)
            shift_y = rows * np.random.uniform(-0.1, 0.1)
            center_row = int(cols / 2 + cols * np.random.uniform(-0.05, 0.05))
            center_col = int(rows / 2 + rows * np.random.uniform(-0.05, 0.05))
            is_horizontal_flip = np.random.choice([True, False])
            is_vertical_flip = np.random.choice([True, False])
            rotate_degree = np.random.choice([0, 90, 270])

            grow_img = raw_img.copy()
            grow_img = Resize_Rotate_Image(grow_img, resize_rate, offset_degree)
            grow_img = Shift_Image(grow_img, shift_x, shift_y)
            # grow_img = Clip_Image(grow_img, center_row, center_col, half_size)
            grow_img = Flip_Image(grow_img, is_horizontal_flip, is_vertical_flip, rotate_degree)
            grow_img = Contrast_and_Brightness(grow_img, 0.9, 1.1, -40, 40)

            expand_images[n * (expand_rate + 1) + i, :, :, :] = grow_img.reshape(grow_img.shape[0], grow_img.shape[1], 1)
            expand_data_info[n * (expand_rate + 1) + i, :] = data_info[n, :]
            print('index=', n * (expand_rate + 1) + i, 'n=', n, 'i=', i)
            # 画一Pattern图
            # plt.imshow(expand_images[n * (expand_rate + 1) + i, :, :, 0], cmap='gray')  # .astype('int32')
            # plt.show()

    return expand_images, expand_data_info

# img = cv2.imread("D:/11.bmp", cv2.IMREAD_UNCHANGED)
# print("img", img.shape)

# data_info = []
# raw_img = cv2.imread("D:/Desiccant10.png", cv2.IMREAD_UNCHANGED)
# raw_img_list = raw_img.reshape((1, 1001, 1001, 1))
# # print("raw_img", raw_img_list.shape)
# Data_Grow(raw_img_list, data_info, 20)

# cv2.imwrite("D:\Clip_Desiccant10.png", clip_image)
