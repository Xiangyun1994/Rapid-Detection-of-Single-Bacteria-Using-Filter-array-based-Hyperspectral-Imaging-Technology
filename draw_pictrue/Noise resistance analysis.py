from tensorflow.keras.models import load_model
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cv2
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Mosaic_train = np.load('E:/基于深度学习的高光谱图像去马赛克方法研究/网络模型/单输入5x5滤波阵列/数据集（有噪声）/测试集/测试集-线性插值图像-标准差为0.npy')
# model = load_model('E:/基于深度学习的高光谱图像去马赛克方法研究/网络模型/单输入5x5滤波阵列/model_有噪声.h5')
# results = model.predict(Mosaic_train)
# np.save("predict_数据集（标准差为0）", results)


def add_noise(in_image, n, var):
    out = np.zeros((200, 200, 1), dtype=np.float32)
    noise = np.random.normal(n, var ** 0.5, (200, 200))
    out[:, :, 0] = in_image[:, :] + noise
    out = np.clip(out, 0, 1.0)
    return out
#
#
# image = np.load('论文2张展示图片.npy')
# print(image.shape)
# Mosaic_images = np.zeros((2, 200, 200, 1), dtype=np.float32)
# images = np.zeros((2, 200, 200, 25), dtype=np.float32)
# for i in range(len(image)):
#     Mosaic_image = image[i, :, :, 50]
#
#     var = 0
#     # var = 0.000625
#     # var = 0.0025
#     # var = 0.005625
#     # var = 0.01
#
#     # Mosaic_image = add_noise(Mosaic_image, 0.1, var)
#     Mosaic_image = add_noise(Mosaic_image, 0, var)
#
#     Mosaic_images[i, :, :, 0] = Mosaic_image[:, :, 0]
#
#     for x in range(0, 200, 5):
#         for y in range(0, 200, 5):
#             images[i, x, y, 0] = Mosaic_image[x, y, 0]
#     for x in range(1, 200, 5):
#         for y in range(0, 200, 5):
#             images[i, x, y, 1] = Mosaic_image[x, y, 0]
#     for x in range(2, 200, 5):
#         for y in range(0, 200, 5):
#             images[i, x, y, 2] = Mosaic_image[x, y, 0]
#     for x in range(3, 200, 5):
#         for y in range(0, 200, 5):
#             images[i, x, y, 3] = Mosaic_image[x, y, 0]
#     for x in range(4, 200, 5):
#         for y in range(0, 200, 5):
#             images[i, x, y, 4] = Mosaic_image[x, y, 0]
#     for x in range(0, 200, 5):
#         for y in range(1, 200, 5):
#             images[i, x, y, 5] = Mosaic_image[x, y, 0]
#     for x in range(1, 200, 5):
#         for y in range(1, 200, 5):
#             images[i, x, y, 6] = Mosaic_image[x, y, 0]
#     for x in range(2, 200, 5):
#         for y in range(1, 200, 5):
#             images[i, x, y, 7] = Mosaic_image[x, y, 0]
#     for x in range(3, 200, 5):
#         for y in range(1, 200, 5):
#             images[i, x, y, 8] = Mosaic_image[x, y, 0]
#     for x in range(4, 200, 5):
#         for y in range(1, 200, 5):
#             images[i, x, y, 9] = Mosaic_image[x, y, 0]
#     for x in range(0, 200, 5):
#         for y in range(2, 200, 5):
#             images[i, x, y, 10] = Mosaic_image[x, y, 0]
#     for x in range(1, 200, 5):
#         for y in range(2, 200, 5):
#             images[i, x, y, 11] = Mosaic_image[x, y, 0]
#     for x in range(2, 200, 5):
#         for y in range(2, 200, 5):
#             images[i, x, y, 12] = Mosaic_image[x, y, 0]
#     for x in range(3, 200, 5):
#         for y in range(2, 200, 5):
#             images[i, x, y, 13] = Mosaic_image[x, y, 0]
#     for x in range(4, 200, 5):
#         for y in range(2, 200, 5):
#             images[i, x, y, 14] = Mosaic_image[x, y, 0]
#     for x in range(0, 200, 5):
#         for y in range(3, 200, 5):
#             images[i, x, y, 15] = Mosaic_image[x, y, 0]
#     for x in range(1, 200, 5):
#         for y in range(3, 200, 5):
#             images[i, x, y, 16] = Mosaic_image[x, y, 0]
#     for x in range(2, 200, 5):
#         for y in range(3, 200, 5):
#             images[i, x, y, 17] = Mosaic_image[x, y, 0]
#     for x in range(3, 200, 5):
#         for y in range(3, 200, 5):
#             images[i, x, y, 18] = Mosaic_image[x, y, 0]
#     for x in range(4, 200, 5):
#         for y in range(3, 200, 5):
#             images[i, x, y, 19] = Mosaic_image[x, y, 0]
#     for x in range(0, 200, 5):
#         for y in range(4, 200, 5):
#             images[i, x, y, 20] = Mosaic_image[x, y, 0]
#     for x in range(1, 200, 5):
#         for y in range(4, 200, 5):
#             images[i, x, y, 21] = Mosaic_image[x, y, 0]
#     for x in range(2, 200, 5):
#         for y in range(4, 200, 5):
#             images[i, x, y, 22] = Mosaic_image[x, y, 0]
#     for x in range(3, 200, 5):
#         for y in range(4, 200, 5):
#             images[i, x, y, 23] = Mosaic_image[x, y, 0]
#     for x in range(4, 200, 5):
#         for y in range(4, 200, 5):
#             images[i, x, y, 24] = Mosaic_image[x, y, 0]
#
#     HH = [[1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15],
#           [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
#           [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
#           [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
#           [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5],
#           [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
#           [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
#           [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
#           [1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15]]
#
#     for n in range(25):
#         images[i, :, :, n] = scipy.signal.convolve2d(images[i, :, :, n], HH, 'same')
#
# np.save("论文2张展示图片-线性插值图像1-标准差为0", images)
# np.save("论文2张展示图片-马赛克图像-标准差为0", Mosaic_images)


images = np.load("论文2张展示图片.npy")
ture_images = images[:, :, :, :25]
# Mosaic_images = np.load("论文2张展示图片-马赛克图像-标准差为0.075.npy")
bi_images = np.load("论文2张展示图片-线性插值图像1-标准差为0.1.npy")
bi_images1 = np.load("论文2张展示图片-线性插值图像-标准差为0.1.npy")
model = load_model('E:/基于深度学习的高光谱图像去马赛克方法研究/网络模型/单输入5x5滤波阵列/model_有噪声.h5')
pred_images = model.predict(bi_images1)


t = 0

m = 23
n = 53

# m = 113
# n = 173

T_spectrum = ture_images[t, m, n, :25]
P_spectrum = pred_images[t, m, n, :25]
BI_spectrum = bi_images[t, m, n, :25]

# plt.title('Spectrum Analysis')
plt.rc('font', family="Times New Roman")
plt.plot(list(range(25)), T_spectrum, color='green', label='True')
plt.plot(list(range(25)), P_spectrum, color='blue', label='Ours')
plt.plot(list(range(25)), BI_spectrum, color='red', label='BI')
# plt.legend(frameon=False)
plt.xlabel('Wavelengths / nm')
plt.ylabel('light intensity')
plt.xticks([0, 4, 8, 12, 16, 20, 24, 26], ('460', '500', '540', '580', '620', '660', '700', ' '))
plt.show()


# for i in range(len(pred_images)):
#     # img = ture_images[i, :, :, 12]
#     # img *= 255 * 2
#     # img = np.clip(img, 0, 255).astype('uint8')
#     # cv2.imwrite('fig-550T_{}.png'.format(i), img)
#     img = pred_images[i, :, :, 12]
#     img *= 255 * 2
#     img = np.clip(img, 0, 255).astype('uint8')
#     cv2.imwrite('fig-550_{}.png'.format(i), img)
#     img = bi_images[i, :, :, 12]
#     img *= 255 * 2
#     img = np.clip(img, 0, 255).astype('uint8')
#     cv2.imwrite('fig-550BI_{}.png'.format(i), img)
#     img = Mosaic_images[i, :, :, 0]
#     img *= 255 * 2
#     img = np.clip(img, 0, 255).astype('uint8')
#     cv2.imwrite('fig-M_{}.png'.format(i), img)
