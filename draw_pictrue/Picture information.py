# 论文表格选择的8张训练集图片在训练集中的编号
# list_i = [10, 39, 177, 189, 216, 290, 929, 983]

import numpy as np
from tensorflow.keras.models import load_model
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


images = np.load("测试集8张图片.npy")
ture_images = images[:, :, :, :25]
bi_images = images[:, :, :, 25:50]
model = load_model('E:/基于深度学习的高光谱图像去马赛克方法研究/网络模型/单输入5x5滤波阵列/model_无噪声.h5')
pred_images = model.predict(bi_images)

for i in range(len(images)):
    PSNR_list = list()
    SSIM_list = list()
    PSNR_images = 0
    SSIM_images = 0
    list_i = [0, 4, 8, 12, 16, 20, 24]
    for n in range(25):
        image1 = ture_images[i, :, :, n]
        image2 = pred_images[i, :, :, n]
        # image2 = bi_images[i, :, :, n]
        PSNR_image = peak_signal_noise_ratio(image1, image2, data_range=1)
        SSIM_image = structural_similarity(image1, image2, data_range=1)
        PSNR_images += PSNR_image
        SSIM_images += SSIM_image
        if n in list_i:
            PSNR_list.append(PSNR_image)
            SSIM_list.append(SSIM_image)
        else:
            pass
    print('PSNR_list_{}'.format(i), PSNR_list)
    print('PSNR_{} :'.format(i), PSNR_images / 25)
    print('SSIM_list_{}'.format(i), SSIM_list)
    print('SSIM_{} :'.format(i), SSIM_images / 25)
