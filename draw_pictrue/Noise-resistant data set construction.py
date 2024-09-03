import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.io import savemat
import os
"""
测试集制作
来源 CAVE高光谱数据库  D:\\高光谱数据库\\3.CAVE 取前31个数据 共32个 取前25通道 共31通道

步骤如下：
把全分辨的图像存为mat形式 一个mat是512*512*25
归一化 加固定高斯噪声（不同倍数） 马赛克 插值
"""


"""测试集构建"""


# 高斯噪声
def add_noise(in_image, n, var, arrsize):
    out = np.zeros((arrsize, arrsize, 25), dtype=np.float32)
    noise = np.random.normal(n, var ** 0.5, (arrsize, arrsize))
    # print(in_image.shape)
    for z in range(0, 25):
        out[:, :, z] = in_image[:, :, z] + noise
    print(np.max(out))
    out = np.clip(out, 0, 255.0)
    print(np.max(out))
    return out


def buildtestdata_512():
    # 随机噪声
    # var = np.random.uniform(0, 0.01)

    # 加固定噪声
    # var1 = 0
    # var1 = 0.000625
    # var1 = 0.0025
    # var1 = 0.005625
    # var1 = 0.01

    a = 0  # 数据集划分
    b = 0  # 数据集划分 D:\bella\datasets\test512\GT512_460-700_mat

    datanum = 620
    for i in range(0, datanum):
        path = 'D:\\highspectrums-datasets\\3.CAVE\\GT_512_mat\\{}.mat'.format(i)
        original_data = loadmat(path)['gtdata']
        # print(original_data)
        # max = np.max(original_data)
        # print(i, 'max', max)
        # img = original_data / max * 255
        img_noise = original_data
        # img_noise = add_noise(original_data, 0.1, var1, 512)

        img_mosaic = np.zeros((512, 512, 25), dtype=np.float32)
        img_chazhi = np.zeros((512, 512, 25), dtype=np.float32)
        for x in range(0, 512, 5):
            for y in range(0, 512, 5):
                img_mosaic[x, y, 0] = img_noise[x, y, 0]
        for x in range(1, 512, 5):
            for y in range(0, 512, 5):
                img_mosaic[x, y, 1] = img_noise[x, y, 1]
        for x in range(2, 512, 5):
            for y in range(0, 512, 5):
                img_mosaic[x, y, 2] = img_noise[x, y, 2]
        for x in range(3, 512, 5):
            for y in range(0, 512, 5):
                img_mosaic[x, y, 3] = img_noise[x, y, 3]
        for x in range(4, 512, 5):
            for y in range(0, 512, 5):
                img_mosaic[x, y, 4] = img_noise[x, y, 4]
        for x in range(0, 512, 5):
            for y in range(1, 512, 5):
                img_mosaic[x, y, 5] = img_noise[x, y, 5]
        for x in range(1, 512, 5):
            for y in range(1, 512, 5):
                img_mosaic[x, y, 6] = img_noise[x, y, 6]
        for x in range(2, 512, 5):
            for y in range(1, 512, 5):
                img_mosaic[x, y, 7] = img_noise[x, y, 7]
        for x in range(3, 512, 5):
            for y in range(1, 512, 5):
                img_mosaic[x, y, 8] = img_noise[x, y, 8]
        for x in range(4, 512, 5):
            for y in range(1, 512, 5):
                img_mosaic[x, y, 9] = img_noise[x, y, 9]
        for x in range(0, 512, 5):
            for y in range(2, 512, 5):
                img_mosaic[x, y, 10] = img_noise[x, y, 10]
        for x in range(1, 512, 5):
            for y in range(2, 512, 5):
                img_mosaic[x, y, 11] = img_noise[x, y, 11]
        for x in range(2, 512, 5):
            for y in range(2, 512, 5):
                img_mosaic[x, y, 12] = img_noise[x, y, 12]
        for x in range(3, 512, 5):
            for y in range(2, 512, 5):
                img_mosaic[x, y, 13] = img_noise[x, y, 13]
        for x in range(4, 512, 5):
            for y in range(2, 512, 5):
                img_mosaic[x, y, 14] = img_noise[x, y, 14]
        for x in range(0, 512, 5):
            for y in range(3, 512, 5):
                img_mosaic[x, y, 15] = img_noise[x, y, 15]
        for x in range(1, 512, 5):
            for y in range(3, 512, 5):
                img_mosaic[x, y, 16] = img_noise[x, y, 16]
        for x in range(2, 512, 5):
            for y in range(3, 512, 5):
                img_mosaic[x, y, 17] = img_noise[x, y, 17]
        for x in range(3, 512, 5):
            for y in range(3, 512, 5):
                img_mosaic[x, y, 18] = img_noise[x, y, 18]
        for x in range(4, 512, 5):
            for y in range(3, 512, 5):
                img_mosaic[x, y, 19] = img_noise[x, y, 19]
        for x in range(0, 512, 5):
            for y in range(4, 512, 5):
                img_mosaic[x, y, 20] = img_noise[x, y, 20]
        for x in range(1, 512, 5):
            for y in range(4, 512, 5):
                img_mosaic[x, y, 21] = img_noise[x, y, 21]
        for x in range(2, 512, 5):
            for y in range(4, 512, 5):
                img_mosaic[x, y, 22] = img_noise[x, y, 22]
        for x in range(3, 512, 5):
            for y in range(4, 512, 5):
                img_mosaic[x, y, 23] = img_noise[x, y, 23]
        for x in range(4, 512, 5):
            for y in range(4, 512, 5):
                img_mosaic[x, y, 24] = img_noise[x, y, 24]

        HH = [[1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15],
              [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
              [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
              [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
              [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5],
              [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
              [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
              [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
              [1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15]]

        for n in range(25):
            img_chazhi[:, :, n] = signal.convolve2d(img_mosaic[:, :, n], HH, 'same')

        """分批做训练集、验证集"""
        # if i in range(1, 6346, 4):
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/验证集/2/{}'.format(a), img_mosaic)
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/验证集/1/{}'.format(a), img_noise)
        #     a += 1
        # else:
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/训练集/2/{}'.format(b), img_mosaic)
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/训练集/1/{}'.format(b), img_noise)
        #     b += 1
        """测试集"""
        # img_noise是加噪声的马赛克
        # np.save('D:\\高光谱数据库\\2\\GT_512_npy_noise\\{}'.format(i), img_noise)
        # np.save('D:\\高光谱数据库\\2\\LR_512\\{}'.format(i), img_mosaic)
        savepath = r'D:\\highspectrums-datasets\\3.CAVE\\LR_512_mat\\\\{}.mat'.format(i)
        print(savepath)
        savemat(savepath, {"testdata": img_chazhi})



def buildtestdata_128():
    # 随机噪声
    # var = np.random.uniform(0, 0.01)

    # 加固定噪声
    # var1 = 0
    # var1 = 0.000625
    # var1 = 0.0025
    # var1 = 0.005625
    var1 = 0.01

    a = 0  # 数据集划分
    b = 0  # 数据集划分 D:\bella\datasets\test512\GT512_460-700_mat

    datanum = 31
    for i in range(0, datanum):
        path = 'D:\\高光谱数据库\\2\\GT_512\\{}.mat'.format(i)
        original_data = loadmat(path)['gtdata']
        print(original_data)
        # max = np.max(original_data)
        # print(i, 'max', max)
        img = original_data / 255
        # img_noise = img
        img_noise = add_noise(img, 0.1, var1)

        img_mosaic = np.zeros((128, 128, 25), dtype=np.float32)
        img_chazhi = np.zeros((128, 128, 25), dtype=np.float32)
        for x in range(0, 128, 5):
            for y in range(0, 128, 5):
                img_mosaic[x, y, 0] = img_noise[x, y, 0]
        for x in range(1, 128, 5):
            for y in range(0, 128, 5):
                img_mosaic[x, y, 1] = img_noise[x, y, 1]
        for x in range(2, 128, 5):
            for y in range(0, 128, 5):
                img_mosaic[x, y, 2] = img_noise[x, y, 2]
        for x in range(3, 128, 5):
            for y in range(0, 128, 5):
                img_mosaic[x, y, 3] = img_noise[x, y, 3]
        for x in range(4, 128, 5):
            for y in range(0, 128, 5):
                img_mosaic[x, y, 4] = img_noise[x, y, 4]
        for x in range(0, 128, 5):
            for y in range(1, 128, 5):
                img_mosaic[x, y, 5] = img_noise[x, y, 5]
        for x in range(1, 128, 5):
            for y in range(1, 128, 5):
                img_mosaic[x, y, 6] = img_noise[x, y, 6]
        for x in range(2, 128, 5):
            for y in range(1, 128, 5):
                img_mosaic[x, y, 7] = img_noise[x, y, 7]
        for x in range(3, 128, 5):
            for y in range(1, 128, 5):
                img_mosaic[x, y, 8] = img_noise[x, y, 8]
        for x in range(4, 128, 5):
            for y in range(1, 128, 5):
                img_mosaic[x, y, 9] = img_noise[x, y, 9]
        for x in range(0, 128, 5):
            for y in range(2, 128, 5):
                img_mosaic[x, y, 10] = img_noise[x, y, 10]
        for x in range(1, 128, 5):
            for y in range(2, 128, 5):
                img_mosaic[x, y, 11] = img_noise[x, y, 11]
        for x in range(2, 128, 5):
            for y in range(2, 128, 5):
                img_mosaic[x, y, 12] = img_noise[x, y, 12]
        for x in range(3, 128, 5):
            for y in range(2, 128, 5):
                img_mosaic[x, y, 13] = img_noise[x, y, 13]
        for x in range(4, 128, 5):
            for y in range(2, 128, 5):
                img_mosaic[x, y, 14] = img_noise[x, y, 14]
        for x in range(0, 128, 5):
            for y in range(3, 128, 5):
                img_mosaic[x, y, 15] = img_noise[x, y, 15]
        for x in range(1, 128, 5):
            for y in range(3, 128, 5):
                img_mosaic[x, y, 16] = img_noise[x, y, 16]
        for x in range(2, 128, 5):
            for y in range(3, 128, 5):
                img_mosaic[x, y, 17] = img_noise[x, y, 17]
        for x in range(3, 128, 5):
            for y in range(3, 128, 5):
                img_mosaic[x, y, 18] = img_noise[x, y, 18]
        for x in range(4, 128, 5):
            for y in range(3, 128, 5):
                img_mosaic[x, y, 19] = img_noise[x, y, 19]
        for x in range(0, 128, 5):
            for y in range(4, 128, 5):
                img_mosaic[x, y, 20] = img_noise[x, y, 20]
        for x in range(1, 128, 5):
            for y in range(4, 128, 5):
                img_mosaic[x, y, 21] = img_noise[x, y, 21]
        for x in range(2, 128, 5):
            for y in range(4, 128, 5):
                img_mosaic[x, y, 22] = img_noise[x, y, 22]
        for x in range(3, 128, 5):
            for y in range(4, 128, 5):
                img_mosaic[x, y, 23] = img_noise[x, y, 23]
        for x in range(4, 128, 5):
            for y in range(4, 128, 5):
                img_mosaic[x, y, 24] = img_noise[x, y, 24]

        HH = [[1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15],
              [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
              [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
              [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
              [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5],
              [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
              [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
              [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
              [1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15]]

        for n in range(25):
            img_chazhi[:, :, n] = signal.convolve2d(img_mosaic[:, :, n], HH, 'same')

        """分批做训练集、验证集"""
        # if i in range(1, 6346, 4):
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/验证集/2/{}'.format(a), img_mosaic)
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/验证集/1/{}'.format(a), img_noise)
        #     a += 1
        # else:
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/训练集/2/{}'.format(b), img_mosaic)
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/训练集/1/{}'.format(b), img_noise)
        #     b += 1
        """测试集"""
        # img_noise是加噪声的马赛克
        # np.save('D:\\高光谱数据库\\2\\GT_512_npy_noise\\{}'.format(i), img_noise)
        # np.save('D:\\高光谱数据库\\2\\LR_512\\{}'.format(i), img_mosaic)  D:\bella\datasets\test128\460-700\LR128_mat_0.025
        savepath = 'D:\\bella\\datasets\\test128\\460-700\\LR128_mat_0\\{}.mat'.format(i)
        savemat(savepath, {"testdata": img_chazhi})
        print(savepath)


def buildtestdata(path, arr_size, savepath):
    # 随机噪声
    # var = np.random.uniform(0, 0.01)

    # 加固定噪声
    # var1 = 0
    # var1 = 0.000625
    # var1 = 0.0025
    var1 = 0.005625
    # var1 = 0.01

    a = 0  # 数据集划分
    b = 0  # 数据集划分 D:\BasicSR\datasets\test100\GT100_mat_400-640

    datanum = len(os.listdir(path))
    # datanum = 10
    # print(arrsize, datanum)
    for i in range(0, datanum):
    # for i in range(7, 8):
        matname = '{}.mat'.format(i)
        mat_path = os.path.join(path, matname)
        original_data = loadmat(mat_path)['gtdata']
        # print(original_data)
        max = np.max(original_data)
        print(i, 'max', max)
        img = original_data / 63238
        img_noise = img
        # img_noise = add_noise(img, 0.1, var1, arr_size)

        img_mosaic = np.zeros((arr_size, arr_size, 25), dtype=np.float32)
        img_chazhi = np.zeros((arr_size, arr_size, 25), dtype=np.float32)
        for x in range(0, arr_size, 5):
            for y in range(0, arr_size, 5):
                img_mosaic[x, y, 0] = img_noise[x, y, 0]
        for x in range(1, arr_size, 5):
            for y in range(0, arr_size, 5):
                img_mosaic[x, y, 1] = img_noise[x, y, 1]
        for x in range(2, arr_size, 5):
            for y in range(0, arr_size, 5):
                img_mosaic[x, y, 2] = img_noise[x, y, 2]
        for x in range(3, arr_size, 5):
            for y in range(0, arr_size, 5):
                img_mosaic[x, y, 3] = img_noise[x, y, 3]
        for x in range(4, arr_size, 5):
            for y in range(0, arr_size, 5):
                img_mosaic[x, y, 4] = img_noise[x, y, 4]
        for x in range(0, arr_size, 5):
            for y in range(1, arr_size, 5):
                img_mosaic[x, y, 5] = img_noise[x, y, 5]
        for x in range(1, arr_size, 5):
            for y in range(1, arr_size, 5):
                img_mosaic[x, y, 6] = img_noise[x, y, 6]
        for x in range(2, arr_size, 5):
            for y in range(1, arr_size, 5):
                img_mosaic[x, y, 7] = img_noise[x, y, 7]
        for x in range(3, arr_size, 5):
            for y in range(1, arr_size, 5):
                img_mosaic[x, y, 8] = img_noise[x, y, 8]
        for x in range(4, arr_size, 5):
            for y in range(1, arr_size, 5):
                img_mosaic[x, y, 9] = img_noise[x, y, 9]
        for x in range(0, arr_size, 5):
            for y in range(2, arr_size, 5):
                img_mosaic[x, y, 10] = img_noise[x, y, 10]
        for x in range(1, arr_size, 5):
            for y in range(2, arr_size, 5):
                img_mosaic[x, y, 11] = img_noise[x, y, 11]
        for x in range(2, arr_size, 5):
            for y in range(2, arr_size, 5):
                img_mosaic[x, y, 12] = img_noise[x, y, 12]
        for x in range(3, arr_size, 5):
            for y in range(2, arr_size, 5):
                img_mosaic[x, y, 13] = img_noise[x, y, 13]
        for x in range(4, arr_size, 5):
            for y in range(2, arr_size, 5):
                img_mosaic[x, y, 14] = img_noise[x, y, 14]
        for x in range(0, arr_size, 5):
            for y in range(3, arr_size, 5):
                img_mosaic[x, y, 15] = img_noise[x, y, 15]
        for x in range(1, arr_size, 5):
            for y in range(3, arr_size, 5):
                img_mosaic[x, y, 16] = img_noise[x, y, 16]
        for x in range(2, arr_size, 5):
            for y in range(3, arr_size, 5):
                img_mosaic[x, y, 17] = img_noise[x, y, 17]
        for x in range(3, arr_size, 5):
            for y in range(3, arr_size, 5):
                img_mosaic[x, y, 18] = img_noise[x, y, 18]
        for x in range(4, arr_size, 5):
            for y in range(3, arr_size, 5):
                img_mosaic[x, y, 19] = img_noise[x, y, 19]
        for x in range(0, arr_size, 5):
            for y in range(4, arr_size, 5):
                img_mosaic[x, y, 20] = img_noise[x, y, 20]
        for x in range(1, arr_size, 5):
            for y in range(4, arr_size, 5):
                img_mosaic[x, y, 21] = img_noise[x, y, 21]
        for x in range(2, arr_size, 5):
            for y in range(4, arr_size, 5):
                img_mosaic[x, y, 22] = img_noise[x, y, 22]
        for x in range(3, arr_size, 5):
            for y in range(4, arr_size, 5):
                img_mosaic[x, y, 23] = img_noise[x, y, 23]
        for x in range(4, arr_size, 5):
            for y in range(4, arr_size, 5):
                img_mosaic[x, y, 24] = img_noise[x, y, 24]

        HH = [[1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15],
              [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
              [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
              [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
              [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5],
              [1 / 5, 4 / 15, 2 / 5, 8 / 15, 4 / 5, 8 / 15, 2 / 5, 4 / 15, 1 / 5],
              [1 / 5, 4 / 15, 1 / 3, 2 / 5, 3 / 5, 2 / 5, 1 / 3, 4 / 15, 1 / 5],
              [2 / 15, 2 / 15, 4 / 15, 4 / 15, 2 / 5, 4 / 15, 4 / 15, 2 / 15, 2 / 15],
              [1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15]]

        for n in range(25):
            img_chazhi[:, :, n] = signal.convolve2d(img_mosaic[:, :, n], HH, 'same')

        """分批做训练集、验证集"""
        # if i in range(1, 6346, 4):
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/验证集/2/{}'.format(a), img_mosaic)
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/验证集/1/{}'.format(a), img_noise)
        #     a += 1
        # else:
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/训练集/2/{}'.format(b), img_mosaic)
        #     # np.save('D:/近红外高光谱图像去马赛克/数据集/训练集/1/{}'.format(b), img_noise)
        #     b += 1
        """测试集"""
        # img_noise是加噪声的马赛克
        # np.save('D:\\高光谱数据库\\2\\GT_512_npy_noise\\{}'.format(i), img_noise)
        # np.save('D:\\高光谱数据库\\2\\LR_512\\{}'.format(i), img_mosaic)  D:\bella\datasets\test128\460-700\LR128_mat_0.025
        newmat_savepath = os.path.join(savepath, '{}.mat'.format(i))
        # newmat_savepath = os.path.join(savepath, '007_0.075.mat')
        print(i, newmat_savepath)
        savemat(newmat_savepath, {"mosaicdata": img_mosaic})


if __name__ == '__main__':
    GTdata = r'D:\BasicSR\datasets\test512\GT512_460-700_mat'
    LRdata = r'D:\BasicSR\datasets\test512\mosaicimg\460-700'
    targetsize = 512
    # ['gtdata']
    buildtestdata_512()
    # buildtestdata(GTdata, targetsize, LRdata)