import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cv2
import os
from scipy.io import loadmat

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 单张图片加噪声
def add_noise(in_image, n, var):
    out = np.zeros((1024, 1024, 1), dtype=np.float32)
    noise = np.random.normal(n, var ** 0.5, (1024, 1024))
    out[:, :, 0] = in_image[:, :] + noise
    out = np.clip(out, 0, 1.0)
    return out


# 多个文件夹下的图片重命名
def ReNameforfile(filepath, new_path):
    folder_list = os.listdir(filepath)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    print(folder_list)
    for i in range(0, 31):
        folder_full_name = os.path.join(filepath, folder_list[i])  # D:\BasicSR\datasets\Mosaic Image512\mosaictest\GT\000
        savepath_full_name = new_path  # savepath_full_name = os.path.join(new_path, folder_list[i])
        print(folder_full_name)
        # print(savepath_full_name)
        file_list = os.listdir(folder_full_name)  # 文件夹下图片名字
        file_list.sort(key=lambda x: int(x.split('_')[0]))  # 按照_前面的数字进行排序后按顺序读取文件夹下的图片
        # print(file_list)

        for j in range(0, 25):
            g = 2
            file_name = file_list[j]  # balloons_ms_01.png
            originalfile_fullpath = folder_full_name + '\\' + file_name  # D:\BasicSR\datasets\Mosaic Image512\mosaictest\GT\000\00000000.png
            real_name = file_name.split(".")[0]  # 只保留了第一个.前面的数字，即00000000
            file_extension = file_name.split(".")[-1]  # 保留了.后面，即png db
            # print('file_name:', file_name, 'real_name:', real_name, 'file_extension:', file_extension)
            if str(file_extension) == 'png':
                # if str(real_name) == '000000010':
                #     newname = '00000010'"." + file_extension
                #     print("原名字：", file_name, "新名字：", newname)
                #     os.rename(originalfile_fullpath, os.path.join(folder_full_name, newname))
                if j < 9:
                    newname = '0000000' + str(j + 1) + "." + file_extension
                    print("原名字：", file_name, "新名字：", newname)
                else:
                    newname = '000000' + str(j + 1) + "." + file_extension
                    print("原名字：", file_name, "新名字：", newname)
                # os.rename(originalfile_fullpath, os.path.join(folder_full_name, newname))


# 单个文件夹下的图片重命名
def ReNameforfile1(filepath, new_path):
    file_list = os.listdir(filepath)  # 文件夹下图片名字
    file_list.sort(key=lambda x: int(x.split('.')[0]))  # 按照_前面的数字进行排序后按顺序读取文件夹下的图片
    print(file_list)
    for j in range(0, 20):
        file_name = file_list[j]  # 1_s001.jpg
        file_fullpath = filepath + '\\' + file_name  # D:\BasicSR\datasets\1227data\no_drag_1024\\1_s001.jpg
        # print(file_fullpath)
        real_name = file_name.split(".")[0]  # 只保留了第一个.前面的数字，即1_s001
        file_extension = file_name.split(".")[-1]  # 保留了.后面，即jpg
        # print('file_name:', file_name, 'real_name:', real_name, 'file_extension:', file_extension)
        if str(file_extension) == 'jpg':
            # if str(real_name) == '000000010':
            #     newname = '00000010'"." + file_extension
            #     print("原名字：", file_name, "新名字：", newname)
            #     os.rename(originalfile_fullpath, os.path.join(folder_full_name, newname))
            if j < 9:
                newname = '0000000' + str(j + 1) + "." + file_extension
                print("原名字：", file_name, "新名字：", newname)
            else:
                newname = '000000' + str(j + 1) + "." + file_extension
                print("原名字：", file_name, "新名字：", newname)
            os.rename(file_fullpath, os.path.join(new_path, newname))


# 根据文件名字顺序制作文件夹
def mkdir(filepath, savepath):
    file_list = os.listdir(filepath)
    print(file_list)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    for file_index in range(0, 45):
        file_name = file_list[file_index].split(".")[0]
        print(file_name)
        if file_index < 9:
            newpath = savepath + '00' + file_name
        elif file_index < 99:
            newpath = savepath + '0' + file_name
        else:
            newpath = savepath + file_name
        print(newpath)
        os.makedirs(newpath)


# 根据25通道npy文件 按顺序画成图片保存到相应文件夹
def Draw(filepath, savepath):
    filelist = os.listdir(filepath)
    filelist.sort(key=lambda x: int(x.split('.')[0]))
    # print(filelist)
    # for i in range(0, len(file_list)):
    for i in range(0, 5):
        singleFilePath = filepath + filelist[i]
        print(singleFilePath)
        raw_data = np.load(singleFilePath)
        filename = filelist[i].split(".")[0]
        if i+1 < 10:
            img_savepath = savepath + '00' + filename + '\\'
        elif i+1 < 99:
            img_savepath = savepath + '0' + filename + '\\'
        else:
            img_savepath = savepath + filename + '\\'
        print(img_savepath)
        for j in range(0, 25):
            data = raw_data[:, :, j]
            # data *= 255
            img = np.clip(data, 0, 255).astype('uint8')
            # img = Image.fromarray(img, mode='RGB')
            if j < 9:
                img_full_savepath = img_savepath + '0000000' + str(j + 1) + '.png'
            else:
                img_full_savepath = img_savepath + '000000' + str(j + 1) + '.png'
            print(img_full_savepath)
            cv2.imwrite(img_full_savepath, img)  # 保存图片


# 固定座标剪裁
def CutImage(filepath, savepath):
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    list1 = []
    # 读取图片
    for filename in file_list:
        filename1 = os.path.splitext(filename)[1]  # 读取文件后缀名
        filename0 = os.path.splitext(filename)[0]  # 读取文件名
        list1.append(filename0 + filename1)

    for i in range(0, len(list1)):
        filea = str(savepath + list1[i])
        img_1 = mpimg.imread(filea)
        # 设置裁剪的位置
        crop_box = (210, 272, 1622, 1044)
        # 裁剪图片
        img_2 = img_1.crop(crop_box)
        # 保存图片
        img_2.save(savepath + list1[i])
    print('已经截图成功')


# 读取单张图片
def ReadImage(filepath):
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    # print("file_list\n", file_list)
    for i in range(0, 45):
        file_full_path = filepath + file_list[i]
        print(file_full_path)
        return file_full_path
    # for file in file_list:  # 遍历文件夹中的文件
    #     file_fullpath = filepath + '\\' + file
    # return file_fullpath


# 读取单张图片做成马赛克 保存成npy文件 一张图对应一个npy
def ImagetoMasic(filepath, savepath):
    """读取图片"""
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    print("file_list\n", file_list)
    for i in range(0, 10):
        file_full_path = filepath + file_list[i]
        print(file_full_path)
        img = mpimg.imread(file_full_path)
        # img放进马赛克容器中并保存 img是双线性插值后的图片
        img1 = MasicVector(img)  # 256*256*25
        # cv2.imwrite("D:/1/{a}.png".format(a=i+1), img)
        print(savepath + "{a}.npy".format(a=i + 1))
        np.save(savepath + "{a}.npy".format(a=i + 1), img1)
        # print(i+1)


# 单张图片的马赛克容器
def MasicVector(img):
    img = np.expand_dims(img, axis=2)  # (1024, 1024, 1)   uint8
    # print("img.shape:", img.shape)
    # img_mosaic = np.zeros((1024, 1024, 25), dtype=np.float32)  # (1024, 1024, 25)
    img2 = np.zeros((256, 256, 25), dtype=np.float32)  # (1024, 1024, 25)
    # print(img_mosaic.shape)

    temp = np.zeros((5, 5, 25), dtype=np.float32)  # (5, 5, 25)
    # print(temp.shape)
    temp[:, :, 0] = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 1] = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 2] = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 3] = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 4] = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 5] = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 6] = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 7] = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 8] = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 9] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 10] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 11] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 12] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 13] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 14] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 15] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 16] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 17] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 18] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 19] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    temp[:, :, 20] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    temp[:, :, 21] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    temp[:, :, 22] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
    temp[:, :, 23] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]
    temp[:, :, 24] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]

    # 针对1024*1024图片 做成1025然后剪掉两个维度
    # mask = np.tile(temp, (205, 205, 1))  # (1025, 1025, 25)
    # # print('mask.shape', mask.shape)
    # # print("mask.dtype:", mask.dtype)
    # mask = np.delete(mask, -1, axis=1)  # (1025, 1024, 25)
    # # print('mask1.shape', mask.shape)
    # mask = np.delete(mask, -1, axis=0)  # (1024, 1024, 25)
    # # print('mask1.shape', mask.shape)
    # # io.savemat('D:/MST/datasets/hyper_data/mask.mat', {'mask': mask})

    # 针对256*256图片 做成255然后再加一个维度
    mask = np.tile(temp, (51, 51, 1))  # (255, 255, 25)
    # print('mask.shape', mask.shape)
    # print("mask.dtype:", mask.dtype)
    # 25通道都添加行
    new_row = np.array([mask[0, :, :]])   # new_row(1, 255, 25)  1行255列 x轴 axis=0
    # print('new_row.shape', new_row.shape)
    mask1 = np.concatenate((mask, new_row), axis=0)   # (256, 255, 25)
    # 25通道都添加列
    new_col = np.array([mask1[:, 0, :]])  # new_col(1, 256, 25)  1行255列 x轴 axis=0
    new_col = np.swapaxes(new_col, 1, 0)  # new_col(256, 1, 25)  256行1列 y轴 axis=1
    # print('new_col.shape', new_col.shape)
    mask2 = np.concatenate((mask1, new_col), axis=1)  #
    # print('mask2.shape', mask2.shape)  # (256, 256, 25)
    # io.savemat('D:/MST/datasets/hyper_data/mask.mat', {'mask': mask})

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
        img2[:, :, n] = img[:, :, 0] * mask2[:, :, n]
        img2[:, :, n] = scipy.signal.convolve2d(img2[:, :, n], HH, 'same')

    # # img_noise *= 255  #  导致画出来的图片变白的原因
    # img = np.clip(img, 0, 255).astype('float32')
    # cv2.imwrite("D:/1/1.png", img)
    # np.save("D:/1/1", img_mosaic)
    return img2


# 单张图片  反向制作成马赛克相机采集的图片
def pictoMasic_foronepic(filepath, savepath):
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    # print("file_list\n", file_list)
    for i in range(0, 31):
        file_full_path = filepath + '\\' + file_list[i]
        print(file_full_path)
        imgdata = np.zeros((512, 512, 25), dtype=np.float32)  # (1024, 1024, 25)
        for j in range(0, 25):
            imgdata = mpimg.imread(file_full_path)
        img = mpimg.imread(file_full_path)
        # img放进马赛克容器中并保存 img1是双线性插值后的图片
        img1 = MasicVector(img)  # 256*256*25
        # cv2.imwrite("D:/1/{a}.png".format(a=i+1), img)
        print(savepath + "{a}.npy".format(a=i + 1))
        np.save(savepath + "{a}.npy".format(a=i + 1), img1)
        # print(i+1)
    img = np.expand_dims(img, axis=2)  # (1024, 1024, 1)   uint8
    # print("img.shape:", img.shape)
    # img_mosaic = np.zeros((1024, 1024, 25), dtype=np.float32)  # (1024, 1024, 25)
    img2 = np.zeros((256, 256, 25), dtype=np.float32)  # (1024, 1024, 25)
    img3 = np.zeros((256, 256, 25), dtype=np.float32)  # (1024, 1024, 25)
    # print(img_mosaic.shape)

    temp = np.zeros((5, 5, 25), dtype=np.float32)  # (5, 5, 25)
    # print(temp.shape)
    temp[:, :, 0] = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 1] = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 2] = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 3] = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 4] = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 5] = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 6] = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 7] = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 8] = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 9] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 10] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 11] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 12] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 13] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 14] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 15] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 16] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 17] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 18] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 19] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    temp[:, :, 20] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    temp[:, :, 21] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    temp[:, :, 22] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
    temp[:, :, 23] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]
    temp[:, :, 24] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]

    # 针对256*256图片 做成255然后再加一个维度
    mask = np.tile(temp, (51, 51, 1))  # (255, 255, 25)
    # print('mask.shape', mask.shape)
    # print("mask.dtype:", mask.dtype)
    # 25通道都添加行
    new_row = np.array([mask[0, :, :]])   # new_row(1, 255, 25)  1行255列 x轴 axis=0
    # print('new_row.shape', new_row.shape)
    mask1 = np.concatenate((mask, new_row), axis=0)   # (256, 255, 25)
    # 25通道都添加列
    new_col = np.array([mask1[:, 0, :]])  # new_col(1, 256, 25)  1行255列 x轴 axis=0
    new_col = np.swapaxes(new_col, 1, 0)  # new_col(256, 1, 25)  256行1列 y轴 axis=1
    # print('new_col.shape', new_col.shape)
    mask2 = np.concatenate((mask1, new_col), axis=1)  #
    # print('mask2.shape', mask2.shape)  # (256, 256, 25)
    # io.savemat('D:/MST/datasets/hyper_data/mask.mat', {'mask': mask})

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
        img2[:, :, n] = img[:, :, 0] * mask2[:, :, n]
        img3[:, :, n] = scipy.signal.convolve2d(img2[:, :, n], HH, 'same')

    # img_noise *= 255  #  导致画出来的图片变白的原因
    targetimg = np.clip(img2, 0, 255).astype('float32')
    cv2.imwrite(savepath + "{a}.npy".format(a=i + 1), targetimg)
    # np.save("D:/1/1", img_mosaic)
    # return img_chazhi


# 单张图片的马赛克容器
def MasicVector_512(img):
    # img (512, 512, 25)
    masoicdata = np.zeros((512, 512, 25), dtype=np.float32)  # (1024, 1024, 25)
    chazhidata = np.zeros((512, 512, 25), dtype=np.float32)  # (1024, 1024, 25)

    temp = np.zeros((5, 5, 25), dtype=np.float32)  # (5, 5, 25)
    temp[:, :, 0] = [[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 1] = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 2] = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 3] = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 4] = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 5] = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 6] = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 7] = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 8] = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 9] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 10] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 11] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 12] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 13] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 14] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 15] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 16] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 17] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 18] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    temp[:, :, 19] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    temp[:, :, 20] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    temp[:, :, 21] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    temp[:, :, 22] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
    temp[:, :, 23] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]]
    temp[:, :, 24] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]

    mask = np.tile(temp, (102, 102, 1))  # (512, 512, 25)
    # 25通道都添加行
    new_row = np.array([mask[0, :, :]])  # new_row(1, 510, 25)  1行255列 x轴 axis=0
    # print('new_row.shape', new_row.shape)
    mask1 = np.concatenate((mask, new_row), axis=0)  # (511, 510, 25)
    mask1 = np.concatenate((mask1, new_row), axis=0)  # (512, 510, 25)
    print('mask1.shape', mask1.shape)  # (512, 510, 25)
    # 25通道都添加列
    new_col = np.array([mask1[:, 0, :]])  # new_col(1, 512, 25)  1行255列 x轴 axis=0
    new_col = np.swapaxes(new_col, 1, 0)  # new_col(512, 1, 25)  256行1列 y轴 axis=1
    mask2 = np.concatenate((mask1, new_col), axis=1)  # (512, 511, 25)
    mask2 = np.concatenate((mask2, new_col), axis=1)  # (512, 512, 25)
    # print('mask2.shape', mask2.shape)  #
    # print("img.shape:", img.shape)  (512, 512, 25)
    # io.savemat('D:/MST/datasets/hyper_data/mask.mat', {'mask': mask})

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
        masoicdata[:, :, n] = img[:, :, n] * mask2[:, :, n]
        chazhidata[:, :, n] = scipy.signal.convolve2d(masoicdata[:, :, n], HH, 'same')

    # # img_noise *= 255  #  导致画出来的图片变白的原因
    # img = np.clip(img, 0, 255).astype('float32')
    # cv2.imwrite("D:/1/1.png", img)
    # np.save("D:/1/1", img_mosaic)
    return chazhidata


# 25张图像对应单个波段 反向制作成马赛克相机采集的图片
def pictoMasic(filepath, savepath):
    fold_list = os.listdir(filepath)
    fold_list.sort(key=lambda x: int(x.split('.')[0]))
    # print("fold_list", fold_list)
    for i in range(0, 1):
        fold_fullpath = os.path.join(filepath, fold_list[i])
        file_list = os.listdir(fold_fullpath)
        # print(fold_fullpath, '\n', file_list)
        file_fullpath = os.path.join(fold_fullpath, file_list[i])
        imgdata = np.zeros((512, 512, 25), dtype=np.float32)  # (1024, 1024, 25)
        print(fold_fullpath, file_fullpath)
        for j in range(0, 25):
            imgdata[:, :, j] = mpimg.imread(file_fullpath)  # (512, 512, 25)
        masoicdata = MasicVector_512(imgdata)  # (512, 512, 25)
        # print(masoicdata.shape)


if __name__ == '__main__':
    """单个文件夹下图片重命名 保存到指定文件夹"""
    old_file_path = r'D:\bella\datasets\mosaicdata\\'
    new_file_path = r'D:\bella\datasets\mosaicdata\\'
    # ReNameforfile(old_file_path, new_file_path)

    """单图制作成马赛克 存为npy格式"""
    original_path = 'D:\\bella\\datasets\\1108data\\'
    npy_path = r'D:\bella\datasets\1108data\原始数据\datanpy\\'
    # ImagetoMasic(original_path, npy_path)
    # ReadImage(original_path)

    """文件夹制作"""
    original_dirpath = 'D:\\bella\\datasets\\1108data\\原始数据\\无农药-256\\nodrag_256_npy'
    new_dirpath = r'D:\bella\datasets\1108data\alldata_256\\'
    # mkdir(npy_path, new_dirpath)

    """把高清图片制作成二维马赛克图片"""
    fulldata_pic = r'D:\bella\datasets\mosaicdata-gt'
    masic_savepath = r'D:\bella\datasets\mosaicdata-lr'
    pictoMasic(fulldata_pic, masic_savepath)
    """25通道npy画成图"""
    Draw(npy_path, new_dirpath)

"""查看数据"""
# img_mosaic = np.load('D:/1/1.npy')
#
# for j in range(25):
#     img = img_mosaic[:, :, j]
#     # img *= 255
#     img = np.clip(img, 0, 255).astype('uint8')
#     # io.savemat('D:/6/result_2-1198.mat', {'testnew2': img})
#     cv2.imwrite("D:/1/1/{a}.png".format(a=j), img)
#     print(j)