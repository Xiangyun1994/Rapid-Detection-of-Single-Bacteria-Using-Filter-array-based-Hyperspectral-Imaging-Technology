import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from scipy.io import savemat
from skimage import io, transform
import h5py
import shutil
from scipy import signal
import matplotlib.pyplot as plot
torch.manual_seed(1)  # reproducible


"""重命名"""


def ReNameforfolder(filepath, newpath):
    file_list = os.listdir(filepath)
    # print(file_list)
    file_list.sort(key=lambda x:int(x.split(".")[0]))
    for file_index in range(0, 620):
        full_name = os.path.join(filepath, file_list[file_index])  # D:\BasicSR\datasets\Mosaic Image512\mosaictest\GT\beads_ms
        file_name = file_list[file_index]  # beads_ms
        # print(full_name)
        # print(file_name)
        # real_name = file_name.split(".")[0]  # 只保留了第一个.前面的数字，即ulm_0328-1118a4
        # print(real_name)
        # file_extension = file_name.split(".")[-1]  # 保留了.后面，即npy
        # if file_index <= 9:
        #     # newname = '00' + str(file_index)
        #     newname = str(file_index) + file_extension
        #     # print(newname)
        # elif file_index <= 99:
        #     # newname = '0' + str(file_index)
        #     newname = str(file_index) + file_extension
        # else:
        #     # newname = str(file_index)
        #     newname = str(file_index)
        #     # print(newname)
        newname = str(file_index) + '.mat'
        savepath = os.path.join(newpath, newname)
        print("原名字：", file_name, "新名字：", newname, "path: ", savepath)

        os.rename(full_name, os.path.join(filepath, newname))


def ReNameforfolder1(filepath, newpath):
    file_list = os.listdir(filepath)
    # file_list.sort(key=lambda x:int(x))
    for file_index in range(0, 155):
        fullfilepath = os.path.join(filepath, file_list[file_index])  # D:\BasicSR\datasets\Mosaic Image512\mosaictest\GT\beads_ms
        newname = str(file_index+1) + '.mat'
        savepath = os.path.join(newpath, newname)
        # print("原名字：", file_list[file_index], "新名字：", newname, "path: ", savepath)
        print("原路径：", fullfilepath, "新路径: ", savepath)

        os.rename(fullfilepath, savepath)


def ReNameforfile_13(folderpath, new_path):
    folder_list = os.listdir(folderpath)
    savefolder_list = os.listdir(new_path)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    savefolder_list.sort(key=lambda x: int(x.split('.')[0]))
    print(folder_list, savefolder_list)
    for i in range(0, 31):
        folder_fullpath = os.path.join(folderpath, folder_list[i])  # D:\BasicSR\datasets\Mosaic Image512\mosaictest\GT\000
        savepath_fullpath = os.path.join(new_path, folder_list[i])  # savepath_full_name = os.path.join(new_path, folder_list[i])
        # print(folder_fullpath, savepath_fullpath)
        file_list = os.listdir(folder_fullpath)  # 文件夹下图片名字
        # file_list.sort(key=lambda x: int(x.split('.')[0]))  # 按照_前面的数字进行排序后按顺序读取文件夹下的图片
        # print(file_list)
        # file_list.sort(key=lambda x: int((x.split('_')[-1]).split('.')[0]))  # 按照_前面的数字进行排序后按顺序读取文件夹下的图片
        for j in range(0, 25):
            file_name = file_list[j]
            real_name = file_name.split(".")[0]  # 只保留了第一个.前面的数字，即00000000
            # file_extension = file_name.split(".")[-1]  # 保留了.后面，即png db
            file_extension = 'png'
            # print('file_name:', file_name, 'real_name:', real_name, 'file_extension:', file_extension)
            img_originalpath = os.path.join(folder_fullpath, file_name)
            if j <= 9:
                newname = '0000000' + str(j) + "." + file_extension
                # newname = '000000' + str(j + 101) + "." + file_extension
                # print("原名字：", file_name, "新名字：", newname)
                # copyfile(originalfile_fullpath, new_path)
            else:
                newname = '000000' + str(j) + "." + file_extension
                # print("原名字：", file_name, "新名字：", newname)
            savepath = os.path.join(savepath_fullpath, newname)
            # print(img_originalpath, savepath)
            print(img_originalpath, savepath)
            os.rename(img_originalpath, savepath)
            # shutil.copyfile(img_originalpath, savepath)  # 复制文件 保留源文件


def ReNameforfile_1(folderpath, new_path):
    folder_list = os.listdir(folderpath)
    savefolder_list = os.listdir(new_path)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    savefolder_list.sort(key=lambda x: int(x.split('.')[0]))
    # print(folder_list, savefolder_list)
    for i in range(601, 602):
        folder_fullpath = os.path.join(folderpath, folder_list[i])  # D:\BasicSR\datasets\Mosaic Image512\mosaictest\GT\000
        savepath_fullpath = os.path.join(new_path, folder_list[i])  # savepath_full_name = os.path.join(new_path, folder_list[i])
        # print(folder_fullpath, savepath_fullpath)
        file_list = os.listdir(folder_fullpath)  # 文件夹下图片名字
        file_list.sort(key=lambda x: int(x.split('.')[0]))  # 按照_前面的数字进行排序后按顺序读取文件夹下的图片
        # file_list.sort(key=lambda x: int((x.split('_')[-1]).split('.')[0]))  # 按照_前面的数字进行排序后按顺序读取文件夹下的图片
        # print(file_list)
        for j in range(0, 25):
            file_name = file_list[j]
            real_name = file_name.split(".")[0]  # 只保留了第一个.前面的数字，即00000000
            file_extension = file_name.split(".")[-1]  # 保留了.后面，即png db
            # print('file_name:', file_name, 'real_name:', real_name, 'file_extension:', file_extension)
            img_originalpath = os.path.join(folder_fullpath, file_name)
            if j == 0:    newname = "25." + file_extension
            elif j == 1:    newname = "24." + file_extension
            elif j == 2:    newname = "23." + file_extension
            elif j == 3:    newname = "22." + file_extension
            elif j == 4:    newname = "21." + file_extension
            elif j == 5:    newname = "20." + file_extension
            elif j == 6:    newname = "19." + file_extension
            elif j == 7:    newname = "18." + file_extension
            elif j == 8:    newname = "17." + file_extension
            elif j == 9:    newname = "16." + file_extension
            elif j == 10:   newname = "15." + file_extension
            elif j == 11:   newname = "14." + file_extension
            else:
                newname = str(j-11) + "." + file_extension
                # print("原名字：", file_name, "新名字：", newname)
                # copyfile(originalfile_fullpath, new_path)
            #     # print("原名字：", file_name, "新名字：", newname)
            savepath = os.path.join(savepath_fullpath, newname)
            # print(img_originalpath, savepath)
            print(img_originalpath, savepath)
            os.rename(img_originalpath, savepath)
            # shutil.copyfile(img_originalpath, savepath)  # 复制文件 保留源文件


"""复制/移动文件  文件夹下多个子文件夹 子文件夹下多张图片 指定图片名字 移动到指定文件夹下   自动遍历图片后缀名"""


def copyforfile(filepath, new_path):
    folder_list = os.listdir(filepath)
    namestr = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008',
               's009', 's010', 's011', 's012', 's013', 's014', 's015', 's016']
    savefile_list = os.listdir(new_path)
    # print(folder_list)
    for i in range(0, 630):  # 文件夹
        folder_full_name = os.path.join(filepath, folder_list[i])  # D:\BasicSR\datasets\1227data\no_drag_mosaic_128\001
        file_list = os.listdir(folder_full_name)  # ['balloons_ms_01.png', ..., 'Thumbs.db']
        savepath = os.path.join(new_path, folder_list[i])
        # print(folder_full_name)
        # print(savepath)
        # for k in range(0, 1):  # 图片后缀名控制 保存路径控制
        #     # save_path = os.path.join(new_path, savefile_list[k+16*i])
        #     if k+16*i <= 9:
        #         save_path = new_path + '\\00' + str(k+16*i)
        #     elif k+16*i <= 99:
        #         save_path = new_path + '\\0' + str(k+16*i)
        #     else:
        #         save_path = new_path + '\\' + str(k+16*i)
        #     print('\n', save_path)
        #     # name = namestr[k-1]
        #     # print('\n', name)
        for j in range(0, 25):  # 找到符合图片后缀名的所有图片
            file_name = file_list[j]  # 00000001_s004_s002.png
            originalfile_fullpath = folder_full_name + '\\' + file_name  # D:\BasicSR\datasets\1227data\no_drag_mosaic_128\001\00000001_s004_s002.png
            real_name = file_name.split(".")[0]  # 只保留了第一个.前面的数字，即00000001_s004_s002
        #     name1 = real_name.split("_")[-1]  # 保留了第一个_的倒数第一个字符串，即s002
        #     name2 = real_name.split("_")[-2]  # 保留了第一个_的倒数第二个字符串，即s004
        #     name3 = real_name.split("_")[0]  # 保留了第一个_的正数第一个字符串，即00000001
            file_extension = file_name.split(".")[-1]  # 保留了.后面，即png db
            # print('file_name:', file_name, 'real_name:', real_name, 'file_extension:', file_extension)
        #     print('real_name:', real_name, 'name1:', name1, 'name2', name2, 'name3:', name3)
            if str(file_extension) == 'png' and real_name == '00000000':  # 自动根据后缀名
        #         # newname = '0000000' + str(j) + "." + file_extension
        #         # print("新名字：", newname)
                save_fullpath = savepath + '-0.png'
                print('old_path', originalfile_fullpath)
                print('save_path', save_fullpath)
                shutil.copyfile(originalfile_fullpath, save_fullpath)   # 复制文件 保留源文件
        #         # shutil.move(originalfile_fullpath, save_fullpath)  # 移动文件 不保留源文件
        # print(i + 1, "Done", '\n')


def find_save_targetfile(file_dir, save_dir):
    '''
    该函数实现从文件夹中根据文件后缀名提取文件，并存储在新的文件夹中
    file_dir指读的文件目录；save_dir为保存文件的目录
    suffix用于存放打算提取文件的后缀名；
    file_dir D:\BasicSR\datasets\1229data\128\

    '''
    folder_list = os.listdir(file_dir)  # ['M128_001', 'M128_002',..., 'M128_044']
    # print(folder_list)
    savefolder_list = os.listdir(save_dir)
    # print(savefolder_list)
    suffix = ['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008',
              's009', 's010', 's011', 's012', 's013', 's014', 's015', 's016',
              's017', 's018', 's019', 's020', 's021', 's022', 's023', 's024',
              's025', 's026', 's027', 's028', 's029', 's030', 's031', 's032',
              's033', 's034', 's035', 's036', 's037', 's038', 's039', 's040',
              's041', 's042', 's043', 's044', 's045', 's046', 's047', 's048',
              's049', 's050', 's051', 's052', 's053', 's054', 's055', 's056',
              's057', 's058', 's059', 's060', 's061', 's062', 's063', 's064']
    """读第一个子文件夹"""
    for j in range(0, 45):
        singledir_fullpath = file_dir + folder_list[j]  # D:\BasicSR\datasets\1229data\马赛克剪裁\128\M128_001
        print(singledir_fullpath)
        file_list = os.listdir(singledir_fullpath)  # ['00000001_s001.png', ..., '00000025_s064.png']
        # print(file_list)
        """读后缀名和保存路径"""
        for i in range(0, 64):
            filelist = []  # 存储要copy的文件全名
            appendix = suffix[i]
            # print(appendix)
            save_fullpath = save_dir + '\\' + savefolder_list[i]  # D:\BasicSR\datasets\1229data\128\000
            print(save_fullpath, appendix)
            """读第一个子文件夹的所有图片"""
            for q in range(0, 1600):
                filename = file_list[q].split('.')[0]  # 00000001_s001
                file_type = file_list[q].split('.')[-1]  # png  对文件名根据.进行分隔，实现文件名，后缀名的分离
                file_suffix = filename.split('_')[-1]  # s001
                if file_suffix in appendix:  # 下面根据后缀名是否在列表中，提取文件
                    # print(file_list[q])  # 00000001_s001.png
                    # 文件全名 D:\BasicSR\datasets\1229data\马赛克剪裁\128\M128_001\00000001_s001.png
                    file_fullname = os.path.join(singledir_fullpath, file_list[q])
                    # print(file_fullname)
                    filelist.append(file_fullname)  # 将符合要求的文件存放在列表中
            """保存符合后缀名的子文件夹的所有图片到新文件夹中"""
            for file in filelist:
                print(file)
                shutil.copy(file, save_fullpath)#将列表中的文件复制到新的文件夹
            continue
        continue


"""根据文件名创建文件夹mkdir  直接创建文件夹mkdir1"""


def mkdir(filepath, savepath):
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('.')[0]))  # ['1.mat',...,'620.mat']
    # print(file_list)
    for file_index in range(0, 31):
        file_name = file_list[file_index].split(".")[0]
        if file_index <= 9:
            newpath = savepath + '00' + file_name
        elif 10 <= file_index <= 99:
            newpath = savepath + '0' + file_name
        else:
            newpath = savepath + file_name  # D:\BasicSR\datasets\Mosaic Image512\train\GT620
        print(newpath)
        # os.makedirs(newpath)


def mkdir1(savepath):
    for file_index in range(30, 31):
        if file_index <= 9:
            foldname = '00' + str(file_index)
        elif 10 <= file_index <= 99:
            foldname = '0' + str(file_index)
        else:
            foldname = str(file_index)
        newpath = os.path.join(savepath, foldname)
        print(newpath)
        os.makedirs(newpath)


"""删除多余的图片"""
def removeforfile(folderpath):
    folder_list = os.listdir(folderpath)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))  # ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014']
    # print(folder_list)
    # for fold in folder_list:
    # for i in range(len(folder_list)):
    for i in range(0, 620):
        folder_fullpath = os.path.join(folderpath, folder_list[i])  # D:\BasicSR\results\DUF_Mosaic_25输入25输出_BN对比\DUF_Mosaic_16_x4_25_512-1\visualization\test128\000
        # print(folder_fullpath)
        file_list = os.listdir(folder_fullpath)  # 00000001_DUF_Mosaic_16_x4_25_512-1.png
        for j in range(0, len(file_list)):
            file_fullpath = os.path.join(folder_fullpath, file_list[j])  # D:\BasicSR\results\DUF_Mosaic_25输入25输出_BN对比\DUF_Mosaic_16_x4_25_512-1\visualization\test128\002\00000001_DUF_Mosaic_16_x4_25_512-1.png
            # print(file_fullpath)
            picturename = file_list[j].split(".")[0]  # 00000002_DUF_Mosaic_16_x4_25_512-1
            suffix = file_list[j].split(".")[-1]
            picturename_index = picturename.split("_")[0]  # 00000002
            # print(picturename, suffix, picturename_index)
            if str(picturename_index) == '0000009':
                path1 = file_fullpath
                savepath = folder_fullpath + '\\00000009.png'
                os.rename(path1, savepath)
                # os.remove(path1)
                print(path1, savepath)
            else:
                path2 = file_fullpath
                continue
                # os.remove(path2)
                # print(path2)


"""图片剪裁"""
one = ['1', '2', '3', '4', '5']  # 1，不要相同就行
two = ['6', '7', '8', '9', '10']  # 2，不要相同就行
three = ['11', '12', '13', '14', '15']  # 3，不要相同就行
four = ['16', '17', '18', '19', '20']  # 4，不要相同就行
five = ['21', '22', '23', '24', '25']  # 5，不要相同就行


def cropimg_512(inputfolder_dir, save_dir):
    folder_list = os.listdir(inputfolder_dir)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))  # ['1.mat',...,'620.mat']
    # print(folder_list)
    for t in range(6, 7):
    # for t in range(len(folder_list)):
        foldpath = os.path.join(inputfolder_dir, folder_list[t])
        save_path = os.path.join(save_dir, folder_list[t])
        imgname = os.listdir(foldpath)[0]
        imgpath = os.path.join(foldpath, imgname)
        print(imgpath, save_path)
        img = Image.open(imgpath)  ## 打开chess.png，并赋值给img
        for i, j, z, q, p, k, in zip(range(2, 2600, 514), one, two, three, four, five):
            # x1 y1 x2 y2
            # print((i, 2, i + 512, 514), save_path + f'{j}.png')
            # print((i, 516, i + 512, 1028), save_path + f'{z}.png')
            # print((i, 1030, i + 512, 1542), save_path + f'{q}.png')
            # print((i, 1544, i + 512, 2056), save_path + f'{p}.png')
            # print((i, 2058, i + 512, 2570), save_path + f'{k}.png')
            region1 = img.crop((i, 2, i + 512, 514))  # 裁剪第1排
            region1.save(save_path + f'\\{j}.png')  # 保存第1排
            # print(save_path + f'\\{j}.png')
            # region1.show()

            region2 = img.crop((i, 516, i + 512, 1028))  # 裁剪第2排
            region2.save(save_path + f'\\{z}.png')  # 保存第2排
            # region2.show()

            region3 = img.crop((i, 1030, i + 512, 1542))  # 裁剪第3排
            region3.save(save_path + f'\\{q}.png')  # 保存第3排
            # region3.show()

            region4 = img.crop((i, 1544, i + 512, 2056))  # 裁剪第4排
            region4.save(save_path + f'\\{p}.png')  # 保存第4排
            # region4.show()

            region5 = img.crop((i, 2058, i + 512, 2570))  # 裁剪第5排
            region5.save(save_path + f'\\{k}.png')  # 保存第5排
            # region5.show()


"""Mat转图片"""


def MattoImage(filepath, new_path):
    filelist = os.listdir(filepath)
    filelist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(filelist)
    for i in range(0, 31):
        first_name, second_name = os.path.splitext(filelist[i])  # 拆分.mat文件的前后缀名字，0 .mat
        # print(first_name, second_name, each_mat)  # 0 .mat 0.mat
        matdatapath = os.path.join(filepath, filelist[i])  # 校验步骤，输出应该是路径
        # print(matdatapath)
        array_struct = loadmat(matdatapath)  # 校验步骤，输出的应该是一个结构体，然后查看你的控制台，看数据被存在了哪个字段里
        # print(array_struct)
        array_data = array_struct['testdata']  # 取出需要的数字矩阵部分  (256, 256, 25)
        # array_data.transpose(1, 2, 0)
        # print(array_data.shape)
        if i <= 9:
            foldname = '00' + str(i)
        elif i <= 100:
            foldname = '0' + str(i)
        else:
            foldname = str(i)
        img_savepath = os.path.join(new_path, foldname)  # D:\bella\datasets\test256_0\030
        print(matdatapath, '\t', img_savepath)  # D:\bella\datasets\测试集\test256_mat_0\0.mat
        for j in range(0, 25):
            data = array_data[:, :, j]
            # img = np.clip(data, 0, 255).astype('uint8')
            # print(np.max(data))
            if j <= 9:
                imgname = '0000000' + str(j) + '.png'
            else:
                imgname = '000000' + str(j) + '.png'
            # if j <= 9:
            #     imgname = str(j+1) + '.png'
            # else:
            #     imgname = str(j+1) + '.png'
            img_full_savepath = os.path.join(img_savepath, imgname)  # D:\bella\datasets\test256_0\030\00000000.png
            print(img_full_savepath)
            # data *= 255
            # img = np.clip(data, 0, 255).astype('float32')  # (256, 256)
            # imagegray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # image = imagegray
            io.imsave(img_full_savepath, data)
            # # cv2画图
            # # img1 = Image.fromarray(img)
            # # print(img.shape)
            # print(np.max(image), image.shape)
            # cv2.imwrite(img_full_savepath, image)  # 保存图片 还是3通道


def MattoImage1(filepath, new_path):
    filelist = os.listdir(filepath)
    filelist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(filelist)
    for i in range(0, 155):
        first_name, second_name = os.path.splitext(filelist[i])  # 拆分.mat文件的前后缀名字，0 .mat
        # print(first_name, second_name)  # 0 .mat 0.mat
        matdatapath = os.path.join(filepath, filelist[i])  # 校验步骤，输出应该是路径
        # array_struct = loadmat(matdatapath)  # 校验步骤，输出的应该是一个结构体，然后查看你的控制台，看数据被存在了哪个字段里
        matdata = h5py.File(matdatapath, mode='r')['rad']
        matdata_array = np.array(matdata)
        if i <= 9:
            foldname = '00' + str(i)
        elif 10 <= i < 100:
            foldname = '0' + str(i)
        else:
            foldname = str(i)
        img_savepath = os.path.join(new_path, foldname)  # D:\bella\datasets\test256_0\030
        print(matdatapath, img_savepath)
        matdata1 = matdata_array.transpose(1, 2, 0)
        # print(matdata.shape, matdata1.shape)
        for j in range(0, 31):
            data = matdata1[:, :, j]
            # print(np.max(data))
            if j <= 9:
                imgname = '0000000' + str(j) + '.png'
            else:
                imgname = '000000' + str(j) + '.png'
            img_full_savepath = os.path.join(img_savepath, imgname)  # D:\bella\datasets\test256_0\030\00000000.png
            print(img_full_savepath)
            # img = np.clip(data, 0, 255).astype('uint8')  # (256, 256)
            io.imsave(img_full_savepath, data)
        #     # # cv2画图
        #     # # img1 = Image.fromarray(img)
        #     # # print(img.shape)
        #     print(np.max(image), image.shape)
        #     cv2.imwrite(img_full_savepath, image)  # 保存图片 还是3通道
"""NPY转图片"""


def NPYtoImage(filepath, savepath):
    file_list = os.listdir(filepath)
    file_list.sort(key=lambda x: int(x.split('.')[0]))
    # print(file_list)
    # for i in range(0, len(file_list)):
    for i in range(0, 100):
        tempfile = filepath + file_list[i]
        raw_data = np.load(tempfile)
        filename = file_list[i].split(".")[0]  # 1.npy——001
        filename = str(int(filename)-1)
        # filename = str(int(file_list[i].split(".")[0]) - 1)  # 1.npy——000 减1
        # print(tempfile)
        # print(filename)
        if i <= 9:
            fold_save = '00' + filename
        elif i <= 99:
            fold_save = '0' + filename
        else:
            fold_save = filename
        savepath1 = os.path.join(savepath, fold_save)
        print(tempfile, "\t", savepath1)
        for j in range(0, 25):
            data = raw_data[:, :, j]
            data *= 255
            if j <= 9:
                imgname = '0000000' + str(j) + '.png'
            else:
                imgname = '000000' + str(j) + '.png'
            img_full_savepath = os.path.join(savepath1, imgname)
            print(img_full_savepath)
            img = np.clip(data, 0, 255).astype('uint8')
            # print(img.shape)
            io.imsave(img_full_savepath, img)
            # cv2.imwrite(img_full_savepath, img)  # 保存图片  cv2.imwrite使用时路径不能有中文否则保存失败


"""旋转图片"""


def rotateimg(filepath, new_path):
    foldlist = os.listdir(filepath)
    foldlist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(foldlist)
    for each_fold in range(0, 30):
        folderpath = os.path.join(filepath, foldlist[each_fold])  # D:\BasicSR\datasets\1227data\no_drag_mosaic_128\001
        file_list = os.listdir(folderpath)  # ['balloons_ms_01.png', ..., 'Thumbs.db']
        file_list.sort(key=lambda x: int(x.split('.')[0]))
        savepath = os.path.join(new_path, foldlist[each_fold])
        # print(folderpath, savepath)
        # print(file_list)
        for file in file_list:
            file_originpath = os.path.join(folderpath, file)
            # print(file_originpath, file_savepath)
            # cv 读取灰度图像
            # image_gray = cv2.imread(file_originpath, cv2.IMREAD_GRAYSCALE)
            # img_rotate = cv2.rotate(image_gray, cv2.ROTATE_90_CLOCKWISE)
            # img_filp = cv2.flip(img_rotate, 1)

            # PIL
            image_gray = Image.open(file_originpath).convert('L')
            img_filp = image_gray.transpose(Image.FLIP_LEFT_RIGHT)
            img_rotate = img_filp.transpose(Image.ROTATE_90)

            # print(img_filp.size)
            # img_index = int(file.split('.')[0])
            # imgname = str(img_index) + '.png'
            # img_savepath = os.path.join(savepath, imgname)  # D:\bella\datasets\test256_0\030\00000000.png
            img_savepath = os.path.join(savepath, file)  # D:\bella\datasets\test256_0\030\0.png
            # cv2.imwrite(img_savepath, img_filp)
            print(file, file_originpath, img_savepath)
            # img_rotate.show()
            # img_rotate.save(img_savepath, img)


"""图片-mat"""


def imgtomat(filepath, savepath):
    foldlist = os.listdir(filepath)
    foldlist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(foldlist)
    for i in range(0, 31):
        folderpath = os.path.join(filepath, foldlist[i])  # D:\BasicSR\datasets\1227data\no_drag_mosaic_128\001
        file_list = os.listdir(folderpath)
        # print(file_list)
        # max = 200
        data = np.ones((2048, 2048, 25), dtype=np.float64)
        # data = np.ones((1024, 1024, 25), dtype=np.float64)
        # data = np.ones((512, 512, 25), dtype=np.float64)
        for j in range(0, 25):
            file_originpath = os.path.join(folderpath, file_list[j])
            # PIL
            image = Image.open(file_originpath)
            image_arr = np.array(image)
            # data[:, :, j] = image_arr[:, :, 0]
            data[:, :, j] = image_arr
            # print(folderpath, file_originpath)
        # if np.max(data) > max:
        #     max = np.max(data)
        matname = foldlist[i] + '.mat'
        mat_savepath = os.path.join(savepath, matname)
        print(folderpath, mat_savepath)
        savemat(mat_savepath, {"gt_data": data})


"""图片-npy"""


def imgtonpy(filepath, savepath):
    foldlist = os.listdir(filepath)
    foldlist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(foldlist)
    for i in range(0, 1):
        folderpath = os.path.join(filepath, foldlist[i])  # D:\BasicSR\datasets\1227data\no_drag_mosaic_128\001
        file_list = os.listdir(folderpath)
        # print(file_list)
        max = 200
        # data = np.ones((512, 512, 25), dtype=np.float64)
        data = np.ones((1, 200, 200, 25), dtype=np.float64)
        for j in range(0, 25):
            file_originpath = os.path.join(folderpath, file_list[j])
            # PIL
            image = Image.open(file_originpath)
            image_arr = np.array(image)
            data[0, :, :, j] = image_arr
            # print(folderpath, file_originpath)
        if np.max(data) > max:
            max = np.max(data)
        print(folderpath, max)
        # savemat(savepath+'\\{}.mat'.format(i), {"gtdata": data})
        np.save(savepath+'\\{}.npy'.format(i), data)


def downsample_image(inputpath, outpath, factor):
    foldlist = os.listdir(inputpath)
    foldlist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(foldlist)
    for each_fold in range(0, 31):
        folderpath = os.path.join(inputpath, foldlist[each_fold])  # D:\BasicSR\datasets\1227data\no_drag_mosaic_128\001
        file_list = os.listdir(folderpath)  # ['balloons_ms_01.png', ..., 'Thumbs.db']
        # file_list.sort(key=lambda x: int(x.split('.')[0]))
        savepath = os.path.join(outpath, foldlist[each_fold])
        # print(folderpath, savepath)
        # print(file_list)
        for file in range(0, 25):
            file_originpath = os.path.join(folderpath, file_list[file])
            resizedimg_savepath = os.path.join(savepath, file_list[file])
            image1 = cv2.imread(file_originpath)
            imagegray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image = imagegray

            # # 固定大小
            # newsize = (512, 512)
            # resized_image1 = cv2.resize(imagegray, newsize)
            # cv2.imwrite(resizedimg_savepath, resized_image1)

            # # 倍数大小
            width, height = image.shape[1], image.shape[0]
            newsize = (width // factor, height // factor)  # newsize = (512, 512)
            resized_image = (transform.resize(image, newsize, preserve_range=True))
            resized_image1 = np.array(resized_image, dtype=np.uint8)
            # print(image.dtype, "\t", image.shape, "\t", resized_image1.shape)
            print(file_originpath, resizedimg_savepath)
            io.imsave(resizedimg_savepath, resized_image1)
            # print(image.dtype, "\t", image.shape)
            # print(resized_image.dtype, "\t", resized_image.shape)
            # print(resized_image1.dtype, "\t", resized_image1.shape)

"""
高光谱马赛克512*512*25 Mat
画成二维图像
"""


def drawmosaicdatatoImage(filepath, new_path):
    filelist = os.listdir(filepath)
    filelist.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字进行排序后按顺序读取文件夹下的图片
    # print(filelist)
    for each_mat in filelist:
        first_name, second_name = os.path.splitext(each_mat)  # 拆分.mat文件的前后缀名字，注意是**路径**
        imgname = str(int(first_name)+1) + '.png'
        img_full_savepath = os.path.join(new_path, imgname)
        # print(first_name, second_name, each_mat)
        matdatapath = os.path.join(filepath, each_mat)  # 校验步骤，输出应该是路径
        print(matdatapath)  # D:\bella\datasets\测试集_马赛克插值_512_mat_460-700\30.mat
        array_struct = loadmat(matdatapath)  # 校验步骤，输出的应该是一个结构体，然后查看你的控制台，看数据被存在了哪个字段里
        # print(array_struct)
        array_data = array_struct['mosaicdata']  # 取出需要的数字矩阵部分
        data = np.ones((512, 512), dtype=np.float32)
        for n in range(25):
            for i in range(array_data.shape[0]):
                for j in range(array_data.shape[1]):
                    data[i, j] += array_data[i, j, n]
        img = data * 255
        img = np.clip(data, 0, 255).astype('float32')
        print(img_full_savepath)
        io.imsave(img_full_savepath, img) # 保存图片
        # print(data.shape)


if __name__ == '__main__':
    mkdirpath = r'D:\highspectrums-datasets\3.CAVE\LR_512_img'
    ReNamepath_before = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\1-superresolution_512_1x-256x8\test256'
    ReNamepath_after = r'D:\BasicSR\datasets\Mosaic Image512\demosaic\mosaic_train\LR_256-1'
    ReNamepath = r'D:\highspectrums-datasets\2\LR_512_img'
    ReNamesavepath = r'D:\BasicSR\datasets\test200\GT_200'
    removepath = r'D:\Bella_2\results\test256_400-640_all\test256_0.025-v1-model2\test256'
    copyfilepath_before = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_train\GT_512'
    copyfilepath_after = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_train\1'

    crop_inputpath = r'D:\highspectrums-datasets\2\GT_512_img'
    crop_outputpath = r'D:\BasicSR\datasets\Mosaic Image512\demosaic\mosaic_train\GT_2048-1'

    matpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\GT_mat_2048'
    Image_path = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_test\GT_2048'

    npypath = r'D:\高光谱数据库\2\LR_512\\'
    npytoimgpath = r'D:\高光谱数据库\2\LR_512_IMG'

    resized_before = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_test\GT_1024'
    resized_after = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_test\GT_2048'
    # ReNameforfolder(ReNamepath, ReNamepath)
    # ReNameforfolder1(ReNamepath_before, ReNamepath_after)
    # ReNameforfile_13(ReNamepath_before, ReNamepath_before)
    # removeforfile(ReNamepath)
    # ReNameforfile_1(crop_outputpath, crop_outputpath)
    # mkdir1(resized_after)
    # copyforfile(copyfilepath_before, copyfilepath_after)
    # removeforfile(crop_inputpath)
    # cropimg_512(crop_inputpath, crop_outputpath)
    # MattoImage(matpath, Image_path)
    # rotateimg(Image_path, Image_path)
    # NPYtoImage(npypath, npytoimgpath)
    imgtomat(Image_path, matpath)
    # imgtonpy(Image_path, npypath)
    # downsample_image(resized_before, resized_after, 0.5)  #131开始
    # buildtestdata_512()
    # buildtestdata_128()
    # drawmosaicdatatoImage(matpath, Image_path)


'''NPY数据格式'''
# transform = transforms.Compose([transforms.ToTensor(), ])  # 将图片转换为Tensor,归一化至[0,1]
#
# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = np.load(data)  # 加载npy数据
#         self.transforms = transform  # 转为tensor形式
#
#     def __getitem__(self, index):
#         hdct = self.data[index, :, :, :]  # 读取每一个npy的数据
#         hdct = np.squeeze(hdct)  # 删掉一维的数据，就是把通道数这个维度删除
#         ldct = 2.5 * skimage.util.random_noise(hdct * (0.4 / 255), mode='poisson', seed=None) * 255  # 加poisson噪声
#         hdct = Image.fromarray(np.uint8(hdct))  # 转成image的形式
#         ldct = Image.fromarray(np.uint8(ldct))  # 转成image的形式
#         hdct = self.transforms(hdct)  # 转为tensor形式
#         ldct = self.transforms(ldct)  # 转为tensor形式
#         return ldct, hdct  # 返回数据还有标签
#
#     def __len__(self):
#         return self.data.shape[0]  # 返回数据的总个数
#
#
# def main():
#     dataset = MyDataset('.\data_npy\img_covid_poisson_glay_clean_BATCH_64_PATS_100.npy')
#     data = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
