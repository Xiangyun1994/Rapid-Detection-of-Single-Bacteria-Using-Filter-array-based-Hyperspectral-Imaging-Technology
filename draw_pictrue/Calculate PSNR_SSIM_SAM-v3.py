from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
from numpy.linalg import norm
import os
import matplotlib.image as mpimg
import xlwt
from PIL import Image
import cv2
# from openpyxl import Workbook
import math
import tensorflow as tf

"""
计算各指标并保存-v3
"""

def mypsnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0)**2)
    if mse < 1e-10:
        return 100
    res = 20*math.log10(255/math.sqrt(mse))
    return res


def myssim(img1, img2):
    hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img1[255, 255, 255] = 0  # ignore all white pixels
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img2[255, 255, 255] = 0  # ignore all white pixels
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # Find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    return metric_val


# 保存数据到excel
def savetoexcel(testpath, gtpath):
    savepath = os.path.dirname(testpath)
    excel_savepath = savepath + '\\PSNR_SSIM.xls'
    print(excel_savepath)

    result = readimgtoarr(testpath, gtpath)
    result_psnr = result[0]
    result_ssim = result[1]

    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet1.write(0, 0, "测试集")  # 第1行第1列
    sheet1.write(0, 1, "PSNR")  # 第1行第2列
    sheet1.write(0, 2, "SSIM")  # 第1行第3列

    # 循环填入数据
    for i in range(len(result_psnr)):
        if i + 1 == len(result_psnr):
            sheet1.write(i + 1, 0, '平均值')  # 第1列序号
            sheet1.write(i + 1, 1, result_psnr[i])  # 第2列数量
            sheet1.write(i + 1, 2, result_ssim[i])  # 第3列误差
        else:
            sheet1.write(i + 1, 0, i)  # 第1列序号
            sheet1.write(i + 1, 1, result_psnr[i])  # 第2列数量
            sheet1.write(i + 1, 2, result_ssim[i])  # 第3列误差
        # sheet1.write(i + 1, 0, i)  # 第1列序号
        # sheet1.write(i + 1, 1, result_psnr[i])  # 第2列数量
        # sheet1.write(i + 1, 2, result_ssim[i])  # 第3列误差
        # print(i+1)
    file.save(excel_savepath)


# 读取所有图片 PSNR SSIM 25波段结果
def readGreyimgtoarr(testpath, gtpath):
    test_foldlist = os.listdir(testpath)
    true_foldlist = os.listdir(gtpath)
    test_number = len(test_foldlist)
    # print(test_number)

    test_number = 1
    # 获取测试结果 存在数组中
    testdata = np.zeros((test_number, 512, 512, test_number * 25), dtype=np.float32)
    trueimg = np.zeros((test_number, 512, 512, test_number * 25), dtype=np.float32)

    for i in range(0, test_number):
        # 图片路径
        test_filelistpath = os.path.join(testpath, test_foldlist[i]) # D:\bella\results\模型3-大论文结构\DUF_Mosaic_test256-000\visualization\testresult\000
        true_filelistpath = os.path.join(gtpath, true_foldlist[i])
        test_imglist = os.listdir(test_filelistpath)  # 1.png
        test_imglist.sort(key=lambda x: int(x.split('.')[0]))
        true_imglist = os.listdir(true_filelistpath)  # 1.png
        true_imglist.sort(key=lambda x: int(x.split('.')[0]))
        # print(test_imglist, '\t', true_imglist)
        # print(test_filelistpath, test_imglist, true_filelistpath, true_imglist)

        # 图片存在数组中
        for j in range(0, 25):
            testdata_imgpath = os.path.join(test_filelistpath, test_imglist[j])
            test_imgdata = cv2.imread(testdata_imgpath)
            # image1_gray = cv2.cvtColor(test_imgdata, cv2.COLOR_BGR2GRAY)
            image1_gray = test_imgdata
            image1_gray_arr = np.array(image1_gray)
            testdata[i, :, :, i * 25 + j] = image1_gray_arr

            truedata_imgpath = os.path.join(true_filelistpath, true_imglist[j])
            true_imgdata = cv2.imread(truedata_imgpath)
            # image2_gray = cv2.cvtColor(true_imgdata, cv2.COLOR_BGR2GRAY)
            image2_gray = true_imgdata[:, :, 0]
            image2_gray_arr = np.array(image2_gray)
            trueimg[i, :, :, i * 25 + j] = image2_gray_arr
            # print(image1_gray.shape, image2_gray.shape)
    return testdata, trueimg


# 读取所有图片 PSNR SSIM 25波段结果
def readimgtoarr(testpath, gtpath):
    test_foldlist = os.listdir(testpath)
    true_foldlist = os.listdir(gtpath)
    # test_number = len(test_foldlist)
    # print(test_number)

    test_number = 2
    # 获取测试结果 存在数组中
    testdata = np.zeros((test_number, 256, 256, test_number * 25), dtype=np.float32)
    trueimg = np.zeros((test_number, 256, 256, test_number * 25), dtype=np.float32)

    for i in range(0, test_number):
        # 图片路径
        test_filelistpath = os.path.join(testpath, test_foldlist[i]) # D:\bella\results\模型3-大论文结构\DUF_Mosaic_test256-000\visualization\testresult\000
        true_filelistpath = os.path.join(gtpath, true_foldlist[i])
        test_imglist = os.listdir(test_filelistpath)  # 1.png
        test_imglist.sort(key=lambda x: int(x.split('.')[0]))
        true_imglist = os.listdir(true_filelistpath)  # 1.png
        true_imglist.sort(key=lambda x: int(x.split('.')[0]))
        print(test_filelistpath, '\t', true_filelistpath)
        # print(test_filelistpath, test_imglist, true_filelistpath, true_imglist)

        # 图片存在数组中
        for j in range(0, 25):
            testdata_imgpath = os.path.join(test_filelistpath, test_imglist[j])
            test_imgdata = cv2.imread(testdata_imgpath)
            truedata_imgpath = os.path.join(true_filelistpath, true_imglist[j])
            true_imgdata = cv2.imread(truedata_imgpath)
            image1_gray = cv2.cvtColor(test_imgdata, cv2.COLOR_BGR2GRAY)
            image2_gray = cv2.cvtColor(true_imgdata, cv2.COLOR_BGR2GRAY)
            image1_gray_arr = np.array(image1_gray)
            image2_gray_arr = np.array(image2_gray)
            testdata[i, :, :, i * 25 + j] = image1_gray_arr
            trueimg[i, :, :, i * 25 + j] = image2_gray_arr
            # print(image1_gray.shape, image2_gray.shape)

            # test_imgdata = mpimg.imread(testdata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            # # image_gray1 = cv2.imread(testdata_imgpath, 0)    # 整数 (512, 512)
            # # print(image_gray1)
            # test_imgdata_arr = np.array(test_imgdata)  # 小数 (512, 512)
            # # test_imgdata_arr = np.array(test_imgdata)[:, :, 0]  # 小数 (512, 512, 3)
            # # print(test_imgdata_arr)
            # # test_imgdata_index = file.split('.')[0]
            # testdata[i, :, :, i * 25 + j] = test_imgdata_arr

            # truedata_imgpath = os.path.join(true_filelistpath, true_imglist[j])
            # true_imgdata = mpimg.imread(truedata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            # # print(true_imgdata.shape)
            # true_imgdata_arr = np.array(true_imgdata)  # 小数 (512, 512, 3)
            # # print(true_imgdata_arr.shape)
            # trueimg[i, :, :, i * 25 + j] = true_imgdata_arr[:,:,0]
            # print(i, i * 25 + j)

    return testdata, trueimg


# 读取所有图片 PSNR SSIM 25波段结果和平均值
def calculate_psnrssim(testpath, gtpath):
    resultdata = readimgtoarr(test_pic, gtpath)
    testimg, trueimg = resultdata[0], resultdata[1]

    test_foldlist = os.listdir(testpath)
    test_number = len(test_foldlist)
    # print(test_number, len(test_foldlist), testdata.shape)
    PSNR_list_all = list()
    SSIM_list_all = list()

    PSNR_all = 0  # 所有测试集 31个
    SSIM_all = 0  # 所有测试集 31个

    # 设置Excel编码
    file_single = xlwt.Workbook('encoding = utf-8')

    test_number = 2
    for i in range(0, test_number):
        # 计算PSNR SSIM
        PSNR_images = 0  # 单个测试集
        SSIM_images = 0  # 单个测试集
        PSNR_list_singletest = list()
        SSIM_list_singletest = list()

        # 创建sheet工作表
        sheet_singleteatdata = file_single.add_sheet('sheet{}'.format(i + 1), cell_overwrite_ok=True)
        # 先填标题
        # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
        sheet_singleteatdata.write(0, 0, "图片")  # 第1行第1列
        sheet_singleteatdata.write(0, 1, "PSNR")  # 第1行第2列
        sheet_singleteatdata.write(0, 2, "SSIM")  # 第1行第3列

        for n in range(0, 25):
            # 计算PSNR SSIM
            image1 = trueimg[i, :, :, i * 25 + n]
            image2 = testimg[i, :, :, i * 25 + n]

            # PSNR_image = peak_signal_noise_ratio(image1, image2, data_range=1)
            # SSIM_image = structural_similarity(image1, image2, data_range=1)
            PSNR_image = mypsnr(image1, image2, data_range=1)
            SSIM_image = structural_similarity(image1, image2, data_range=1)
            # 25波段的总和
            SSIM_images += SSIM_image
            PSNR_images += PSNR_image
            PSNR_list_singletest.append(PSNR_image)
            SSIM_list_singletest.append(SSIM_image)
            # 保存
            print(i + 1, n + 1, "\tPSNR_image：{:.4f}".format(PSNR_image), "\tSSIM_image：{:.4f}".format(SSIM_image))
            sheet_singleteatdata.write(n + 1, 0, n + 1)  # 第1列序号
            sheet_singleteatdata.write(n + 1, 1, PSNR_list_singletest[n])  # 第2列数量
            sheet_singleteatdata.write(n + 1, 2, SSIM_list_singletest[n])  # 第3列误差

        # 单个测试集 25波段的平均值
        PSNR_images_aver_single = PSNR_images / 25
        SSIM_images_aver_single = SSIM_images / 25
        sheet_singleteatdata.write(27, 0, '平均值')  # 第1列序号
        sheet_singleteatdata.write(27, 1, PSNR_images_aver_single)  # 第2列数量
        sheet_singleteatdata.write(27, 2, SSIM_images_aver_single)  # 第3列误差
        excel_savepath_single = os.path.dirname(testpath) + '\\PSNR_SSIM_single.xls'
        file_single.save(excel_savepath_single)

        # 31组测试集总和
        PSNR_all += PSNR_images_aver_single
        SSIM_all += SSIM_images_aver_single
        # 31组测试集结果 每个测试集的 25波段平均值
        PSNR_list_all.append(PSNR_images_aver_single)
        SSIM_list_all.append(SSIM_images_aver_single)
        print(i + 1, "\tPSNR_images_aver_single：{:.4f}".format(PSNR_images_aver_single),
              "\tSSIM_images_aver_single：{:.4f}".format(SSIM_images_aver_single))

    PSNR_all_aver = PSNR_all / test_number
    SSIM_all_aver = SSIM_all / test_number
    PSNR_list_all.append(PSNR_all_aver)
    SSIM_list_all.append(SSIM_all_aver)
    # print('PSNR_list_all', PSNR_list_all, '\tlen(PSNR_list)', len(PSNR_list_all))
    # print('SSIM_list_all', SSIM_list_all, '\tlen(SSIM_list)', len(SSIM_list_all))

    file_all = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet_all = file_all.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet_all.write(0, 0, "测试集")  # 第1行第1列
    sheet_all.write(0, 1, "PSNR")  # 第1行第2列
    sheet_all.write(0, 2, "SSIM")  # 第1行第3列
    for i in range(test_number + 1):
        if i == test_number:
            sheet_all.write(i + 1, 0, '平均值')  # 第1列序号
            sheet_all.write(i + 1, 1, PSNR_all_aver)  # 第2列数量
            sheet_all.write(i + 1, 2, SSIM_all_aver)  # 第3列误差
        else:
            sheet_all.write(i + 1, 0, i)  # 第1列序号
            sheet_all.write(i + 1, 1, PSNR_list_all[i])  # 第2列数量
            sheet_all.write(i + 1, 2, SSIM_list_all[i])  # 第3列误差
        # print(i+1)

    excel_savepath = os.path.dirname(testpath) + '\\PSNR_SSIM_all.xls'
    file_all.save(excel_savepath)
    # print("done")


# 读取所有图片 PSNR SSIM SAM 25波段平均值 保存NPY文件和excel
def calculate_psnraver(testpath, gtpath):
    resultdata = readGreyimgtoarr(test_pic, gtpath)
    testimg, trueimg = resultdata[0], resultdata[1]

    test_foldlist = os.listdir(testpath)
    test_number = len(test_foldlist)
    # print(test_number, len(test_foldlist), testdata.shape)
    PSNR_list_all = list()
    PSNR_all = 0  # 所有测试集 31个

    # 设置Excel编码
    file_single = xlwt.Workbook('encoding = utf-8')

    test_number = 1
    for i in range(0, test_number):
        # 计算PSNR SSIM
        PSNR_images = 0  # 单个测试集
        PSNR_list_singletest = list()

        # 创建sheet工作表
        sheet_singleteatdata = file_single.add_sheet('sheet{}'.format(i + 1), cell_overwrite_ok=True)
        # 先填标题
        # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
        sheet_singleteatdata.write(0, 0, "图片")  # 第1行第1列
        sheet_singleteatdata.write(0, 1, "PSNR")  # 第1行第2列

        for n in range(0, 25):
            # 计算PSNR SSIM
            image1 = trueimg[i, :, :, i * 25 + n]
            image2 = testimg[i, :, :, i * 25 + n]

            PSNR_image = mypsnr(image1, image2)
            # SSIM_image = structural_similarity(image1, image2, data_range=1, full=True)
            # 25波段的总和
            PSNR_images += PSNR_image
            PSNR_list_singletest.append(PSNR_image)
            # 保存
            print(i+1, n + 1, "\tPSNR_image：{:.4f}".format(PSNR_image))
            sheet_singleteatdata.write(n + 1, 0, n+1)  # 第1列序号
            sheet_singleteatdata.write(n + 1, 1, PSNR_list_singletest[n])  # 第2列数量

        # 单个测试集 25波段的平均值
        PSNR_images_aver_single = PSNR_images / 25
        sheet_singleteatdata.write(27, 0, '平均值')  # 第1列序号
        sheet_singleteatdata.write(27, 1, PSNR_images_aver_single)  # 第2列数量
        excel_savepath_single = os.path.dirname(testpath) + '\\PSNR_256_0.075.xls'
        file_single.save(excel_savepath_single)

        # 31组测试集总和
        PSNR_all += PSNR_images_aver_single
        # 31组测试集结果 每个测试集的 25波段平均值
        PSNR_list_all.append(PSNR_images_aver_single)
        print(i + 1, "\tPSNR_single：{:.4f}".format(PSNR_images_aver_single))

    PSNR_all_aver = PSNR_all / test_number
    PSNR_list_all.append(PSNR_all_aver)
    # print('PSNR_list_all', PSNR_list_all, '\tlen(PSNR_list)', len(PSNR_list_all))

    file_all = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet_all = file_all.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet_all.write(0, 0, "测试集")  # 第1行第1列
    sheet_all.write(0, 1, "PSNR")  # 第1行第2列
    for i in range(test_number+1):
        if i == test_number:
            sheet_all.write(i + 1, 0, '平均值')  # 第1列序号
            sheet_all.write(i + 1, 1, PSNR_all_aver)  # 第2列数量
        else:
            sheet_all.write(i + 1, 0, i)  # 第1列序号
            sheet_all.write(i + 1, 1, PSNR_list_all[i])  # 第2列数量
        # print(i+1)

    excel_savepath = os.path.dirname(testpath) + '\\PSNRall_256_0.075.xls'
    file_all.save(excel_savepath)
    # print("done")


# 读取所有图片 PSNR SSIM 25波段结果和平均值
def calculate_ssimaver(testpath, gtpath):
    resultdata = readGreyimgtoarr(test_pic, gtpath)
    testimg, trueimg = resultdata[0], resultdata[1]

    test_foldlist = os.listdir(testpath)
    test_number = len(test_foldlist)
    # print(test_number, len(test_foldlist), testdata.shape)
    SSIM_list_all = list()
    SSIM_all = 0  # 所有测试集 31个

    # 设置Excel编码
    file_single = xlwt.Workbook('encoding = utf-8')

    # test_number = 2
    for i in range(0, test_number):
        # 计算SSIM
        SSIM_images = 0  # 单个测试集
        SSIM_list_singletest = list()

        # 创建sheet工作表
        sheet_singleteatdata = file_single.add_sheet('sheet{}'.format(i + 1), cell_overwrite_ok=True)
        # 先填标题
        # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
        sheet_singleteatdata.write(0, 0, "图片")  # 第1行第1列
        sheet_singleteatdata.write(0, 1, "SSIM")  # 第1行第3列

        for n in range(0, 25):
            # 计算PSNR SSIM
            image1 = trueimg[i, :, :, i * 25 + n]
            image2 = testimg[i, :, :, i * 25 + n]

            # PSNR_image = peak_signal_noise_ratio(image1, image2, data_range=1)
            SSIM_image = structural_similarity(image1, image2, data_range=1)
            # 25波段的总和
            SSIM_images += SSIM_image
            SSIM_list_singletest.append(SSIM_image)
            # 保存
            print(i+1, n + 1, "\tSSIM_image：{:.4f}".format(SSIM_image))
            sheet_singleteatdata.write(n + 1, 0, n+1)  # 第1列序号
            sheet_singleteatdata.write(n + 1, 1, SSIM_list_singletest[n])  # 第3列误差

        # 单个测试集 25波段的平均值
        SSIM_images_aver_single = SSIM_images / 25
        sheet_singleteatdata.write(27, 0, '平均值')  # 第1列序号
        sheet_singleteatdata.write(27, 1, SSIM_images_aver_single)  # 第3列误差
        excel_savepath_single = os.path.dirname(testpath) + '\\SSIM_256_0.05.xls'
        file_single.save(excel_savepath_single)

        # 31组测试集总和
        SSIM_all += SSIM_images_aver_single
        # 31组测试集结果 每个测试集的 25波段平均值
        SSIM_list_all.append(SSIM_images_aver_single)
        print(i + 1, "\tSSIM_images_aver_single：{:.4f}".format(SSIM_images_aver_single))

    SSIM_all_aver = SSIM_all / test_number
    SSIM_list_all.append(SSIM_all_aver)
    # print('PSNR_list_all', PSNR_list_all, '\tlen(PSNR_list)', len(PSNR_list_all))
    # print('SSIM_list_all', SSIM_list_all, '\tlen(SSIM_list)', len(SSIM_list_all))

    file_all = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet_all = file_all.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet_all.write(0, 0, "测试集")  # 第1行第1列
    sheet_all.write(0, 1, "SSIM")  # 第1行第3列
    for i in range(test_number+1):
        if i == test_number:
            sheet_all.write(i + 1, 0, '平均值')  # 第1列序号
            sheet_all.write(i + 1, 1, SSIM_all_aver)  # 第3列误差
        else:
            sheet_all.write(i + 1, 0, i)  # 第1列序号
            sheet_all.write(i + 1, 1, SSIM_list_all[i])  # 第3列误差
        # print(i+1)

    excel_savepath = os.path.dirname(testpath) + '\\SSIMall_256_0.05.xls'
    file_all.save(excel_savepath)
    # print("done")


# 读取所有图片 SAM 25波段平均值
def calculate_samaver(testpath, gtpath):
    resultdata = readGreyimgtoarr(test_pic, gtpath)
    testimg, trueimg = resultdata[0], resultdata[1]

    test_foldlist = os.listdir(testpath)
    test_number = len(test_foldlist)
    # print(test_number, len(test_foldlist), testdata.shape)
    SAM_list_all = list()
    SAM_all = 0  # 所有测试集 31个

    test_number = 1
    for i in range(0, test_number):
        # 计算PSNR SSIM
        # x_pred = testimg[i, :, :, :]
        # x_true = trueimg[i, :, :, :]
        x_pred1 = testimg[i, :, :, i*25:(i+1)*25]
        x_true1 = trueimg[i, :, :, i*25:(i+1)*25]
        # print(i, i*25)
        sam_rad = np.zeros((512, 512), dtype=np.float32)
        for x in range(512):
            for y in range(512):
                tmp_pred = x_pred1[x, y].ravel()
                tmp_true = x_true1[x, y].ravel()
                np.nan_to_num(norm(tmp_pred), nan=0.5)
                np.nan_to_num(norm(tmp_true), nan=0.5)
                sam_rad[x, y] = np.arccos((tmp_pred * tmp_true).sum() / (norm(tmp_pred) * norm(tmp_true)))
        np.nan_to_num(sam_rad, nan=0.5)
        sam_deg = sam_rad.mean() * 180 / np.pi
        SAM_images = sam_deg / 25
        SAM_all += SAM_images
        SAM_list_all.append(SAM_images)
        print(SAM_images)
    SAM_all_aver = SAM_all / test_number
    print('SAM_list', SAM_list_all)
    print('SAM :', SAM_all_aver)

    # 设置Excel编码
    file_sam = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet_all = file_sam.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet_all.write(0, 0, "测试集")  # 第1行第1列
    sheet_all.write(0, 1, "SAM")  # 第1行第2列
    for i in range(test_number+1):
        if i == test_number:
            sheet_all.write(i + 1, 0, '平均值')  # 第1列序号
            sheet_all.write(i + 1, 1, SAM_all_aver)  # 第2列数量
        else:
            sheet_all.write(i + 1, 0, i)  # 第1列序号
            sheet_all.write(i + 1, 1, SAM_list_all[i])  # 第2列数量
        # print(i+1)

    excel_savepath = os.path.dirname(testpath) + '\\SAM_512_0.xls'
    file_sam.save(excel_savepath)
    # # print("done")



if __name__ == '__main__':
    test_pic = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\6-superresolution_512_2x-256x2\test256'
    true_pic = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_test\GT_512'
    true_pic1 = r'D:\BasicSR\results\test256_400-640-finalresult\GT_512-460-700'

    test_npy = r'D:\bella\results\Mosaic_x4_BN\DUF_Mosaic_16_x4_25_512-1\test256_LR512_15.npy'
    true_npy = r'D:\bella\datasets\test256\test256-测试集的GT-不同文件个数'
    test_pic1 = r'D:\bella\results\Mosaic_x4_conv\DUF_Mosaic_16_x4_25_512-10\test128_crop'
    # readdata(true_pic)

    """指标"""
    # calculate_img_aver(test_pic, true_pic1)
    # readimgall(test_pic, true_pic)
    calculate_samaver(test_pic, true_pic)
    # calculate_psnraver(test_pic, true_pic)
    # calculate_ssimaver(test_pic, true_pic)

