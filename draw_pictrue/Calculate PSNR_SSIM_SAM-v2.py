from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
from numpy.linalg import norm
import os
import matplotlib.image as mpimg
import xlwt
from PIL import Image
import cv2

"""
计算各指标并保存-v2
"""

# 所有数据 数据存数组 GT
def read_gtdata_all(gtpath):
    gt_foldlist = os.listdir(gtpath)
    savepath = os.path.dirname(gtpath)
    savename = 'test256_GT512_' + str(len(gt_foldlist)) + '.npy'
    savepath = os.path.join(savepath, savename)
    # print(len(gt_foldlist), truedata.shape)
    gt_number = len(gt_foldlist)
    truedata = np.zeros((gt_number, 512, 512, gt_number * 25), dtype=np.float32)  # 31 (31, 512, 512, 25)
    for i in range(0, gt_number):
        # 图片路径
        gt_filelistpath = os.path.join(gtpath, gt_foldlist[i])  # D:\bella\datasets\test256\GT_512\000
        gt_imglist = os.listdir(gt_filelistpath)  # 00000001.png
        gt_imglist.sort(key=lambda x: int(x.split('.')[0]))
        # 图片存在数组中
        for j in range(0, 25):  # for j in range(0, 1):
            gt_imgpath = os.path.join(gt_filelistpath, gt_imglist[j])
            gt_imgdata_arr = np.array(mpimg.imread(gt_imgpath))  # (512, 512)
            # print(gt_imgdata, gt_imgdata_arr)
            truedata[i, :, :, i * 25 + j] = gt_imgdata_arr[:, :]
            # print(i, i*25+j, truedata.shape)
        print(i, '\t', gt_filelistpath)
    # np.save(savepath, truedata)
    print(savepath)
    return truedata


# 所有测试集 PSNR SSIM 测试集的平均值
def cal_psnrssim_all(testpath, gtpath):
    # 读取图片
    # testimg = read_lrdata_all(testpath)   # (1, 512, 512, 25)
    # trueimg = read_gtdata_all(gtpath)      # (512, 512, 25)
    # print(testimg.shape, trueimg.shape)

    testnpyname = os.path.basename(testpath)  # test256_LR256_15.npy
    testnpy_num = (testnpyname.split('_')[-1]).split('.')[0]  # 15
    gtnpyname = 'test256_GT512_' + testnpy_num + '.npy'  # test256_GT512_15.npy
    gtpath1 = os.path.join(gtpath, gtnpyname)
    # print(testnpyname, testnpy_num, gtnpyname)
    print(testpath, gtpath1)

    # 读取npy文件
    testimg = np.load(testpath)
    trueimg = np.load(gtpath1)
    # print(testimg.shape)

    # 计算测试集图像的 PSNR 和 SSIM
    PSNR_list = list()
    SSIM_list = list()
    PSNR_all = 0  # 所有测试集 31个
    SSIM_all = 0  # 所有测试集 31个

    npynumber = 31
    # 个数选择
    if testnpy_num == '6':
        npynumber = 6
    elif testnpy_num == '7':
        npynumber = 7
    elif testnpy_num == '8':
        npynumber = 8
    elif testnpy_num == '15':
        npynumber = 15
    # print(testpath, gtpath1, npynumber)
    for i in range(0, npynumber):
        PSNR_images = 0  # 单个测试集
        SSIM_images = 0  # 单个测试集
        for n in range(25):
            image1 = testimg[i, :, :, n]
            image2 = trueimg[i, :, :, n]
            PSNR_image = peak_signal_noise_ratio(image1, image2, data_range=1)
            PSNR_images += PSNR_image
            SSIM_image = structural_similarity(image1, image2, data_range=1)
            SSIM_images += SSIM_image
        PSNR_images = PSNR_images / 25
        SSIM_images = SSIM_images / 25
        PSNR_all += PSNR_images
        SSIM_all += SSIM_images
        PSNR_list.append(PSNR_images)
        SSIM_list.append(SSIM_images)
        print('PSNR_images: ', PSNR_images, '\tSSIM_images: ', SSIM_images)
    aver_psnr = PSNR_all / npynumber
    aver_ssim = SSIM_all / npynumber
    PSNR_list.append(aver_psnr)
    SSIM_list.append(aver_ssim)
    print('\nPSNR_list', PSNR_list, '\naver_psnr :', aver_psnr, '\tlen(PSNR_list)', len(PSNR_list))
    print('\nSSIM_list', SSIM_list, '\naver_ssim :', aver_ssim, '\tlen(SSIM_list)', len(SSIM_list))
    return PSNR_list, SSIM_list


# 保存数据到excel
def savetoexcel_psnrssim(testpath, gtpath):
    savepath = os.path.dirname(testpath)
    excel_savepath = savepath + '\\PSNR_SSIM.xls'
    print(excel_savepath)

    result = cal_psnrssim_all(testpath, gtpath)
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


# 读取图片 所有测试集 PSNR SSIM 测试集的平均值 SAM 平均值 保存NPY文件和excel
def cal_psnrssim_all_img(testpath, gtpath):
    test_foldlist = os.listdir(testpath)
    test_number = len(test_foldlist)

    # test256npy存放路径
    # npy_savename = 'test256_LR512_8.5w' + str(test_number) + '.npy'
    # testnpy_savepath = os.path.join(os.path.dirname(testpath), npy_savename)
    excel_savepath = os.path.dirname(testpath) + '\\PSNR_SSIM_SAM.xls'

    # 理想图片路径
    gtnpyname = 'GT_512_' + str(test_number)  # test256_GT512_15.npy
    gtpath1 = os.path.join(gtpath, gtnpyname)
    true_foldlist = os.listdir(gtpath1)

    print('\n', testpath, gtpath1)
    # print(testnpy_savepath, excel_savepath)

    # 获取测试结果 存在数组中
    testdata = np.zeros((test_number, 512, 512, test_number * 25), dtype=np.float32)
    trueimg = np.zeros((test_number, 512, 512, test_number * 25), dtype=np.float32)
    # print(len(test_foldlist), testdata.shape)

    PSNR_list = list()
    SSIM_list = list()
    SAM_list = list()
    PSNR_all = 0  # 所有测试集 31个
    SSIM_all = 0  # 所有测试集 31个
    SAM_all = 0
    # test_number = 2
    for i in range(0, test_number):
        # 图片路径
        test_filelistpath = os.path.join(testpath, test_foldlist[i]) # D:\bella\results\模型3-大论文结构\DUF_Mosaic_test256-000\visualization\testresult\000
        true_filelistpath = os.path.join(gtpath1, true_foldlist[i])
        test_imglist = os.listdir(test_filelistpath)  # 1.png
        test_imglist.sort(key=lambda x: int(x.split('.')[0]))
        true_imglist = os.listdir(true_filelistpath)  # 1.png
        true_imglist.sort(key=lambda x: int(x.split('.')[0]))
        # print(test_filelistpath, test_imglist, true_filelistpath, true_imglist)

        # 计算PSNR SSIM
        PSNR_images = 0  # 单个测试集
        SSIM_images = 0  # 单个测试集

        # 图片存在数组中
        for j in range(0, 25):
            testdata_imgpath = os.path.join(test_filelistpath, test_imglist[j])
            test_imgdata = mpimg.imread(testdata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            # image_gray1 = cv2.imread(testdata_imgpath, 0)    # 整数 (512, 512)
            test_imgdata_arr = np.array(test_imgdata)[:, :, 0]  # 小数 (512, 512, 3)
            # test_imgdata_index = file.split('.')[0]
            testdata[i, :, :, i * 25 + j] = test_imgdata_arr

            truedata_imgpath = os.path.join(true_filelistpath, true_imglist[j])
            true_imgdata = mpimg.imread(truedata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            true_imgdata_arr = np.array(true_imgdata)  # 小数 (512, 512, 3)
            trueimg[i, :, :, i * 25 + j] = true_imgdata_arr

            # 计算PSNR SSIM
            image1 = test_imgdata_arr
            image2 = true_imgdata_arr
            PSNR_image = peak_signal_noise_ratio(image1, image2, data_range=1)
            PSNR_images += PSNR_image
            SSIM_image = structural_similarity(image1, image2, data_range=1)
            SSIM_images += SSIM_image

        # 计算25张的SAM
        x_pred = testdata[i, :, :, i*25:(i+1) * 25]
        x_true = trueimg[i, :, :, i*25:(i+1) * 25]
        # print(x_true.shape, x_pred.shape, (i+1) * 25)

        sam_rad = np.zeros((512, 512), dtype=np.float32)
        for x in range(512):
            for y in range(512):
                tmp_pred = x_pred[x, y].ravel()  # 转成一维向量 相当于reshape(-1)
                tmp_true = x_true[x, y].ravel()
                # norm() 计算范数函数 2范数(平方和开平方)
                sam_rad[x, y] = np.arccos((tmp_pred * tmp_true).sum() / (norm(tmp_pred) * norm(tmp_true)))
        sam_deg = sam_rad.mean() * 180 / np.pi      # 25张图 512*512个像素点 的平均光谱夹角

        PSNR_images = PSNR_images / 25
        SSIM_images = SSIM_images / 25
        SAM_images = sam_deg / 25  # 25张图的平均值
        PSNR_all += PSNR_images
        SSIM_all += SSIM_images
        SAM_all += SAM_images
        PSNR_list.append(PSNR_images)
        SSIM_list.append(SSIM_images)
        SAM_list.append(SAM_images)
        print(i, '\t', test_filelistpath)
        print('PSNR_images: ', PSNR_images, '\tSSIM_images: ', SSIM_images, '\tSAM_images: ', SAM_images)
    aver_psnr = PSNR_all / test_number
    aver_ssim = SSIM_all / test_number
    aver_sam = SAM_all / test_number
    print('PSNR_list', PSNR_list, '\taver_psnr :', aver_psnr, '\tlen(PSNR_list)', len(PSNR_list))
    print('SSIM_list', SSIM_list, '\taver_ssim :', aver_ssim, '\tlen(SSIM_list)', len(SSIM_list))
    print('SAM_list', SAM_list, '\taver_sam :', aver_sam, '\tlen(SAM_list)', len(SAM_list))
    # 保存psnr ssim 到excel
    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet1.write(0, 0, "测试集")  # 第1行第1列
    sheet1.write(0, 1, "PSNR")  # 第1行第2列
    sheet1.write(0, 2, "SSIM")  # 第1行第3列
    sheet1.write(0, 3, "SAM")  # 第1行第3列

    # 循环填入数据
    for i in range(test_number+1):
        if i == test_number:
            sheet1.write(i + 1, 0, '平均值')  # 第1列序号
            sheet1.write(i + 1, 1, aver_psnr)  # 第2列数量
            sheet1.write(i + 1, 2, aver_ssim)  # 第3列误差
            sheet1.write(i + 1, 3, aver_sam)  # 第3列误差
        else:
            sheet1.write(i + 1, 0, i)  # 第1列序号
            sheet1.write(i + 1, 1, PSNR_list[i])  # 第2列数量
            sheet1.write(i + 1, 2, SSIM_list[i])  # 第3列误差
            sheet1.write(i + 1, 3, SAM_list[i])  # 第3列误差
        # print(i+1)
    # print("done")
    file.save(excel_savepath)
    # np.save(testnpy_savepath, testdata)


# 读取npy 所有测试集 PSNR SSIM 测试集的平均值 SAM 平均值 保存NPY文件和excel
def cal_sam_all(testpath, gtpath):
    test_foldlist = os.listdir(testpath)
    test_number = len(test_foldlist)

    # test256npy存放路径
    npy_savename = 'test256_LR512_8.5w' + str(test_number) + '.npy'
    testnpy_savepath = os.path.join(os.path.dirname(testpath), npy_savename)
    excel_savepath = os.path.dirname(testpath) + '\\PSNR_SSIM_SAM_8.5w.xls'

    # 理想图片路径
    gtnpyname = 'test256_GT512_' + str(test_number) + '.npy'  # test256_GT512_15.npy
    gtpath1 = os.path.join(gtpath, gtnpyname)

    print('\n', testpath, gtpath1)
    # print(testnpy_savepath, excel_savepath)

    # 获取测试结果 存在数组中
    testdata = np.zeros((test_number, 512, 512, test_number * 25), dtype=np.float32)
    # print(len(test_foldlist), testdata.shape)
    trueimg = np.load(gtpath1)

    PSNR_list = list()
    SSIM_list = list()
    SAM_list = list()
    PSNR_all = 0  # 所有测试集 31个
    SSIM_all = 0  # 所有测试集 31个
    SAM_all = 0
    # test_number = 2
    for i in range(0, test_number):
        # 图片路径
        test_filelistpath = os.path.join(testpath, test_foldlist[i]) # D:\bella\results\模型3-大论文结构\DUF_Mosaic_test256-000\visualization\testresult\000
        test_imglist = os.listdir(test_filelistpath)  # 1.png
        test_imglist.sort(key=lambda x: int(x.split('.')[0]))
        # print(test_filelistpath, test_imglist, true_filelistpath, true_imglist)

        # 计算PSNR SSIM
        PSNR_images = 0  # 单个测试集
        SSIM_images = 0  # 单个测试集

        # 图片存在数组中
        for j in range(0, 25):
            testdata_imgpath = os.path.join(test_filelistpath, test_imglist[j])
            test_imgdata = mpimg.imread(testdata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            # image_gray1 = cv2.imread(testdata_imgpath, 0)    # 整数 (512, 512)
            test_imgdata_arr = np.array(test_imgdata)[:, :, 0]  # 小数 (512, 512, 3)
            # test_imgdata_index = file.split('.')[0]
            testdata[i, :, :, i * 25 + j] = test_imgdata_arr

            # 计算PSNR SSIM
            image1 = test_imgdata_arr
            image2 = trueimg[i, :, :, i * 25 + j]
            PSNR_image = peak_signal_noise_ratio(image1, image2, data_range=1)
            PSNR_images += PSNR_image
            SSIM_image = structural_similarity(image1, image2, data_range=1)
            SSIM_images += SSIM_image

        # 计算25张的SAM
        x_pred = testdata[i, :, :, i*25:(i+1) * 25]
        x_true = trueimg[i, :, :, i*25:(i+1) * 25]
        # print(x_true.shape, x_pred.shape, (i+1) * 25)

        sam_rad = np.zeros((512, 512), dtype=np.float32)
        for x in range(512):
            for y in range(512):
                tmp_pred = x_pred[x, y].ravel()  # 转成一维向量 相当于reshape(-1)
                tmp_true = x_true[x, y].ravel()
                # norm() 计算范数函数 2范数(平方和开平方)
                sam_rad[x, y] = np.arccos((tmp_pred * tmp_true).sum() / (norm(tmp_pred) * norm(tmp_true)))
        sam_deg = sam_rad.mean() * 180 / np.pi      # 25张图 512*512个像素点 的平均光谱夹角

        PSNR_images = PSNR_images / 25
        SSIM_images = SSIM_images / 25
        SAM_images = sam_deg / 25  # 25张图的平均值
        PSNR_all += PSNR_images
        SSIM_all += SSIM_images
        SAM_all += SAM_images
        PSNR_list.append(PSNR_images)
        SSIM_list.append(SSIM_images)
        SAM_list.append(SAM_images)
        print(i, '\t', test_filelistpath)
        print('PSNR_images: ', PSNR_images, '\tSSIM_images: ', SSIM_images, '\tSAM_images: ', SAM_images)
    aver_psnr = PSNR_all / test_number
    aver_ssim = SSIM_all / test_number
    aver_sam = SAM_all / test_number
    print('PSNR_list', PSNR_list, '\taver_psnr :', aver_psnr, '\tlen(PSNR_list)', len(PSNR_list))
    print('SSIM_list', SSIM_list, '\taver_ssim :', aver_ssim, '\tlen(SSIM_list)', len(SSIM_list))
    print('SAM_list', SAM_list, '\taver_sam :', aver_sam, '\tlen(SAM_list)', len(SAM_list))
    # 保存psnr ssim 到excel
    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet1.write(0, 0, "测试集")  # 第1行第1列
    sheet1.write(0, 1, "PSNR")  # 第1行第2列
    sheet1.write(0, 2, "SSIM")  # 第1行第3列
    sheet1.write(0, 3, "SAM")  # 第1行第3列

    # 循环填入数据
    for i in range(test_number+1):
        if i == test_number:
            sheet1.write(i + 1, 0, '平均值')  # 第1列序号
            sheet1.write(i + 1, 1, aver_psnr)  # 第2列数量
            sheet1.write(i + 1, 2, aver_ssim)  # 第3列误差
            sheet1.write(i + 1, 3, aver_sam)  # 第3列误差
        else:
            sheet1.write(i + 1, 0, i)  # 第1列序号
            sheet1.write(i + 1, 1, PSNR_list[i])  # 第2列数量
            sheet1.write(i + 1, 2, SSIM_list[i])  # 第3列误差
            sheet1.write(i + 1, 3, SAM_list[i])  # 第3列误差
        # print(i+1)
    # print("done")
    file.save(excel_savepath)
    # np.save(testnpy_savepath, testdata)


if __name__ == '__main__':
    test_pic = r'D:\bella\results\Mosaic_x4_BN\DUF_Mosaic_16_x4_25_512-1\test128_crop'
    true_pic = r'D:\bella\datasets\test256\test256-测试集的GT-不同文件个数\GT_512_7-003'
    true_pic1 = r'D:\bella\datasets\test256\test256-测试集的GT-不同文件个数'

    test_npy = r'D:\bella\results\Mosaic_x4_BN\DUF_Mosaic_16_x4_25_512-1\test256_LR512_15.npy'
    true_npy = r'D:\bella\datasets\test256\test256-测试集的GT-不同文件个数'
    test_pic1 = r'D:\bella\results\Mosaic_x4_conv\DUF_Mosaic_16_x4_25_512-10\test128_crop'

    """指标"""
