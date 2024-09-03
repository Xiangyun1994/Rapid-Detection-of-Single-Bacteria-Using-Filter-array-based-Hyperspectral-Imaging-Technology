import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import os
import numpy as np
from numpy.linalg import norm

def showimg():
    # 读取灰度图像 旋转
    image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_ret3 = cv2.rotate(image_gray, cv2.ROTATE_90_CLOCKWISE)
    img_filp = cv2.flip(img_ret3, 1)
    print(img_filp.shape)

    # 显示旋转后的图片
    cv2.imshow('Rotated Image', img_ret3)
    cv2.imshow('filp Image', img_filp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(img_grey, image_gray)


def readimg(path):
    img = cv2.imread(path1)
    img1 = io.imread(path1)
    img2 = plt.imread(path1)
    # print(img1.shape)
    arr = [1,2,3,4,5]
    aver = np.mean(arr)
    print(aver)


# 读取所有图片 PSNR SSIM 25波段结果
def readimgall(testpath, gtpath):
    test_foldlist = os.listdir(testpath)
    true_foldlist = os.listdir(gtpath)
    test_number = len(test_foldlist)
    print(test_number)

    # test_number = 5
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
        print(test_filelistpath, '\t', true_filelistpath)
        # print(test_filelistpath, test_imglist, true_filelistpath, true_imglist)

        # 图片存在数组中
        for j in range(0, 25):
            testdata_imgpath = os.path.join(test_filelistpath, test_imglist[j])
            test_imgdata = mpimg.imread(testdata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            test_imgdata_arr = np.array(test_imgdata)[:, :, 0]  # 小数 (512, 512, 3)
            testdata[i, :, :, i * 25 + j] = test_imgdata_arr
            # image_gray1 = cv2.imread(testdata_imgpath, 0)  # 整数 (512, 512)
            # testdata[i, :, :, i * 25 + j] = image_gray1

            truedata_imgpath = os.path.join(true_filelistpath, true_imglist[j])
            true_imgdata = mpimg.imread(truedata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            true_imgdata_arr = np.array(true_imgdata)  # 小数 (512, 512, 3)
            trueimg[i, :, :, i * 25 + j] = true_imgdata_arr
            # true_imgdata = cv2.imread(truedata_imgpath, 0)
            # trueimg[i, :, :, i * 25 + j] = true_imgdata
            # print(i, i * 25 + j)
            # print(test_imgdata.shape, true_imgdata.shape)

    return testdata, trueimg


# 读取所有图片 PSNR SSIM 25波段结果
def readimgall_chazhi(testpath, gtpath):
    test_foldlist = os.listdir(testpath)
    true_foldlist = os.listdir(gtpath)
    test_number = len(test_foldlist)
    print(test_number)

    # test_number = 5
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
        print(test_filelistpath, '\t', true_filelistpath)
        # print(test_filelistpath, test_imglist, true_filelistpath, true_imglist)

        # 图片存在数组中
        for j in range(0, 25):
            testdata_imgpath = os.path.join(test_filelistpath, test_imglist[j])
            test_imgdata = mpimg.imread(testdata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            test_imgdata_arr = np.array(test_imgdata)  # 小数 (512, 512, 3)
            testdata[i, :, :, i * 25 + j] = test_imgdata_arr

            truedata_imgpath = os.path.join(true_filelistpath, true_imglist[j])
            true_imgdata = mpimg.imread(truedata_imgpath)  # 小数 cv2读取图片时会转成整型数组
            true_imgdata_arr = np.array(true_imgdata)  # 小数 (512, 512, 3)
            trueimg[i, :, :, i * 25 + j] = true_imgdata_arr
            # print(i, i * 25 + j)
            # print(test_imgdata.shape, true_imgdata.shape)

    return testdata, trueimg


def samtest(testpath, gtpath):
    resultdata = readimgall(testpath, gtpath)
    testimg, trueimg = resultdata[0], resultdata[1]
    imgnum = len(os.listdir(testpath))
    print(imgnum)
    test_images = testimg[:, :, :, :25]
    ture_images = trueimg[:, :, :, :25]
    for i in range(imgnum):
        # x_pred = pred_images[i, :, :, :]
        x_pred = test_images[i, :, :, :]
        x_true = ture_images[i, :, :, :]
        sam_1 = np.zeros((1, 6), dtype=np.float32)
        sam_rad = np.zeros((512, 512), dtype=np.float32)
        for x in range(512):
            for y in range(512):
                tmp_pred = x_pred[x, y].ravel()
                tmp_true = x_true[x, y].ravel()
                sam_rad[x, y] = np.arccos((tmp_pred * tmp_true).sum() / (norm(tmp_pred) * norm(tmp_true)))

        # sam_1[0, 0] = sam_rad[50, 50] * 180 / np.pi
        # sam_1[0, 1] = sam_rad[50, 150] * 180 / np.pi
        # sam_1[0, 2] = sam_rad[100, 100] * 180 / np.pi
        # sam_1[0, 3] = sam_rad[150, 50] * 180 / np.pi
        # sam_1[0, 4] = sam_rad[150, 150] * 180 / np.pi
        # sam_deg = sam_rad.mean() * 180 / np.pi
        # sam_1[0, 5] = sam_deg

        sam_rad = np.nan_to_num(sam_rad)
        sam_1[0, 0] = sam_rad[50, 50] * 180 / np.pi / 25
        sam_1[0, 1] = sam_rad[50, 150] * 180 / np.pi / 25
        sam_1[0, 2] = sam_rad[100, 100] * 180 / np.pi / 25
        sam_1[0, 3] = sam_rad[150, 50] * 180 / np.pi / 25
        sam_1[0, 4] = sam_rad[150, 150] * 180 / np.pi / 25
        sam_deg = sam_rad.mean() * 180 / np.pi / 25
        sam_1[0, 5] = sam_deg

        np.set_printoptions(precision=3)
        print('SAM_list_{}'.format(i), np.around(sam_1, 3))

if __name__ == '__main__':
    path = r'D:\bella\datasets\test256_0\000\00000001.png'
    path1 = r'D:\BasicSR\datasets\test100\GT200_460-700\000\00000001.jpg'
    testimg = r'D:\BasicSR\results\test256_400-640-finalresult\test\00'
    trueimg = r'D:\BasicSR\results\test256_400-640-finalresult\test\1'
    samtest(testimg, trueimg)
    # readimg(path1)
