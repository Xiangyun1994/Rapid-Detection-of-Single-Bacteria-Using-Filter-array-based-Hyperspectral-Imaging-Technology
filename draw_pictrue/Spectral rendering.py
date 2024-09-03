import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.io import loadmat
from matplotlib.widgets import Cursor
from matplotlib.font_manager import FontProperties

"""图片点坐标 获取像素点的值 画图"""


def getpoint(imgpath, savepath):
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            print(x, y)
            cv2.circle(img, (x, y), 2, (0, 0, 255))
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
            cv2.imshow("image", img)

    imglist = os.listdir(imgpath)
    for i in range(23, 24):
        imgfullpath = os.path.join(imgpath, imglist[i])
        savefullpath = os.path.join(savepath, imglist[i])
        img = cv2.imread(imgfullpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bright = (img_gray * 1.6).clip(0, 255).astype(np.uint8)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        while (1):
            cv2.imshow("image", img_bright)
            # cv2.imwrite(sevapath, img)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()


"""
在图上画点：在高清图像上标点

"""
def drawpoint(outpath):
    gt_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\GT_mat_1024'

    gt_datalist = os.listdir(gt_inputpath)

    # # 016羽毛数据
    # data_gt = loadmat(os.path.join(gt_inputpath, gt_datalist[3]))['gt_data']
    # # 018花数据
    # data_gt = loadmat(os.path.join(gt_inputpath, gt_datalist[17]))['gt_data']
    # 023调色板数据
    data_gt = loadmat(os.path.join(gt_inputpath, gt_datalist[22]))['gt_data']

    filename, filesuffix = os.path.splitext(gt_datalist[3])
    file_savename = filename + '.png'
    savepath = os.path.join(outpath, file_savename)
    print(savepath)
    spectral_dim_to_display = 19

    # 定义图像大小 ax是光标浮窗  画布
    fig, img = plt.subplots(figsize=(8, 6))
    plt.rc('font', family="Times New Roman")
    # 初始显示光谱数据的二维图像
    img.imshow(data_gt[:, :, spectral_dim_to_display], cmap='gray')
    # 显示光谱曲线的图表
    fig_spectrum, img_spectrum = plt.subplots(figsize=(8, 6))
    # 添加光标 useblit加快绘图速度
    cursor = Cursor(img, useblit=True, color='red', linewidth=1)
    point_list = []
    spectrum_list = []
    numberlist = [1, 2, 3, 4]

    # 点击事件处理函数
    def on_click(event):
        if event.inaxes == img:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            point_list.append({x, y})
            print(point_list)

            spectrum_at_pixel = data_gt[y, x, :]
            spectrum_list.append(spectrum_at_pixel)
            # print(spectrum_at_pixel)

            # 画小圆圈
            circle = plt.Circle((x, y), radius=5, color='red', fill=False)
            # plt.text(x, y, count, fontsize=10, color='red', ha='center', va='left')
            # plt.text(x, y, (x, y), fontsize=18, color='red', ha='center', va='bottom')
            img.add_patch(circle)

            # 更新光谱曲线
            line, = img_spectrum.plot(spectrum_at_pixel, label=f'Pixel ({x}, {y}) Spectrum', alpha=0.7)
            # ax.plot(spectrum_at_pixel, label=f'Pixel ({x}, {y}) Spectrum', alpha=0.5)
            img_spectrum.legend(loc='right', bbox_to_anchor=(2.39, 1.0), bbox_transform=plt.gcf().transFigure,
                                borderaxespad=0.1, prop={'size': 10})  # ax.legend(loc='best')
            # fig_spectrum.legend()
            fig_spectrum.canvas.draw()  # 更新显示

    # 连接点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)
    # print("2", points)
    plt.show()
    # print(point_list)
    # print(spectrum_list)
    plt.imsave(savepath, fig)


"""
根据选取的坐标点画光谱 各算法重建光谱曲线的对比 一种噪声0.025

"""
def drawspectrum():
    gt_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\GT_mat'
    target_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\1_mat\1_test256_0.025'
    chazhi_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\chazhi_mat\0.025'
    duf_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\DUF_mat\DUF_test256_0.025'

    gt_datalist = os.listdir(gt_inputpath)
    target025_datalist = os.listdir(target_inputpath)
    chazhi025_datalist = os.listdir(chazhi_inputpath)
    duf025_datalist = os.listdir(duf_inputpath)
    # print(gt_datalist[10], '\n', target025_datalist[10], '\n', chazhi025_datalist[21], '\n', duf025_datalist[10])

    # 016羽毛数据
    data_gt = loadmat(os.path.join(gt_inputpath, gt_datalist[3]))['gt_data']
    data_target = loadmat(os.path.join(target_inputpath, target025_datalist[3]))['result_data']
    data_chazhi = loadmat(os.path.join(chazhi_inputpath, chazhi025_datalist[3]))['result_data']
    data_duf = loadmat(os.path.join(duf_inputpath, duf025_datalist[3]))['result_data']

    # # 画图展示
    # spectral_dim_to_display = 24
    # image = data_gt[0, :, :, spectral_dim_to_display]
    # plt.imshow(image)
    # plt.show()

    # 坐标点 {58, 326}, {231, 388}, {288, 279}, {364, 336} 备选点{243, 367}
    m = 231
    n = 388
    gt_spectrum = data_gt[n, m, :]
    target_spectrum = data_target[n, m, :]
    chazhi_spectrum = data_chazhi[n, m, :]
    duf_spectrum = data_duf[n, m, :]

    savapath = r'D:\BasicSR\results\test256_finalresult\showresult\017-sprectrums\像素点2.png'
    print("像素点2 坐标 x:", m, " y:", n)
    # plt.rcParams["font.sans-serif"] = ["simsun"]  # 宋体 simsun
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({'font.size': 18})
    print("理想：\t", list(gt_spectrum))
    print("本算法：\t", list(target_spectrum))
    print("插值：\t", list(chazhi_spectrum))
    print("DUF：\t", list(duf_spectrum))
    titlename = '(' + str(m) + ', ' + str(n) + ')'
    plt.title(titlename)
    # 理想光谱-blue 本算法-red 双线性插值算法-green DUF算法-black
    plt.plot(list(range(25)), gt_spectrum, color='blue')
    plt.plot(list(range(25)), target_spectrum, color='red')
    plt.plot(list(range(25)), chazhi_spectrum, color='green')
    plt.plot(list(range(25)), duf_spectrum, color='black')
    plt.xticks([0, 4, 8, 12, 16, 20, 24, 26], ('460', '500', '540', '580', '620', '660', '700', ' '))
    plt.savefig(savapath, dpi=1200)
    plt.show()


"""
根据选取的坐标点画光谱 各算法重建光谱曲线的对比 五种噪声 0 0.025 0.05 0.075 0.1

"""
def drawspectrum_kangzaoxing():
    gt_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\GT_mat'
    target_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\1_mat\1_test256_0.1'
    chazhi_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\chazhi_mat\0.1'
    duf_inputpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\result_mat\DUF_mat\DUF_test256_0.1'

    gt_datalist = os.listdir(gt_inputpath)
    target025_datalist = os.listdir(target_inputpath)
    chazhi025_datalist = os.listdir(chazhi_inputpath)
    duf025_datalist = os.listdir(duf_inputpath)

    # 016羽毛数据
    data_gt = loadmat(os.path.join(gt_inputpath, gt_datalist[3]))['gt_data']
    data_target = loadmat(os.path.join(target_inputpath, target025_datalist[3]))['result_data']
    data_chazhi = loadmat(os.path.join(chazhi_inputpath, chazhi025_datalist[3]))['result_data']
    data_duf = loadmat(os.path.join(duf_inputpath, duf025_datalist[3]))['result_data']

    # # 画图展示
    # spectral_dim_to_display = 24
    # image = data_gt[0, :, :, spectral_dim_to_display]
    # plt.imshow(image)
    # plt.show()

    # 坐标点 {58, 326}, {288, 279}
    m = 288
    n = 279
    gt_spectrum = data_gt[n, m, :]
    target_spectrum = data_target[n, m, :]
    chazhi_spectrum = data_chazhi[n, m, :]
    duf_spectrum = data_duf[n, m, :]

    savapath = r'D:\BasicSR\results\test256_finalresult\showresult\017-sprectrums\像素点3-0.1.png'
    print("像素点1 坐标 x:", m, " y:", n)
    # plt.rcParams["font.sans-serif"] = ["simsun"]  # 宋体 simsun
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({'font.size': 16})

    print("理想：\t", list(gt_spectrum))
    print("本算法：\t", list(target_spectrum))
    print("插值：\t", list(chazhi_spectrum))
    print("DUF：\t", list(duf_spectrum))
    titlename = '(' + str(m) + ', ' + str(n) + ')'
    plt.title(titlename)
    # 理想光谱-blue 本算法-red 双线性插值算法-green DUF算法-black
    plt.plot(list(range(25)), gt_spectrum, color='blue')
    plt.plot(list(range(25)), target_spectrum, color='red')
    plt.plot(list(range(25)), chazhi_spectrum, color='green')
    plt.plot(list(range(25)), duf_spectrum, color='black')
    plt.xticks([0, 4, 8, 12, 16, 20, 24, 26], ('460', '500', '540', '580', '620', '660', '700', ' '))
    plt.savefig(savapath, dpi=1200)
    plt.show()


"""
噪声趋势变化图
根据数值直接画线
"""


def drawNoisechage(outpath):
    # PSNR
    data_PSNR_chazhi = [19.485, 18.847, 18.050, 17.788, 17.413]
    data_PSNR_duf = [17.131, 15.120, 15.102, 15.067, 15.000]
    data_PSNR_target = [25.837, 25.792, 25.092, 24.738, 23.852]
    # SSIM
    data_SSIM_chazhi = [0.303, 0.202, 0.165, 0.143, 0.127]
    data_SSIM_duf = [0.625, 0.620, 0.620, 0.617, 0.616]
    data_SSIM_target = [0.868, 0.866, 0.855, 0.848, 0.831]
    # SAM
    data_SAM_chazhi = [0.776, 0.789, 0.882, 0.977, 1.069]
    data_SAM_duf = [0.888, 0.915, 0.939, 0.962, 0.991]
    data_SAM_target = [0.416, 0.436, 0.460, 0.491, 0.515]
    x_label = [0, 0.25, 0.5, 0.75, 1]

    # plt.title('Spectrum Analysis')
    # plt.rc('font', family="Times New Roman")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.size'] = 16  # 设置字体大小为14
    plt.figure()
    # # PSNR
    # plt.plot(x_label, data_PSNR_chazhi, linestyle='-', color='green', label='双线性插值算法')
    # plt.plot(x_label, data_PSNR_duf, linestyle='-', color='black', label='DUF算法')
    # plt.plot(x_label, data_PSNR_target, linestyle='-', color='red', label='本算法')
    # plt.ylabel('PSNR / dB')
    # plt.xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], ('0', '0.025', '0.05', '0.075', '0.1', ' ', ' '))
    # plt.yticks([14, 16, 18, 20, 22, 24, 26, 28])
    # savename = 'PSNR趋势变化.png'

    # SSIM
    plt.plot(x_label, data_SSIM_chazhi, linestyle='-', color='green', label='双线性插值算法')
    plt.plot(x_label, data_SSIM_duf, linestyle='-', color='black', label='DUF算法')
    plt.plot(x_label, data_SSIM_target, linestyle='-', color='red', label='本算法')
    savename = 'SSIM趋势变化.png'
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.ylabel('SSIM')

    # SAM
    # plt.plot(x_label, data_SAM_chazhi, linestyle='-', color='green', label='双线性插值算法')
    # plt.plot(x_label, data_SAM_duf, linestyle='-', color='black', label='DUF算法')
    # plt.plot(x_label, data_SAM_target, linestyle='-', color='red', label='本算法')
    # savename = 'SAM趋势变化.png'
    # plt.ylabel('SAM')
    # plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])

    savapath = os.path.join(outpath, savename)
    # plt.legend(frameon=False, loc='right')
    plt.xlabel('σ / nm')
    plt.xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], ('0', '0.025', '0.05', '0.075', '0.1', ' ', ' '))
    # plt.yticks([14, 16, 18, 20, 22, 24, 26, 28])
    plt.savefig(savapath, dpi=1200)
    print(savapath)
    plt.show()


"""
抗噪性光谱 图的坐标 图例调整
根据情况画不同噪声水平下各算法某点处的光谱曲线对比图
四个列表存放不同噪声水平下各算法某点处的25歌光谱值
"""


def drawnspectrum_noise_point3(outpath):
    # 噪声0  0.025  0.05  0.075  0.1
    # gt_0为测试集噪声水平为0下理想光谱值列表
    gt_0 = []
    chazhi_0 = []
    duf_0 = []
    target_0 = []
    # plt.title('Spectrum Analysis')
    x_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.size'] = 14  # 设置字体大小为14

    # 噪声0  0.025  0.05  0.075  0.1
    plt.plot(x_list, gt_0, linestyle='-', color='blue', label='理想光谱')
    plt.plot(x_list, target_0, linestyle='-', color='red', label='本算法')
    plt.plot(x_list, chazhi_0, linestyle='-', color='green', label='双线性插值算法')
    plt.plot(x_list, duf_0, linestyle='-', color='black', label='DUF算法')
    plt.yticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 325])
    savename = '0\\图\\大论文展光谱曲线图-像素点3-有图例.png'
    # savename = '0.025\\图\\大论文展光谱曲线图-像素点3-没图例.png'
    # savename = '0.05\\图\\大论文展光谱曲线图-像素点3-没图例.png'
    # savename = '0.075\\图\\大论文展光谱曲线图-像素点4-没图例.png'
    # savename = '0.1\\图\\大论文展光谱曲线图-像素点4-没图例.png'

    savapath = os.path.join(outpath, savename)
    plt.legend(frameon=False, loc='best')
    plt.xticks([0, 4, 8, 12, 16, 20, 24, 28], ('460', '500', '540', '580', '620', '660', '700', ' '))
    plt.xlabel('波长 / nm')
    plt.savefig(savapath, dpi=1200)
    print(savapath)
    plt.show()


"""光谱图 真实图像"""
def drawspectrum_zhenshiimg():
    matdatapath = r'D:\BasicSR\results\模型2-大论文结构\DUF_Mosaic_testdata_model50000\visualization\test256_crop_mat'
    matdatalist = os.listdir(matdatapath)  # 前缀 顺序：1 10 2 3 4 5 6 7 8 9，最后一个spectrums
    matdatalist.sort(key=lambda x: int(x.split('.')[0]))
    drag_pearpath = os.path.join(matdatapath, matdatalist[9])
    nodrag_pearpath = os.path.join(matdatapath, matdatalist[15])
    cucumberpath = os.path.join(matdatapath, matdatalist[22])
    drag_pear = loadmat(drag_pearpath)['data']
    nodrag_pear = loadmat(nodrag_pearpath)['data']
    cucumber = loadmat(cucumberpath)['data']
    # # 梨的坐标 (240,342) (262, 190) (319, 437) (346, 172)
    # m = 126
    # n = 281
    # drag_pear_spectrum = drag_pear[n, m, :]
    # nodrag_pear_spectrum = nodrag_pear[n, m, :]

    # 黄瓜坐标
    m_drag = 298
    n_drag = 99

    m_nodrag = 257
    n_nodrag = 392

    m_background = 394
    n_background = 181

    dragcucumber_spectrum = cucumber[n_drag, m_drag, :]
    nodragcucumber_spectrum = cucumber[n_nodrag, m_nodrag, :]
    backgroundcucumber_spectrum = cucumber[n_background, m_background, :]
    x_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.size'] = 14  # 设置字体大小为14
    # plt.plot(x_list, drag_pear_spectrum, linestyle='-', color='blue', label='有农药1号梨')
    # plt.plot(x_list, nodrag_pear_spectrum, linestyle='-', color='red', label='没农药2号梨')

    plt.plot(x_list, dragcucumber_spectrum, linestyle='-', color='blue', label='有农药黄瓜')
    plt.plot(x_list, nodragcucumber_spectrum, linestyle='-', color='red', label='没农药黄瓜')
    plt.plot(x_list, backgroundcucumber_spectrum, linestyle='-', color='green', label='黄瓜图背景')

    # plt.yticks([0, 25, 50, 75, 100, 125, 150, 175, 200])  # 坐标3
    tilename_drag = '(' + str(m_drag) + ', ' + str(n_drag) + ') (' + str(m_nodrag) + ', ' + str(n_nodrag) + ') (' + str(m_background) + ', ' + str(n_background) + ')'
    savename_drag = '黄瓜光谱曲线图-5.png'
    savepath = 'E:\\谭嵋\\谭嵋-大论文\\数据\\真实马赛克数据-测试结果\\测试结果\\本算法\\第四章光谱对比图\\'
    imgdrag_savepath = os.path.join(savepath, savename_drag)
    print(imgdrag_savepath)

    plt.legend(frameon=False, loc='best')
    plt.xticks([0, 4, 8, 12, 16, 20, 24, 28], ('460', '500', '540', '580', '620', '660', '700', ' '))
    plt.title(tilename_drag)
    plt.savefig(imgdrag_savepath, dpi=1200)
    plt.show()


"""
超分辨重建图像 光谱 根据选取的坐标点画光谱 噪声0

"""
def drawspectrum_superresolution():
    gt2xpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\GT_mat_512'
    gt4xpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\GT_mat_1024'
    gt8xpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\GT_mat_2048'
    target2xpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\result512_mat'
    target4xpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\result1024_mat'
    target8xpath = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\result2048_mat'

    gt2x_datalist = os.listdir(gt2xpath)
    target2x_datalist = os.listdir(target2xpath)
    target4x_datalist = os.listdir(target4xpath)
    target8x_datalist = os.listdir(target8xpath)

    # 018羽毛数据
    gt2x_018 = loadmat(os.path.join(gt2xpath, gt2x_datalist[17]))['gt_data']
    gt2x_023 = loadmat(os.path.join(gt2xpath, gt2x_datalist[22]))['gt_data']

    target2x_018 = loadmat(os.path.join(target2xpath, target2x_datalist[17]))['test_data']
    target2x_023 = loadmat(os.path.join(target2xpath, target2x_datalist[22]))['test_data']

    target4x_018 = loadmat(os.path.join(target4xpath, target4x_datalist[17]))['test_data']
    target4x_023 = loadmat(os.path.join(target4xpath, target4x_datalist[22]))['test_data']

    target8x_018 = loadmat(os.path.join(target8xpath, target8x_datalist[17]))['test_data']
    target8x_023 = loadmat(os.path.join(target8xpath, target8x_datalist[22]))['test_data']

    m_2x = 395
    n_2x = 407
    m_4x = 790
    n_4x = 814
    m_8x = 1580
    n_8x = 1628

    gt2x_023_spectrum = gt2x_023[n_2x, m_2x, :]
    target2x_023_spectrum = target2x_023[n_2x, m_2x, :]
    target4x_023_spectrum = target4x_023[n_4x, m_4x, :]
    target8x_023_spectrum = target8x_023[n_8x, m_8x, :]

    # 归一化图
    gt2x_023_spectrum_arr = np.array(gt2x_023_spectrum)
    target2x_023_spectrum_arr = np.array(target2x_023_spectrum)
    target4x_023_spectrum_arr = np.array(target4x_023_spectrum)
    target8x_023_spectrum_arr = np.array(target8x_023_spectrum)
    gt2x_023_spectrum1 = (gt2x_023_spectrum_arr - gt2x_023_spectrum_arr.min()) / (gt2x_023_spectrum_arr.max() - gt2x_023_spectrum_arr.min())
    target2x_023_spectrum1 = (target2x_023_spectrum_arr - target2x_023_spectrum_arr.min()) / (target2x_023_spectrum_arr.max() - target2x_023_spectrum_arr.min())
    target4x_023_spectrum1 = (target4x_023_spectrum_arr - target4x_023_spectrum_arr.min()) / (target4x_023_spectrum_arr.max() - target4x_023_spectrum_arr.min())
    target8x_023_spectrum1 = (target8x_023_spectrum_arr - target8x_023_spectrum_arr.min()) / (target8x_023_spectrum_arr.max() - target8x_023_spectrum_arr.min())

    gt2x_023_spectrum1 = np.around(gt2x_023_spectrum1, decimals=7)
    target2x_023_spectrum1 = np.around(target2x_023_spectrum1, decimals=7)
    target4x_023_spectrum1 = np.around(target4x_023_spectrum1, decimals=7)
    target8x_023_spectrum1 = np.around(target8x_023_spectrum1, decimals=7)

    savapath1 = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\spectrum\023点2-归一化-1.png'
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams.update({'font.size': 20})
    print("理想：\t", list(gt2x_023_spectrum1))
    print("2倍：\t", list(target2x_023_spectrum1))
    print("4倍：\t", list(target4x_023_spectrum1))
    print("8倍：\t", list(target8x_023_spectrum1))
    titlename = 'normalize (' + str(m_2x) + ', ' + str(n_2x) + ')'
    plt.title(titlename)
    # 理想光谱-blue 本算法-red 双线性插值算法-green DUF算法-black
    plt.plot(list(range(25)), gt2x_023_spectrum1, color='blue', label='GT')
    plt.plot(list(range(25)), target2x_023_spectrum1, color='red', label='2x')
    plt.plot(list(range(25)), target4x_023_spectrum1, color='green', label='4x')
    plt.plot(list(range(25)), target8x_023_spectrum1, color='m', label='8x')
    plt.legend(frameon=False, loc='best')
    plt.xticks([0, 4, 8, 12, 16, 20, 24, 26], ('460', '500', '540', '580', '620', '660', '700', ' '))
    plt.savefig(savapath1, dpi=1200)
    plt.show()


if __name__ == '__main__':
    input_dir = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_test\GT_512'
    save_dir = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\超分辨重建结果\spectrum\1024-理想光谱与标记点'
    # drawpoint(save_dir)
    # drawspectrum()
    # drawspectrum_kangzaoxing()
    drawspectrum_superresolution()

    input_dir1 = r'D:\BasicSR\results\test256_finalresult\showresult\五种噪声-光谱曲线图\spectrum\0\matdata'
    # drawspectrum_kangzaoxing(input_dir1)

    zhibiao_savepath = r'D:\BasicSR\results\test256_finalresult\showresult\五种噪声-三个指标'
    # drawNoisechage(zhibiao_savepath)

    spectrum_nosie = r'E:\谭嵋\谭嵋-大论文\数据\模拟马赛克数据-测试结果\五种噪声-光谱曲线图\spectrum'
    # drawnspectrum_noise_point3(spectrum_nosie)
    # drawspectrum_zhenshiimg()
    # drawspectrum_zhenshiimg()