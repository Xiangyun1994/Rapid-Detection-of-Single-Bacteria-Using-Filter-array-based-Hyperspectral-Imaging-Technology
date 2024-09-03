import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import spectral as spy
from scipy.io import loadmat
from PIL import Image
import os
import scipy.io
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

"""
高光谱去马赛克图片转成三维数据
鼠标曲线绘制坐标点处光谱曲线并保存
"""


# 假设光谱数据是一个三维数组，形状为 (width, height, spectral_dim)
# 这里假设光谱数据存储在一个名为 spectrum_cube 的变量中

def imgto3DArray(folderpath, savepath):
    folder_list = os.listdir(folderpath)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    for i in range(len(folder_list)):
        # for i in range(0, 1):
        filepath = folderpath + '\\' + str(folder_list[i]) + '\\'
        print(filepath)
        imgvec = np.zeros((512, 512, 25), dtype=np.float32)
        for j in range(1, 25):
            imgpath = filepath + '\\' + str(j) + '.png'
            # print(imgpath)
            img = Image.open(imgpath).convert('L')
            imgarr = np.array(img)
            imgvec[:, :, j - 1] = imgarr
            # print(np.array(img).shape)  # (512, 512)
            # print(imgvec.shape)  # (512, 512, 25)
        mat_savepath = savepath + '\\' + f'{i + 1}.mat'
        print(mat_savepath)
        scipy.io.savemat(mat_savepath, mdict={'data': imgvec})
    # return imgvec


def drawspc_png(spectrum_cube, spectral_dim_to_display):
    # 定义图像大小
    fig, ax = plt.subplots(figsize=(8, 6))
    # aa = spectrum_cube[:, :, spectral_dim_to_display]
    # 初始显示光谱数据的二维图像
    img = ax.imshow(spectrum_cube[:, :, spectral_dim_to_display], cmap='viridis')
    ax.set_title(f'Spectral Dimension {spectral_dim_to_display}')

    # 添加光标
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    # 显示光谱曲线的图表1
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4))
    lines = []
    circles = []
    spectrumvec = []

    # 点击事件处理函数
    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            spectrum_at_pixel = spectrum_cube[y, x, :]
            spectrumvec.append(spectrum_at_pixel)
            print(spectrum_at_pixel)
            # 画小圆圈
            circle = plt.Circle((x, y), radius=5, color='red', fill=False)
            ax.add_patch(circle)
            circles.append(circle)

            # 更新光谱曲线
            line, = ax_spectrum.plot(spectrum_at_pixel, label=f'Pixel ({x}, {y}) Spectrum', alpha=0.7)
            lines.append(line)

            ax_spectrum.legend(loc='center right')
            fig_spectrum.canvas.draw()

    # 连接点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()


def drawspc_mat(spectrum_cube_path, spectral_dim_to_display, img_savepath, data_savepath):
    folder_list = os.listdir(spectrum_cube_path)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    # print(folder_list)

    for i in range(9, 10):
        filepath = spectrum_cube_path + '\\' + folder_list[i]
        # print(filepath)
        spectrum_cube = loadmat(filepath)['data']
        data = {}
        specvec = []
        pointsvec = []
        data['dimension'] = spectral_dim_to_display

        # 定义图像大小 ax是光标浮窗  画布
        fig, img = plt.subplots(figsize=(8, 6))
        # aa = spectrum_cube[:, :, spectral_dim_to_display]
        # 初始显示光谱数据的二维图像
        img.imshow(spectrum_cube[:, :, spectral_dim_to_display], cmap='viridis')
        # 显示光谱曲线的图表
        fig_spectrum, img_spectrum = plt.subplots(figsize=(8, 6))

        if i + 1 <= 10:
            img.set_title(f'DragPear Background Image Spectral Dimension {spectral_dim_to_display}')
            img_spectrum.set_title(f'DragPear Background Spectrums Spectral Dimension {spectral_dim_to_display}')
            # img.set_title(f'Pear Drag Image Spectral Dimension {spectral_dim_to_display}')
            # img_spectrum.set_title(f'Pear Drag Spectrums Spectral Dimension {spectral_dim_to_display}')
        elif i + 1 <= 20:
            img.set_title(f'NoDragPear Background Image Spectral Dimension {spectral_dim_to_display}')
            img_spectrum.set_title(f'NoDragPear Background Spectrums Spectral Dimension {spectral_dim_to_display}')
            # img.set_title(f'Pear NoDrag Image Spectral Dimension {spectral_dim_to_display}')
            # img_spectrum.set_title(f'Pear NoDrag Spectrums Spectral Dimension {spectral_dim_to_display}')
        else:
            img.set_title(f'Cucumber Drag Image Spectral Dimension {spectral_dim_to_display}')
            img_spectrum.set_title(f'Cucumber Drag Spectrums Spectral Dimension {spectral_dim_to_display}')
            # img.set_title(f'Background Image Spectral Dimension {spectral_dim_to_display}')
            # img_spectrum.set_title(f'Background Spectrums Spectral Dimension {spectral_dim_to_display}')

        # 添加光标 useblit加快绘图速度
        cursor = Cursor(img, useblit=True, color='red', linewidth=1)

        # 点击事件处理函数
        def on_click(event):
            if event.inaxes == img:
                x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
                pointsvec.append((x,))
                data.update({'points': pointsvec})

                spectrum_at_pixel = spectrum_cube[y, x, :]
                # print(spectrum_at_pixel)
                specvec.append(spectrum_at_pixel)
                data.update({'spectrums': specvec})

                # 画小圆圈
                circle = plt.Circle((x, y), radius=5, color='red', fill=False)
                img.add_patch(circle)

                # 更新光谱曲线
                line, = img_spectrum.plot(spectrum_at_pixel, label=f'Pixel ({x}, {y}) Spectrum', alpha=0.7)
                # ax.plot(spectrum_at_pixel, label=f'Pixel ({x}, {y}) Spectrum', alpha=0.5)
                img_spectrum.legend(loc='right', bbox_to_anchor=(2.39, 1.0), bbox_transform=plt.gcf().transFigure,
                                    borderaxespad=0.1, prop={'size': 10})  # ax.legend(loc='best')
                fig_spectrum.canvas.draw()  # 更新显示
            # print("1", data)

        # 连接点击事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        # print("2", points)
        plt.show()

        # save
        PearDrag_imgpath = r'D:\bella\results\DUF_Mosaic1114_testdata_model50000\visualization\Pear_Drag_img'
        PearNoDrag_imgpath = r'D:\bella\results\DUF_Mosaic1114_testdata_model50000\visualization\Pear_NoDrag_img'
        PearDrag_matpath = r'D:\bella\results\DUF_Mosaic1114_testdata_model50000\visualization\Pear_Drag_matdata'
        PearNoDrag_matpath = r'D:\bella\results\DUF_Mosaic1114_testdata_model50000\visualization\Pear_NoDrag_matdata'
        PearDrag_background_imgpath = img_savepath + '\\Pear_Drag_Background_img'
        PearDrag_background_matpath = data_savepath + '\\Pear_Drag_Background_matdata'
        PearNoDrag_background_imgpath = img_savepath + '\\Pear_NoDrag_Background_img'
        PearNoDrag_background_matpath = data_savepath + '\\Pear_NoDrag_Background_matdata'
        print(i + 1)
        if i + 1 <= 10:
            # 背景
            img.figure.savefig(PearDrag_background_imgpath + f'\\DragPear_Background_Image_{i + 1}.png')
            img_spectrum.figure.savefig(PearDrag_background_imgpath + f'\\DragPear_Background_Spectrum_{i + 1}.png')
            scipy.io.savemat(PearDrag_background_matpath + f'\\DragPear_Background_alldata_{i + 1}.mat', mdict={'data': data})
            scipy.io.savemat(PearDrag_background_matpath + f'\\DragPear_Background_spectrums_{i + 1}.mat', mdict={'spectrum': specvec})
            # #  有农药梨
            # img.figure.savefig(Pear_Dragimgpath + f'\\Pear_Drag_Image_{i + 1}.png')
            # img_spectrum.figure.savefig(Pear_Dragimgpath + f'\\Pear_Drag_Spectrum_{i + 1}.png')
            # scipy.io.savemat(Pear_Drag_matpath + f'\\Pear_Drag_alldata_{i + 1}.mat', mdict={'data': data})
            # scipy.io.savemat(Pear_Drag_matpath + f'\\Pear_Drag_spectrums_{i + 1}.mat', mdict={'spectrum': specvec})
        elif i + 1 <= 20:
            # 背景
            img.figure.savefig(PearNoDrag_background_imgpath + f'\\NoDragPear_Background_Image_{i + 1}.png')
            img_spectrum.figure.savefig(PearNoDrag_background_imgpath + f'\\NoDragPear_Background_Spectrum_{i + 1}.png')
            scipy.io.savemat(PearNoDrag_background_matpath + f'\\NoDragPear_Background_alldata_{i + 1}.mat', mdict={'data': data})
            scipy.io.savemat(PearNoDrag_background_matpath + f'\\NoDragPear_Background_spectrums_{i + 1}.mat',
                             mdict={'spectrum': specvec})
            # # 没农药梨
            # img.figure.savefig(PearNoDrag_imgpath + f'\\Pear_NoDrag_Image_{i + 1}.png')
            # img_spectrum.figure.savefig(PearNoDrag_imgpath + f'\\Pear_NoDrag_Spectrum_{i + 1}.png')
            # scipy.io.savemat(PearNoDrag_matpath + f'\\Pear_NoDrag_alldata_{i + 1}.mat', mdict={'data': data})
            # scipy.io.savemat(PearNoDrag_matpath + f'\\Pear_NoDrag_spectrums_{i + 1}.mat', mdict={'spectrum': specvec})
        else:
            # 有农药黄瓜
            img.figure.savefig(img_savepath + f'\\Cucumber_Drag_Image_{i + 1}.png')
            img_spectrum.figure.savefig(img_savepath + f'\\Cucumber_Drag_Spectrum_{i + 1}.png')
            scipy.io.savemat(data_savepath + f'\\Cucumber_Drag_alldata_{i + 1}.mat', mdict={'data': data})
            scipy.io.savemat(data_savepath + f'\\Cucumber_Drag_spectrums_{i + 1}.mat', mdict={'spectrum': specvec})
            # 没农药黄瓜
            # img.figure.savefig(img_savepath + f'\\Cucumber_Drag_Image_{i + 1}.png')
            # img_spectrum.figure.savefig(img_savepath + f'\\Cucumber_Drag_Spectrum_{i + 1}.png')
            # scipy.io.savemat(data_savepath + f'\\Cucumber_Drag_alldata_{i + 1}.mat', mdict={'data': data})
            # scipy.io.savemat(data_savepath + f'\\Cucumber_Drag_spectrums_{i + 1}.mat', mdict={'spectrum': specvec})
            # 黄瓜背景
            # img.figure.savefig(img_savepath + f'\\Background_Image_{i + 1}.png')
            # img_spectrum.figure.savefig(img_savepath + f'\\Background_Spectrum_{i + 1}.png')
            # scipy.io.savemat(data_savepath + f'\\Background_alldata_{i + 1}.mat', mdict={'data': data})
            # scipy.io.savemat(data_savepath + f'\\Background_spectrums_{i + 1}.mat', mdict={'spectrum': specvec})

        # img.figure.savefig(imgpath)
        # img_spectrum.figure.savefig(img_spectrumpath)
        # scipy.io.savemat(data_savepath + f'\\alldata_{i + 1}.mat', mdict={'data': data})
        # scipy.io.savemat(data_savepath + f'\\spectrums_{i + 1}.mat', mdict={'spectrum': specvec})
        # print(img_savepath + f'\\img_{i + 1}.png')
        # print(img_savepath + f'\\spec_{i + 1}.mat')
        # print(data_savepath + f'\\spectrum_points_{i + 1}.mat')


def getspectrumdata(folderpath, spectrum_cube_path):
    spectral_dim_to_display = 8  # 光谱
    x = 101  # 点x坐标
    y = 178  # 点y坐标
    folder_list = os.listdir(folderpath)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    for i in range(0, 1):
        filepath = folderpath + '\\' + str(folder_list[i]) + '\\'
        print(filepath)
        imgvec = np.zeros((512, 512, 25), dtype=np.float32)
        for j in range(1, 25):
            imgpath = filepath + '\\' + str(j) + '.png'
            # print(imgpath)
            img = Image.open(imgpath).convert('L')
            # (h, w) 高和宽
            imgarr = np.array(img)
            # (h, w， spec) 高和宽 光谱
            imgvec[:, :, j - 1] = imgarr
            # print(np.array(img).shape)  # (512, 512)
            # print(imgvec.shape)  # (512, 512, 25)
        # mat_savepath = savepath + '\\' + f'{i + 1}.mat'
        # print(mat_savepath)
        # scipy.io.savemat(mat_savepath, mdict={'data': imgvec})
        spec_cube1 = imgvec
        spectrum_at_pixel1 = spec_cube1[y, x, :]
        spectrum_at_pixel2 = spec_cube1[x, y, :]
        print(spectrum_at_pixel1)
        print(spectrum_at_pixel2)

    folder_list = os.listdir(spectrum_cube_path)
    folder_list.sort(key=lambda x: int(x.split('.')[0]))
    # print(folder_list)
    for i in range(0, 1):
        filepath = spectrum_cube_path + '\\' + folder_list[i]
        print(filepath)
        spectrum_cube = loadmat(filepath)['data']
        spectrum_at_pixel1 = spectrum_cube[y, x, :]
        print(spectrum_at_pixel1)
        spectrum_at_pixel2 = spectrum_cube[x, y, :]
        print(spectrum_at_pixel2)




if __name__ == '__main__':
    imgdir = r'D:\bella\results\DUF_Mosaic1114_testdata_model50000\visualization\test_result'
    specvec_mat_savepath = r'E:\谭嵋\谭嵋-大论文\数据\真实马赛克数据-测试结果\测试结果\本算法\test256_crop_mat'
    # imgto3DArray(imgdir, specvec_mat_savepath)
    # drawspc_png(imgvec, 8)

    specvec_mat_img_savepath = r'D:\BasicSR\results\模型2-大论文结构\DUF_Mosaic_testdata_model50000\visualization'
    specvec_mat_data_savepath = r'D:\BasicSR\results\模型2-大论文结构\DUF_Mosaic_testdata_model50000\visualization'
    spectral_dim_to_display = 8  # 选择要显示的光谱维度
    drawspc_mat(specvec_mat_savepath, spectral_dim_to_display, specvec_mat_img_savepath, specvec_mat_data_savepath)
    # test_spec_dim(imgdir, specvec_mat_savepath)
