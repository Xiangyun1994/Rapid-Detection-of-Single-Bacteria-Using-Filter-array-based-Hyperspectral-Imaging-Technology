from os import path as osp
from PIL import Image
import os


def generate_meta_info_mosaicimage():
    """Generate meta info for mosaic image dataset.  马赛克图片
    """

    gt_folder = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\mosaic_valid\GT_2048\\'
    meta_info_txt = r'D:\BasicSR\datasets\Mosaic Image512\super-resolution\meta_info_mosaic_valid_2048_GT.txt'
    pathList = os.listdir(gt_folder)  # 子文件夹名字 ['000', ..., '209']
    pathList.sort(key=lambda x: int(x))  # 子文件夹名字 ['000', ..., '209']
    # print(pathList)
    for allFile in pathList:  # 按顺序取路径
        single_file = gt_folder + str(allFile)  # 子文件夹全路径 D:/BasicSR/datasets/mosaic image/train/GT/000
        img_list = sorted(list(os.listdir(single_file)))  # ['00000000.png',, ..., '00000024.png']
        # print(single_file, img_list)
        with open(meta_info_txt, 'a') as f:
            for idx, single_png in enumerate(img_list):  # idx: 0-24, single_png:00000000.png-00000024.png
                # print(single_png)
                img = Image.open(osp.join(single_file, single_png))  # lazy load D:/BasicSR/datasets/REDS/train_sharp/053\00000000.png
                width, height = img.size  # 200*200
                # print(width, height)
                mode = img.mode  # 1
                # print(mode)
                if mode == 'RGB':
                    n_channel = 3
                elif mode == 'L':
                    n_channel = 1
                else:
                    raise ValueError(f'Unsupported mode {mode}.')
                info = f'{str(allFile)} {25} ({height},{width},{n_channel})'
                # print(idx)
                if idx+1 == 25:
                    # print("done")
                    print(info)
                    f.write(f'{info}\n')
                    f.flush()


if __name__ == '__main__':
    generate_meta_info_mosaicimage()