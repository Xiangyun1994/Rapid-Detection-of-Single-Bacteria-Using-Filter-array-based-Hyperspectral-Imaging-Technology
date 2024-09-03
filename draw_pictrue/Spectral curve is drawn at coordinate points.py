import cv2
from PIL import Image

"""
根据坐标点画光谱曲线  鼠标取点
"""

"""图片点坐标 获取像素点的值"""
def on_EVENT_LBUTTONDOWN(event, x, y, img):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(x, y)
        cv2.circle(img, (x, y), 2, (0, 0, 255))
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        cv2.imshow("image", img)

def getPoints(img):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while (1):
        cv2.imshow("image", img)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def drawPoints(input_dir, save_dir):
    # img = readimg(input_dir)
    for i in range(0, 30):
        if i <= 9:
            print(input_dir + '00' + str(i) + '/13.png')
            sevapath = save_dir + '00' + str(i) + '_point.png'
            img = cv2.imread(input_dir + '00' + str(i) + '/13.png')
        elif 10 <= i:
            print(input_dir + '0' + str(i) + '/13.png')
            sevapath = save_dir + '0' + str(i) + '_point.png'
            img = cv2.imread(input_dir + '0' + str(i) + '/13.png')
        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
        while (1):
            cv2.imshow("image", img)
            cv2.imwrite(sevapath, img)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    input_dir = 'D:/BasicSR/results/DUF_Mosaic1229data_28x4_25_512_0305/visualization/test128/111/'
    save_dir = 'D:\\BasicSR\\results\\DUF_Mosaic1229data_28x4_25_512_0305\\visualization\\test128\\111\\points\\'

