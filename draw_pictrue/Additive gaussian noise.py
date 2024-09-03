import numpy as np
import cv2



# 读取图片
path = r'D:\BasicSR\datasets\test256\1\17.png'
savepath = r'D:\BasicSR\datasets\test256\1\17-0.1.png'
image = cv2.imread(path, cv2.ROTATE_90_CLOCKWISE)

# # 生成高斯噪声
# mean = 0.1
# # var = 0.000625
# # var = 0.0025
# var = 0.005625
# # var = 0.01
#
# sigma = var ** 0.5
# # gaussian = np.random.normal(mean, sigma, image.shape).astype('uint8')
# gaussian = np.random.normal(mean, var ** 0.5, (512, 512))
# # noisy_image = cv2.add(image, gaussian)
# noisy_image = image + gaussian
# img = np.clip(image, 0, 255)
# img = img/255
#
# # 显示带噪声的图像
# cv2.imshow('Noisy Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # cv2.imread(savepath, noisy_image)


# 生成高斯噪声
img = np.asarray(image / 255, dtype=np.float32)
mean = 0.1
stddev = 0.6  # 标准差越大，噪声越明显
noise = np.random.normal(mean, stddev, image.shape).astype("uint8")

# 将原始图像与噪声相加
# result = cv2.addWeighted(image, 1, noise, 1, 0)
result = img + noise
result = np.clip(result, 0, 1)
result = np.uint8(result * 255)
# 保存结果图像
cv2.imwrite(savepath, result)