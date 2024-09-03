import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import spectral as spy
from scipy.io import loadmat

"""
根据mat数据画光谱-理想光谱曲线  鼠标取点
"""

# 假设光谱数据是一个三维数组，形状为 (width, height, spectral_dim)
# 这里假设光谱数据存储在一个名为 spectrum_cube 的变量中

# 选择要显示的光谱维度
spectral_dim_to_display = 8
# spectrum_cube = loadmat('E:\\pycode\\multistitch\\show_mat\\Indian_pines.mat')['indian_pines']
spectrum_cube = loadmat('E:\\pycode\\multistitch\\show_mat\\Indian_pines.mat')['data']

# 定义图像大小
fig, ax = plt.subplots(figsize=(8, 6))
aa = spectrum_cube[:, :, spectral_dim_to_display]
# 初始显示光谱数据的二维图像
img = ax.imshow(spectrum_cube[:, :, spectral_dim_to_display], cmap='viridis')
ax.set_title(f'Spectral Dimension {spectral_dim_to_display}')

# 添加光标
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# 显示光谱曲线的图表
fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 4))
lines = []
circles = []


# 点击事件处理函数
def on_click(event):
    if event.inaxes == ax:
        x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
        spectrum_at_pixel = spectrum_cube[y, x, :]

        # 画小圆圈
        circle = plt.Circle((x, y), radius=5, color='red', fill=False)
        ax.add_patch(circle)
        circles.append(circle)

        # 更新光谱曲线
        line, = ax_spectrum.plot(spectrum_at_pixel, label=f'Pixel ({x}, {y}) Spectrum', alpha=0.7)
        lines.append(line)

        ax_spectrum.legend()
        fig_spectrum.canvas.draw()


# 连接点击事件
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
