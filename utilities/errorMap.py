import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_error_map(original_img, superres_img):
    # 计算原始图像与超分辨率图像之间的像素差异
    error_map = np.abs(original_img - superres_img)
    return error_map


def visualize_error_map(original_img, superres_img, error_map):
    # 可视化误差图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Super-Resolved Image')
    plt.imshow(superres_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Error Map')
    plt.imshow(error_map, cmap='jet',vmin=0, vmax=0.1)  # 误差图使用伪彩色表示
    plt.colorbar()
    plt.axis('off')

    plt.show()


# 读取原始图像和超分辨率图像
original_img = cv2.imread('/data2/cjc/dataSets/ixi/test_HR_T2/IXI002-Guys-0828-T2_41.png', cv2.IMREAD_GRAYSCALE)/255
superres_img = cv2.imread('/data2/cjc/MRIPapers/Demo02/SR/BI/OmniSR_X4_DIV2K_test/ixi/x4/IXI002-Guys-0828-T2_41.png', cv2.IMREAD_GRAYSCALE)/255
# 计算误差图
error_map = compute_error_map(original_img, superres_img)

# 可视化结果
visualize_error_map(original_img, superres_img, error_map)
