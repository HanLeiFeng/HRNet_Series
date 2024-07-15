import cv2

# 读取图像
image = cv2.imread('img/library.jpg', cv2.IMREAD_GRAYSCALE)
# 应用Sobel算子检测边缘
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 合并两个方向的边缘
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
# 计算灰度共生矩阵
glcm = cv2.HuMoments(image)
# 显示图像
cv2.imshow('Structural Texture', sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('structural_texture.png', sobel)


# 保存图像到img文件夹
output_path = 'img/structural_texture.png'
cv2.imwrite(output_path, sobel)

# 请注意，如果img文件夹不存在，您需要先创建它。