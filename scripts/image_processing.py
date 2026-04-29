import cv2
import numpy as np
import os
import time
from numpy import float32, uint8
from numpy.typing import NDArray

# путь к изображениям
PAINTINGS_DIR = "paintings"


# 1 приведение цветного изображения к полутоновому
def grayscale_manual(image: NDArray[uint8]) -> NDArray[uint8]:
    weights = np.array((0.114, 0.587, 0.299), float32)
    return np.clip(image @ weights, 0, 255).astype(uint8)

def sliding_window_view(image, kernel_shape):
    kernel_h, kernel_w = kernel_shape
    # сколько раз ядро поместится по h,w
    count_h = image.shape[0] - kernel_h + 1
    count_w = image.shape[1] - kernel_w + 1
    # 4х мерный массив для хранения всех окон
    windows = np.zeros((count_h, count_w, kernel_h, kernel_w), dtype=image.dtype)
    for i in range(count_h):
        for j in range(count_w):
            windows[i, j] = image[i:i+kernel_h, j:j+kernel_w]
    return windows

# 2 свёртка и использованием двумерной маски
def convolve_manual(image: NDArray[uint8], kernel: NDArray[float32]) -> NDArray[float32]:
    kernel_shape = kernel.shape
    # вычисляем "радиус" чтобы понять, на сколько расширить картинку
    padded_height, padded_width = kernel_shape[0] // 2, kernel_shape[1] // 2
    # добавляем пиксели сверху-снизу и слева-справа
    pad_width = [(padded_height, padded_height), (padded_width, padded_width)]
    if image.ndim == 3:
        pad_width.append((0, 0))
    # дублируем края тем же цветом
    padded = np.pad(image, pad_width, mode='edge')
    # нарезаем картинку на множество мелких окон размером с наш фильтр
    windows = sliding_window_view(padded, kernel_shape)
    return np.clip(np.tensordot(windows, kernel), 0, 255)


# генерация ядра Гаусса
def get_gaussian_kernel(kernel_size: int, sigma: float) -> NDArray[float32]:
    # создаем одномерную сетку координат
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # вычисляем одномерное распределение
    gauss = np.exp(-((ax / sigma) ** 2) / 2)
    kernel = np.outer(gauss, gauss)
    return (kernel / kernel.sum()).astype(float32)


# 3 сглаживание (применение оператора Гаусса)
def smooth_manual(image: NDArray[uint8], kernel_size: int) -> NDArray[uint8]:
    kernel = get_gaussian_kernel(kernel_size, kernel_size / 6)
    return convolve_manual(image, kernel).astype(uint8)


# 4 выделение границ (применение оператора Собеля)
def detect_edges_manual(image: NDArray[uint8]) -> NDArray[uint8]:
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float32)
    gx = convolve_manual(image, kernel_x)
    gy = convolve_manual(image, kernel_y)
    # вычисляем гипотенузу
    gipot = np.sqrt(gx**2 + gy**2)
    # снова корректируем значения
    return np.clip(gipot, 0, 255).astype(uint8)


# 5 гамма-коррекция изображения
def gamma_correction_manual(image: NDArray[uint8], gamma: float = 2.2) -> NDArray[uint8]:
    inv_gamma = 1.0 / gamma
    # 1/y < 1 - светлее
    table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(uint8)
    return table[image]


# поиск файлов
images = [f for f in os.listdir(PAINTINGS_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
if not images:
    exit()

image = cv2.imread(os.path.join(PAINTINGS_DIR, images[0]))
gray_cv_base = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# замеры времени и выполнение операций

# grayscale

start = time.time()
gray_manual = grayscale_manual(image)
print(f"полутоновый ручной: {time.time() - start:.4f} сек")

start = time.time()
gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"полутоновый opencv: {time.time() - start:.4f} сек")


# свертка

kernel = get_gaussian_kernel(5, 1)

start = time.time()
conv_manual = convolve_manual(gray_cv_base, kernel).astype(uint8)
print(f"свертка ручная: {time.time() - start:.4f} сек")

start = time.time()
conv_cv = cv2.filter2D(gray_cv_base, -1, kernel)
print(f"свертка opencv: {time.time() - start:.4f} сек")


# gaussian

start = time.time()
gaussian_manual = smooth_manual(gray_cv_base, 5)
print(f"гаусс ручной: {time.time() - start:.4f} сек")

start = time.time()
gaussian_cv = cv2.GaussianBlur(gray_cv_base, (5, 5), 0)
print(f"гаусс opencv: {time.time() - start:.4f} сек")


# sobel

start = time.time()
sobel_manual = detect_edges_manual(gray_cv_base)
print(f"собель ручной: {time.time() - start:.4f} сек")

start = time.time()
gx_cv = cv2.Sobel(gray_cv_base, cv2.CV_64F, 1, 0, ksize=3)
gy_cv = cv2.Sobel(gray_cv_base, cv2.CV_64F, 0, 1, ksize=3)
sobel_cv = cv2.magnitude(gx_cv, gy_cv)
sobel_cv = np.clip(sobel_cv, 0, 255).astype(uint8)
print(f"собель opencv: {time.time() - start:.4f} сек")


# gamma

start = time.time()
gamma_manual = gamma_correction_manual(image, 2.2)
print(f"гамма ручной: {time.time() - start:.4f} сек")

start = time.time()
inv_gamma = 1.0 / 2.2
table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(uint8)
gamma_cv = cv2.LUT(image, table)
print(f"гамма opencv: {time.time() - start:.4f} сек")

# сохранение
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gray_manual.jpg"), gray_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gray_cv.jpg"), gray_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "conv_manual.jpg"), conv_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "conv_cv.jpg"), conv_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "gaussian_manual.jpg"), gaussian_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gaussian_cv.jpg"), gaussian_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "sobel_manual.jpg"), sobel_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "sobel_cv.jpg"), sobel_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "gamma_manual.jpg"), gamma_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gamma_cv.jpg"), gamma_cv)

print("обработка завершена")
