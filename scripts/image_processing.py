import cv2
import numpy as np
import os
import time

PAINTINGS_DIR = "paintings"


# приведение цветного изображения к полутоновому (ручная реализация)
def rgb_to_gray_manual(image):
    height, width, _ = image.shape
    gray = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            gray[i, j] = int(0.299*r + 0.587*g + 0.114*b)

    return gray


# свёртка с использованием двумерной маски (ручная реализация)
def convolution_manual(image, kernel):
    height, width = image.shape
    k = kernel.shape[0] // 2
    padded = cv2.copyMakeBorder(image, k, k, k, k, cv2.BORDER_REFLECT)
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            region = padded[i:i+2*k+1, j:j+2*k+1]
            value = np.sum(region * kernel)
            output[i, j] = np.clip(value, 0, 255)

    return output


# ядро Гаусса для сглаживания
def gaussian_kernel():
    return np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16


# ядро Собеля по оси X (выделение вертикальных границ)
def sobel_kernel_x():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])


# ядро Собеля по оси Y (выделение горизонтальных границ)
def sobel_kernel_y():
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])


# гамма-коррекция изображения (ручная реализация)
def gamma_correction(image, gamma):
    normalized = image / 255.0

    corrected = np.power(normalized, gamma)

    corrected = np.uint8(corrected * 255)
    return corrected


# выравнивание гистограммы изображения (ручная реализация)
def histogram_equalization(image):
    hist = np.zeros(256)
    for value in image.flatten():
        hist[value] += 1
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]

    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = cdf_normalized[image[i, j]]

    return result.astype(np.uint8)


# пороговая обработка (ручная реализация)
def threshold_manual(image, thresh=127):
    h, w = image.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # если яркость выше порога - белый, иначе - черный
            binary[i, j] = 255 if image[i, j] > thresh else 0
    return binary


# поиск изображения в директории
images = [f for f in os.listdir(PAINTINGS_DIR) if f.endswith(".jpg")]

if not images:
    print("Нет изображений для обработки")
    exit()

# открываем первое найденное изображение
image_path = os.path.join(PAINTINGS_DIR, images[0])
# преобразование в numpy-массив через imread
image = cv2.imread(image_path)

print("Обрабатываем:", image_path)

# приведение цветного изображения к полутоновому ----------
start = time.time()
gray_manual = rgb_to_gray_manual(image)
end = time.time()
print("Ручная реализация (grayscale):", end - start)

# библиотечная функция cv2.cvtColor()
start = time.time()
gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
end = time.time()
print("Библиотечная реализация (grayscale):", end - start)

# сглаживание (применение оператора Гаусса) ----------
kernel = gaussian_kernel()

start = time.time()
gaussian_manual = convolution_manual(gray_manual, kernel)
end = time.time()
print("Ручная реализация (гаусс):", end - start)

# библиотечная функция cv2.filter2D()
start = time.time()
gaussian_cv = cv2.filter2D(gray_cv, -1, kernel)
end = time.time()
print("Библиотечная реализация (гаусс):", end - start)

# выделение границ (оператор Собеля) ----------
kx = sobel_kernel_x()
ky = sobel_kernel_y()

# ручная реализация оператора Собеля ----------
start = time.time()
sobel_x = convolution_manual(gray_manual, kx)
sobel_y = convolution_manual(gray_manual, ky)
# модуль градиента: sqrt(Gx^2 + Gy^2)
sobel_manual = np.sqrt(sobel_x.astype(np.float32)**2 + sobel_y.astype(np.float32)**2)
sobel_manual = np.clip(sobel_manual, 0, 255).astype(np.uint8)
end = time.time()
print("Ручная реализация (Собель):", end - start)

# библиотечная реализация оператора Собеля через cv2.Sobel() ----------
start = time.time()
sobelx = cv2.Sobel(gray_cv, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_cv, cv2.CV_64F, 0, 1, ksize=3)
sobel_cv = cv2.magnitude(sobelx, sobely)
sobel_cv = np.clip(sobel_cv, 0, 255).astype(np.uint8)
end = time.time()
print("Библиотечная реализация (Собель):", end - start)

# выделение углов (cv2.cornerHarris) ----------
start = time.time()
gray_float = np.float32(gray_cv)
harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
# нормализация для визуализации
harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
end = time.time()
print("Библиотечная реализация (Harris):", end - start)

# гамма-коррекция изображения ----------
start = time.time()
gamma_img = gamma_correction(gray_cv, 2.2)
end = time.time()
print("Ручная реализация (гамма-коррекция):", end - start)

# выравнивание гистограммы изображения ----------
start = time.time()
hist_manual = histogram_equalization(gray_cv)
end = time.time()
print("Ручная реализация (выравнивание гистограммы):", end - start)

# библиотечная функция cv2.equalizeHist()
start = time.time()
hist_cv = cv2.equalizeHist(gray_cv)
end = time.time()
print("Библиотечная реализация (выравнивание гистограммы):", end - start)

# выравнивание гистограммы цветного изображения ----------
# перевод RGB изображения в LAB, выравнивание L-канала, перевод обратно в RGB
start = time.time()
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_eq = cv2.equalizeHist(l)
lab_eq = cv2.merge((l_eq, a, b))
rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
end = time.time()
print("Выравнивание гистограммы цветного изображения (LAB):", end - start)

# пороговая обработка ----------
start = time.time()
binary_manual = threshold_manual(gray_cv, 127)
end = time.time()
print("Ручная реализация (пороговая обработка):", end - start)

# библиотечная функция cv2.threshold() ----------
start = time.time()
_, binary_cv = cv2.threshold(gray_cv, 127, 255, cv2.THRESH_BINARY)
end = time.time()
print("Библиотечная реализация (пороговая обработка):", end - start)

# сохранение в ту же директорию, что и оригинальное ----------
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gray_manual.jpg"), gray_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gray_cv.jpg"), gray_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "gaussian_manual.jpg"), gaussian_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "gaussian_cv.jpg"), gaussian_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "sobel_manual.jpg"), sobel_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "sobel_cv.jpg"), sobel_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "harris.jpg"), harris_norm)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "gamma.jpg"), gamma_img)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "hist_manual.jpg"), hist_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "hist_cv.jpg"), hist_cv)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "lab_equalized.jpg"), rgb_eq)

cv2.imwrite(os.path.join(PAINTINGS_DIR, "binary_manual.jpg"), binary_manual)
cv2.imwrite(os.path.join(PAINTINGS_DIR, "binary_cv.jpg"), binary_cv)
