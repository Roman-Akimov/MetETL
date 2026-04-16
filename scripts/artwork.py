from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override
import cv2
import numpy as np
from numpy import float32, float64, uint8
from numpy.typing import NDArray

from .decorators import timer


@dataclass(frozen=True, slots=True)
class ArtworkMetadata:
    """Неизменяемый контейнер для метаданных иллюстрации"""
    objectID: str
    title: str = "Неизвестно"
    primaryImage: str = "Неизвестно"


class Artwork(ABC):
    """Абстрактный базовый класс для иллюстрации, управляющий данными изображения и метаданными"""

    __slots__: tuple[str, ...] = ("_image", "_metadata")

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        self._image: NDArray[uint8] = image
        self._metadata: ArtworkMetadata = metadata

    @property
    def image(self) -> NDArray[uint8]:
        """Возвращает копию массива изображения для предотвращения изменений"""
        return self._image.copy()

    @property
    def metadata(self) -> ArtworkMetadata:
        """Доступ к неизменяемым метаданным произведения"""
        return self._metadata

    @abstractmethod
    def to_grayscale(self, use_opencv: bool = False) -> "Artwork":
        """Абстрактный метод преобразования изображения в оттенки серого"""
        pass

    def _sliding_window(self, padded_img: NDArray[float32], kh: int, kw: int):
        """
        Генератор скользящего окна
        Позволяет итерироваться по окнам, не создавая гигантских массивов в памяти.
        """
        h, w = padded_img.shape[:2]
        out_h = h - kh + 1
        out_w = w - kw + 1

        for i in range(out_h):
            for j in range(out_w):
                # Возвращает окно (slice) для текущей позиции
                if padded_img.ndim == 3:
                    yield i, j, padded_img[i:i + kh, j:j + kw, :]
                else:
                    yield i, j, padded_img[i:i + kh, j:j + kw]

    def _convolve_float(self, kernel: NDArray[float32], use_opencv: bool = False) -> NDArray[float32]:
        """Свертка изображения с использованием скользящего окна"""
        if use_opencv:
            return cv2.filter2D(self._image, cv2.CV_32F, kernel).astype(float32)

        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Настройка паддинга
        if self._image.ndim == 3:
            pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
            h, w, c = self._image.shape
            out = np.zeros((h, w, c), dtype=np.float32)
        else:
            pad_width = ((pad_h, pad_h), (pad_w, pad_w))
            h, w = self._image.shape
            out = np.zeros((h, w), dtype=np.float32)

        padded = np.pad(self._image.astype(float32), pad_width, mode="reflect")

        # Основной цикл свертки через sliding_window
        for i, j, window in self._sliding_window(padded, kh, kw):
            if window.ndim == 3:
                # Умножаем ядро на каждый канал окна отдельно (broadcasting)
                # kernel[:, :, None] превращает (3,3) в (3,3,1) для корректного умножения на (3,3,3)
                out[i, j] = np.sum(window * kernel[:, :, None], axis=(0, 1))
            else:
                out[i, j] = np.sum(window * kernel)

        return out

    def _convolve(self, kernel: NDArray[float32], use_opencv: bool = False) -> "Artwork":
        """Применяет свертку и возвращает новый объект правильного класса"""
        float_result = self._convolve_float(kernel, use_opencv)
        result = np.clip(float_result, 0, 255).astype(uint8)
        return self.__class__(result, self.metadata)

    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> NDArray[float32]:
        """Создает 2D-ядро Гаусса для сглаживания"""
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-((ax / sigma) ** 2) / 2).astype(float64)
        kernel = np.outer(gauss, gauss)

        return (kernel / np.sum(kernel)).astype(float32)

    @timer
    def smooth(self, kernel_size: int, use_opencv: bool = False) -> "Artwork":
        """Применяет размытие по Гауссу для уменьшения шума"""
        kernel = self._get_gaussian_kernel(kernel_size, kernel_size / 6)
        convolved_image = self._convolve(kernel, use_opencv)._image
        return self.__class__(convolved_image, self.metadata)

    @timer
    def detect_edges(self, use_opencv: bool = False) -> "Artwork":
        """Выделяет границы с использованием оператора Собеля"""
        if use_opencv:
            # OpenCV Собель
            gx = cv2.Sobel(self._image, cv2.CV_64F, dx=1, dy=0, ksize=3)
            gy = cv2.Sobel(self._image, cv2.CV_64F, dx=0, dy=1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
        else:
            # Обычный Собель
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float32)
            kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float32)

            gx = self._convolve_float(kernel_x)
            gy = self._convolve_float(kernel_y)
            magnitude = np.sqrt(gx**2 + gy**2)

        return self.__class__(np.clip(magnitude, 0, 255).astype(uint8), self.metadata)

    @timer
    def gamma_correction(self, gamma: float, use_opencv: bool = False) -> "Artwork":
        """Корректирует яркость изображения с помощью гамма-коррекции"""
        inv_gamma = 1.0 / gamma
        table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(uint8)

        if use_opencv:
            corrected = cv2.LUT(self._image, table)
        else:
            corrected = table[self._image]

        return self.__class__(corrected.astype(uint8), self.metadata)

    @override
    def __str__(self) -> str:
        """Возвращает строковое представление метаданных произведения"""
        return (
            f"Object ID: {self._metadata.objectID}\n"
            f"Title: {self._metadata.title}\n"
            f"Primary Image: {self._metadata.primaryImage}\n"
        )

    @timer
    def __add__(self, other: "Artwork") -> "Artwork":
        """Смешивает два произведения путем сложения значений их пикселей"""
        # 1. определяем целевой (максимальный) размер
        h1, w1 = self._image.shape[:2]
        h2, w2 = other._image.shape[:2]

        target_h = max(h1, h2)
        target_w = max(w1, w2)
        target_size = (target_w, target_h)
        # openCV использует формат (ширина, высота)

        # 2. подгоняем изображения под целевой размер, если они не совпадают
        img_self_raw = self._image
        if (h1, w1) != (target_h, target_w):
            img_self_raw = cv2.resize(self._image, target_size, interpolation=cv2.INTER_LINEAR)

        img_other_raw = other._image
        if (h2, w2) != (target_h, target_w):
            img_other_raw = cv2.resize(other._image, target_size, interpolation=cv2.INTER_LINEAR)

        # 3. приведение к float32 для точности вычислений
        img_self = img_self_raw.astype(float32)
        img_other = img_other_raw.astype(float32)

        # 4. совместимость каналов (Цвет + Ч/Б)
        if img_self.ndim == 2 and img_other.ndim == 3:
            img_self = np.stack((img_self,) * 3, axis=-1)
        elif img_self.ndim == 3 and img_other.ndim == 2:
            img_other = np.stack((img_other,) * 3, axis=-1)

        # 5. сложение с насыщением
        combined_image = cv2.add(img_self, img_other)
        # обрезаем значения (clip)
        result_image = np.clip(combined_image, 0, 255).astype(uint8)

        if result_image.ndim == 2:
            return GrayscaleArtwork(result_image, self.metadata)
        return ColorArtwork(result_image, self.metadata)


class GrayscaleArtwork(Artwork):
    """Подкласс Artwork специально для одноканальных изображений в оттенках серого"""

    __slots__ = ()

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        if image.ndim != 2:
            raise ValueError("Для работы в оттенках серого нужен двумерный массив")

        super().__init__(image, metadata)

    def to_grayscale(self, use_opencv: bool = False) -> "GrayscaleArtwork":
        """Преобразует цветное изображение в оттенки серого, используя веса яркости"""
        return GrayscaleArtwork(self.image, self.metadata)


class ColorArtwork(Artwork):
    """ Подкласс Artwork для трехканальных цветных изображений (обычно в формате BGR)"""

    __slots__ = ()

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        """ Преобразует цветное изображение в оттенки серого, используя веса яркости"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Для ColorArtwork нужны 3 параметра!")

        super().__init__(image, metadata)

    def to_grayscale(self, use_opencv: bool = False) -> "GrayscaleArtwork":
        """ Преобразует цветное изображение в оттенки серого, используя веса яркости """
        if use_opencv:
            gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY).astype(uint8)
        else:
            # Формула яркости: 0.299R + 0.587G + 0.114B
            weights = np.array([0.114, 0.587, 0.299], dtype=float32)
            gray_image = np.clip(self._image @ weights, 0, 255).astype(uint8)

        return GrayscaleArtwork(gray_image, self.metadata)
