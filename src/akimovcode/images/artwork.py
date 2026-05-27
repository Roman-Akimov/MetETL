from abc import ABC, abstractmethod
from dataclasses import dataclass
import cv2
import numpy as np
from numpy import float32, float64, uint8
from numpy.typing import NDArray
import logging
from akimovcode.logging_config import setup_logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArtworkMetadata:
    """контейнер для метаданных картинок"""
    objectID: str
    title: str = "Неизвестно"
    primaryImage: str = "Неизвестно"


class Artwork(ABC):
    """Абстрактный базовый класс для картинок, управляющий данными изображения и метаданными"""
    __slots__: tuple[str, ...] = ("_image", "_metadata")

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        self._image: NDArray[uint8] = image
        self._metadata: ArtworkMetadata = metadata

    @property
    def image(self) -> NDArray[uint8]:
        return self._image.copy()

    @property
    def metadata(self) -> ArtworkMetadata:
        return self._metadata

    @abstractmethod
    def to_grayscale(self, use_opencv: bool = False) -> "Artwork":
        pass

    def _convolve_float(self, kernel: NDArray[float32], use_opencv: bool = False) -> NDArray[float32]:
        """Универсальная быстрая свертка через векторные сдвиги"""
        if use_opencv:
            return cv2.filter2D(self._image, cv2.CV_32F, kernel).astype(float32)

        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        h, w = self._image.shape[:2]

        if self._image.ndim == 3:
            pad_cfg = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
        else:
            pad_cfg = ((pad_h, pad_h), (pad_w, pad_w))
        pad_img = np.pad(self._image.astype(float32), pad_cfg, mode='constant')
        conv_img = np.zeros_like(self._image, dtype=np.float32)
        for i in range(k_h):
            for j in range(k_w):
                weight = kernel[i, j]
                if weight == 0:
                    continue
                if self._image.ndim == 3:
                    conv_img += pad_img[i:i + h, j:j + w, :] * weight
                else:
                    conv_img += pad_img[i:i + h, j:j + w] * weight
        return conv_img

    def _convolve(self, kernel: NDArray[float32], use_opencv: bool = False) -> "Artwork":
        """Применяет свертку и возвращает новый объект"""
        float_result = self._convolve_float(kernel, use_opencv)
        result = np.clip(float_result, 0, 255).astype(uint8)
        return self.__class__(result, self.metadata)

    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> NDArray[float32]:
        """2d ядро Гаусса для сглаживания"""
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-((ax / sigma) ** 2) / 2).astype(float64)
        kernel = np.outer(gauss, gauss)
        return (kernel / np.sum(kernel)).astype(float32)

    def smooth(self, kernel_size: int, use_opencv: bool = False) -> "Artwork":
        """Применяем размытие по Гауссу для уменьшения шума"""
        kernel = self._get_gaussian_kernel(kernel_size, kernel_size / 6)
        convolved_image = self._convolve(kernel, use_opencv)._image
        return self.__class__(convolved_image, self.metadata)

    def detect_edges(self, use_opencv: bool = False) -> "Artwork":
        if use_opencv:
            # OpenCV Собель
            gx = cv2.Sobel(self._image, cv2.CV_64F, dx=1, dy=0, ksize=3)
            gy = cv2.Sobel(self._image, cv2.CV_64F, dx=0, dy=1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
        else:
            # Обычный
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float32)
            kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float32)
            gx = self._convolve_float(kernel_x)
            gy = self._convolve_float(kernel_y)
            magnitude = np.sqrt(gx**2 + gy**2)

        return self.__class__(np.clip(magnitude, 0, 255).astype(uint8), self.metadata)

    def gamma_correction(self, gamma: float, use_opencv: bool = False) -> "Artwork":
        inv_gamma = 1.0 / gamma
        table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(uint8)
        if use_opencv:
            corrected = cv2.LUT(self._image, table)
        else:
            corrected = table[self._image]
        return self.__class__(corrected.astype(uint8), self.metadata)

    def __str__(self) -> str:
        """Возвращает строковое представление метаданных произведения"""
        return (
            f"id: {self._metadata.objectID}\n"
            f"имя: {self._metadata.title}\n"
            f"начальный вид: {self._metadata.primaryImage}\n"
        )

    def __add__(self, other: "Artwork") -> "Artwork":
        """Смешивает два произведения путем сложения значений их пикселей"""
        try:
            max_height = max(self._image.shape[0], other._image.shape[0])
            max_width = max(self._image.shape[1], other._image.shape[1])

            c_self = self._image.shape[2] if self._image.ndim == 3 else 1
            c_other = other._image.shape[2] if other._image.ndim == 3 else 1

            if c_self == 3 and c_other == 1:
                raise TypeError("ошибка: невозможно смешать цветное и черно-белое изображение")
            if c_self == 1 and c_other == 3:
                raise TypeError("ошибка: невозможно смешать черно-белое и цветное изображение")

            # итоговое кол-во каналов
            final_channels = max(c_self, c_other)
            if final_channels == 1:
                canvas = np.zeros((max_height, max_width), dtype=np.float32)
            else:
                canvas = np.zeros((max_height, max_width, final_channels), dtype=np.float32)

            # преобразуем изображения в формат float32 для точных вычислений
            image_self = self._image.astype(np.float32)
            image_other = other._image.astype(np.float32)

            # размещаем изображения
            if canvas.ndim == 2:
                # для чб
                canvas[:self._image.shape[0], :self._image.shape[1]] += image_self
                canvas[:other._image.shape[0], :other._image.shape[1]] += image_other
            else:
                # для цветных
                canvas[:self._image.shape[0], :self._image.shape[1], :] += image_self
                canvas[:other._image.shape[0], :other._image.shape[1], :] += image_other

            result_image = np.clip(canvas, 0, 255).astype(uint8)
            if result_image.ndim == 2:
                return GrayscaleArtwork(result_image, self.metadata)
            return ColorArtwork(result_image, self.metadata)

        except TypeError as error:
            logger.error(f"Ошибка при смешивании изображений: {error}")
            return self.__class__(self._image.copy(), self.metadata)


class GrayscaleArtwork(Artwork):
    __slots__ = ()

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        if image.ndim != 2:
            raise ValueError("Для работы в оттенках серого нужен двумерный массив")

        super().__init__(image, metadata)

    def to_grayscale(self, use_opencv: bool = False) -> "GrayscaleArtwork":
        """Преобразует цветное изображение в оттенки серого, используя веса яркости"""
        return GrayscaleArtwork(self.image, self.metadata)


class ColorArtwork(Artwork):
    __slots__ = ()

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Для ColorArtwork нужны 3 параметра!")

        super().__init__(image, metadata)

    def to_grayscale(self, use_opencv: bool = False) -> "GrayscaleArtwork":
        if use_opencv:
            gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY).astype(uint8)
        else:
            # Формула яркости: 0.299R + 0.587G + 0.114B
            weights = np.array([0.114, 0.587, 0.299], dtype=float32)
            gray_image = np.clip(self._image @ weights, 0, 255).astype(uint8)

        return GrayscaleArtwork(gray_image, self.metadata)


# Настройка логирования при прямом запуске файла
if __name__ == "__main__":
    setup_logging()
    logger.info("Модуль artwork.py загружен")
