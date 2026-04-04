from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final, override
import cv2
import numpy as np
from numpy import float32, float64, uint8
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from .decorators import timer


@dataclass(frozen=True, slots=True)
class ArtworkMetadata:
    """Неизменяемый контейнер для метаданных иллюстрации.

    Аргументы:
        ObjectId (str): Уникальный идентификатор объекта.
        title (str): Название иллюстрации. По умолчанию используется значение "Неизвестно".
        primaryImage (str): URL-адрес основного изображения. По умолчанию используется значение "Неизвестно".
    """

    objectID: str
    title: str = "Неизвестно"
    primaryImage: str = "Неизвестно"


class Artwork(ABC):
    """Абстрактный базовый класс для иллюстрации, управляющий данными изображения и метаданными.

    Аргументы:
        image (NDArray[uint8]): Данные изображения в виде числового массива.
        метаданные (ArtworkMetadata): метаданные, связанные с иллюстрацией.
    """

    __slots__: tuple[str, ...] = ("_image", "_metadata")

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        self._image: NDArray[uint8] = image
        self._metadata: ArtworkMetadata = metadata

    @property
    def image(self) -> NDArray[uint8]:
        """Возвращает копию массива изображения для предотвращения изменений.

        Возвращается:
            NDArray[uint8]: Копия данных изображения.
        """
        return self._image.copy()

    @property
    def metadata(self) -> ArtworkMetadata:
        """Доступ к неизменяемым метаданным произведения.

        Возвращается:
            ArtworkMetadata: Объект метаданных.
        """
        return self._metadata

    @abstractmethod
    def to_grayscale(self, use_opencv: bool = False) -> "Artwork":
        """Абстрактный метод преобразования изображения в оттенки серого.

        Аргументы:
            use_opencv (bool, необязательно): Реализация OpenCV. По умолчанию используется значение False.

        Возвращается:
            Иллюстрация: Новый экземпляр иллюстрации в оттенках серого.
        """
        pass

    def _convolve_float(self, kernel: NDArray[float32], use_opencv: bool = False) -> NDArray[float32]:
        """Выполняется свертка, возвращающая результаты float32 для точности.

        Аргументы:
            ядро (NDArray[float32]): Ядро свертки.
            use_opencv (bool, необязательно): cv2.filter2D, если значение равно True. используется значение False.

        Возвращается:
            NDArray[float32]: Массив необработанных свернутых изображений.
        """
        if use_opencv:
            return cv2.filter2D(self._image, cv2.CV_32F, kernel).astype(float32)

        kernel_shape: tuple[int, int] = kernel.shape
        kh, kw = kernel_shape

        pad_h, pad_w = kh // 2, kw // 2
        pad_width = [(pad_h, pad_h), (pad_w, pad_w)]
        if self._image.ndim == 3:
            pad_width.append((0, 0))

        padded = np.pad(self._image, pad_width, mode="reflect")
        axes = np.array((0, 1))
        windows = sliding_window_view(padded, kernel_shape, axes)

        return np.tensordot(windows, kernel).astype(float32)

    def _convolve(self, kernel: NDArray[float32], use_opencv: bool = False) -> "Artwork":
        """Применяет свертку и ограничивает результаты допустимым диапазоном uint8 (0-255).

        Аргументы:
            kernel (NDArray[float32]): Ядро свертки.
            use_opencv (bool, необязательно): Использовать OpenCV для фильтрации. По умолчанию - False.

        Возвращается:
            Artwork: Новый экземпляр произведения с обработанными пикселями.
        """
        result = np.clip(self._convolve_float(kernel, use_opencv), 0, 255).astype(uint8)
        return self.__class__(result, self.metadata)

    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> NDArray[float32]:
        """Создает 2D-ядро Гаусса для сглаживания.

        Аргументы:
            kernel_size (int): Размер квадратного ядра.
            sigma (float): Стандартное отклонение распределения Гаусса.

        Возвращается:
            NDArray[float32]: Нормализованное ядро Гаусса.
        """
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-((ax / sigma) ** 2) / 2).astype(float64)
        kernel = np.outer(gauss, gauss)

        return (kernel / np.sum(kernel)).astype(float32)

    @timer
    def smooth(self, kernel_size: int, use_opencv: bool = False) -> "Artwork":
        """Применяет размытие по Гауссу для уменьшения шума.

        Аргументы:
            kernel_size (int): Размер ядра размытия.
            use_opencv (bool, необязательно): Использовать реализацию OpenCV. По умолчанию — False.

        Возвращается:
            Artwork: Сглаженное произведение искусства.
        """
        kernel = self._get_gaussian_kernel(kernel_size, kernel_size / 6)
        convolved_image = self._convolve(kernel, use_opencv)._image
        return self.__class__(convolved_image, self.metadata)

    @timer
    def detect_edges(self, use_opencv: bool = False) -> "Artwork":
        """Выделяет границы с использованием оператора Собеля.

        Аргументы:
            use_opencv (bool, необязательно): Использовать cv2.Sobel. По умолчанию — False.

        Возвращается:
            Artwork: Произведение с выделенными границами.
        """
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
        """Корректирует яркость изображения с помощью гамма-коррекции.

        Аргументы:
            gamma (float): Значение гаммы (>1 осветляет, <1 затемняет).
            use_opencv (bool, необязательно): Использовать таблицу поиска (LUT). По умолчанию — False.

        Возвращается:
            Artwork: Произведение с примененной гамма-коррекцией.
        """
        inv_gamma = 1.0 / gamma
        table = ((np.arange(256) / 255.0) ** inv_gamma * 255).astype(uint8)

        if use_opencv:
            corrected = cv2.LUT(self._image, table)
        else:
            corrected = table[self._image]

        return self.__class__(corrected.astype(uint8), self.metadata)

    @override
    def __str__(self) -> str:
        """Возвращает строковое представление метаданных произведения.

        Возвращается:
            str: Отформатированная строка с ID, названием и URL изображения.
        """
        return (
            f"Object ID: {self._metadata.objectID}\n"
            f"Title: {self._metadata.title}\n"
            f"Primary Image: {self._metadata.primaryImage}\n"
        )

    @timer
    def __add__(self, other: "Artwork") -> "Artwork":
        """Смешивает два произведения путем сложения значений их пикселей.
        Аргументы:
            other (Artwork): Другой экземпляр Artwork тех же размеров.
        Исключения:
            ValueError: Если размеры изображений не совпадают.
        Возвращается:
            Artwork: Смешанное произведение.
        """
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


@final
class GrayscaleArtwork(Artwork):
    """Подкласс Artwork специально для одноканальных изображений в оттенках серого.

    Аргументы:
        image (NDArray[uint8]): 2D-массив пикселей NumPy.
        metadata (ArtworkMetadata): Связанные метаданные произведения.
    """

    __slots__ = ()

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        if image.ndim != 2:
            raise ValueError("Для работы в оттенках серого нужен двумерный массив")

        super().__init__(image, metadata)

    @override
    def to_grayscale(self, use_opencv: bool = False) -> "GrayscaleArtwork":
        """Преобразует цветное изображение в оттенки серого, используя веса яркости.

        Аргументы:
            use_opencv (bool, необязательно): Использовать метод cvtColor из OpenCV. По умолчанию — False.

        Возвращается:
            GrayscaleArtwork: Новый экземпляр, содержащий версию изображения в оттенках серого.
        """
        return GrayscaleArtwork(self.image, self.metadata)


@final
class ColorArtwork(Artwork):
    """
    Подкласс Artwork для трехканальных цветных изображений (обычно в формате BGR).

    Аргументы:
        image (NDArray[uint8]): 3D-массив NumPy (Высота, Ширина, 3).
        metadata (ArtworkMetadata): Связанные метаданные произведения.
    """

    __slots__ = ()

    def __init__(self, image: NDArray[uint8], metadata: ArtworkMetadata) -> None:
        """
        Преобразует цветное изображение в оттенки серого, используя веса яркости.

        Аргументы:
            use_opencv (bool, необязательно): Использовать метод cvtColor из OpenCV. По умолчанию — False.

        Возвращается:
            GrayscaleArtwork: Новый экземпляр, содержащий версию изображения в оттенках серого.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Для ColorArtwork нужны 3 параметра!")

        super().__init__(image, metadata)

    @override
    def to_grayscale(self, use_opencv: bool = False) -> "GrayscaleArtwork":
        """Преобразует цветное изображение в оттенки серого, используя веса яркости.

        Аргументы:
            use_opencv (bool, необязательно): Использовать метод cvtColor из OpenCV
            По умолчанию — False.
        Возвращается:
            GrayscaleArtwork: Новый экземпляр, содержащий версию изображения
            в оттенках серого.
        """
        if use_opencv:
            gray_image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY).astype(uint8)
        else:
            # Формула яркости: 0.299R + 0.587G + 0.114B
            weights = np.array([0.114, 0.587, 0.299], dtype=float32)
            gray_image = np.clip(self._image @ weights, 0, 255).astype(uint8)

        return GrayscaleArtwork(gray_image, self.metadata)
