import logging
import os
from dataclasses import asdict, dataclass
from json import dump
import random
import csv

import cv2
import numpy as np
import requests

from .artwork import Artwork, ArtworkMetadata, ColorArtwork, GrayscaleArtwork
from .decorators import timer


@dataclass(slots=True)
class ImageProcessorPaths:
    """Конфигурация путей к файлам и конечных точек API."""

    download_dir: str = "paintings/stocks/"
    output_dir: str = "paintings/processed/"
    metadata_url: str = (
        "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
    )


class ImageProcessor:
    """Организует загрузку, обработку и хранение произведений искусства."""

    __slots__ = "_paths"

    def __init__(self, paths: ImageProcessorPaths | dict[str, str]) -> None:
        """Инициализирует конструктор и проверяет наличие выходных директорий"""
        if isinstance(paths, dict):
            paths = ImageProcessorPaths(**paths)
        self._paths = paths
        os.makedirs(self._paths.download_dir, exist_ok=True)
        os.makedirs(self._paths.output_dir, exist_ok=True)

    @timer
    def _get_random_painting_id(self) -> str:
        """Выбирает случайный ID картины из CSV-файла коллекции."""
        painting_ids = []

        with open("data/MetObjects.csv", mode="r", encoding="utf-8") as file:
            # Используем DictReader, чтобы обращаться к столбцам по именам
            reader = csv.DictReader(file)
            for row in reader:
                # Фильтруем: только картины (Paintings) и только те, что в открытом доступе (Is Public Domain)
                if row.get('Classification') == 'Paintings' and row.get('Is Public Domain') == 'True':
                    obj_id = row.get('Object ID')
                    if obj_id:
                        painting_ids.append(obj_id)

        if not painting_ids:
            raise ValueError("В CSV не найдено подходящих картин для скачивания.")
        return random.choice(painting_ids)

    @timer
    def _fetch_painting_metadata(self, object_id: str) -> ArtworkMetadata:
        """Получает описательные данные для конкретной работы через API Метрополитен"""

        logging.info(f"Получение метаданных для ID объекта: {object_id}")
        response: requests.Response = requests.get(
            f"{self._paths.metadata_url}{object_id}", timeout=10
        )
        response.raise_for_status()
        data: dict[str, str] = response.json()

        return ArtworkMetadata(
            objectID=data["objectID"],
            title=data.get("title", "Неизвестно"),
            primaryImage=data.get("primaryImage", "Неизвестно"),
        )

    @timer
    def _save_painting(self, metadata: ArtworkMetadata) -> str:
        """Скачивает файл изображения и сохраняет соответствующие метаданные на диск"""

        painting_id = metadata.objectID

        # Сохранение метаданных в JSON
        file_path = os.path.join(self._paths.download_dir, f"{painting_id}.json")
        with open(file_path, "w", encoding="utf-8") as json_file:
            dump(asdict(metadata), json_file, indent=4, ensure_ascii=False)

        # Скачивание и сохранение изображения
        response = requests.get(metadata.primaryImage, timeout=10)
        response.raise_for_status()

        file_path = os.path.join(self._paths.download_dir, f"{painting_id}.jpg")
        with open(file_path, "wb") as img_file:
            _ = img_file.write(response.content)
        return file_path

    @timer
    def _download_random_image(self) -> Artwork:
        """Высокоуровневый метод для выбора, скачивания и загрузки случайной картины"""

        logging.info("Начало загрузки случайного изображения...")
        random_painting_id = self._get_random_painting_id()
        logging.info(f"Успешно выбран ID случайной картины: {random_painting_id}")

        logging.info("Получение метаданных картины...")
        metadata = self._fetch_painting_metadata(random_painting_id)
        logging.info("Метаданные успешно получены")

        logging.info("Сохранение данных и изображения картины...")
        _ = self._save_painting(metadata)
        logging.info("Картинку скачали")

        return self.get_artwork_by_id(metadata.objectID)

    @timer
    def run_pipeline(
        self,
        operations: tuple[str, ...],
        opencv_uses: tuple[bool, ...] | bool,
        painting_id: str = "",
    ) -> Artwork:
        """Выполняет последовательность операций обработки изображения над картиной"""

        if not isinstance(opencv_uses, bool):
            if len(operations) != len(opencv_uses):
                raise ValueError(
                    "Количество флагов opencv_uses должно совпадать с количеством операций"
                )
        else:
            opencv_uses = (opencv_uses,) * len(operations)

        if not painting_id:
            logging.info("ID картины не предоставлен, скачивание случайного изображения...")
            artwork = self._download_random_image()
        else:
            logging.info(f"Загрузка изображения с ID: {painting_id}")
            artwork = self.get_artwork_by_id(painting_id)

        logging.info("Изображение успешно загружено")

        for op, use_opencv in zip(operations, opencv_uses):
            logging.info(f"Применение операции: {op}")
            if op == "smooth":
                artwork = artwork.smooth(5, use_opencv=use_opencv)
            elif op == "detect_edges":
                artwork = artwork.detect_edges(use_opencv=use_opencv)
            elif op == "gamma_correction":
                artwork = artwork.gamma_correction(2.2, use_opencv=use_opencv)
            elif op == "grayscale":
                artwork = artwork.to_grayscale(use_opencv=use_opencv)
            else:
                raise ValueError(f"Неподдерживаемая операция - {op}")

        file_suffix = "_".join(operations) + ("_opencv" if any(opencv_uses) else "_manual")

        # Save the processed image
        output_path = os.path.join(
            self._paths.output_dir, f"{artwork.metadata.objectID}_{file_suffix}"
        )

        with open(f"{output_path}.json", 'w') as file:
            dump(asdict(artwork.metadata), file, indent=4, ensure_ascii=False)

        _ = cv2.imwrite(f"{output_path}.jpg", artwork.image)
        logging.info(f"Обработанное изображение сохранено в {output_path}")
        logging.info("Конвейер обработки завершен")

        return artwork

    @timer
    def get_artwork_by_id(self, painting_id: str = "") -> Artwork:
        """Загружает изображение с диска и создает экземпляр нужного подкласса Artwork"""

        if not painting_id:
            painting_id = self._get_random_painting_id

        image = cv2.imread(
            f"{self._paths.download_dir}{painting_id}.jpg", cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError(f"Не удалось прочесть ID: {painting_id}")

        metadata = self._fetch_painting_metadata(painting_id)
        if image.ndim == 2:
            return GrayscaleArtwork(image.astype(np.uint8), metadata)
        return ColorArtwork(image.astype(np.uint8), metadata)

    @timer
    def save_artwork(self, artwork: Artwork, suffix: str = "_processed") -> None:
        """Сохраняет объект Artwork"""
        metadata = artwork.metadata
        # Формируем имя: ID_операция_тип.jpg
        file_name = f"{metadata.objectID}{suffix}.jpg"
        file_path = os.path.join(self._paths.output_dir, file_name)

        cv2.imwrite(file_path, artwork.image)
        logging.info(f"Сохранено: {file_path}")
