import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from os import getpid
from typing import AsyncGenerator

import aiofiles
import aiohttp
import cv2
import numpy as np

from .artwork import ArtworkMetadata, ColorArtwork


@dataclass(slots=True)
class ImageProcessorPaths:
    download_dir: str
    output_dir: str
    metadata_url: str = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"


@dataclass(slots=True)
class ProcessedImage:
    image_number: int
    metadata: ArtworkMetadata
    results: dict[str, np.ndarray]


def process_image(image_bytes: bytes, metadata: ArtworkMetadata, image_number: int) -> ProcessedImage:
    logging.info(f"Cвертка для изображения {image_number} началась (PID {getpid()})")

    array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    artwork = ColorArtwork(image, metadata)

    results = {
        "grayscale_manual": artwork.to_grayscale(False).image,
        "grayscale_opencv": artwork.to_grayscale(True).image,
        "smooth_manual": artwork.smooth(5, False).image,
        "smooth_opencv": artwork.smooth(5, True).image,
        "edges_manual": artwork.detect_edges(False).image,
        "edges_opencv": artwork.detect_edges(True).image,
        "gamma_manual": artwork.gamma_correction(2.2, False).image,
        "gamma_opencv": artwork.gamma_correction(2.2, True).image,
    }

    logging.info(f"Cвертка для изображения {image_number} завершена (PID {getpid()})")
    return ProcessedImage(image_number=image_number, metadata=metadata, results=results)


class AsyncImagePipeline:
    def __init__(self, paths: ImageProcessorPaths) -> None:
        self.paths = paths
        os.makedirs(self.paths.download_dir, exist_ok=True)
        os.makedirs(self.paths.output_dir, exist_ok=True)
        self.pool = ProcessPoolExecutor()

    async def fetch_metadata(self, session: aiohttp.ClientSession, object_id: str) -> ArtworkMetadata:
        async with session.get(f"{self.paths.metadata_url}{object_id}") as response:
            if response.status != 200:
                logging.warning(f"Object ID {object_id} не найден (HTTP {response.status})")
                return None
            data = await response.json()

            if "objectID" not in data:
                logging.warning(f"Object ID {object_id}: в ответе нет object_id")
                return None

            return ArtworkMetadata(
                objectID=str(data["objectID"]),
                title=data.get("title", "Unknown"),
                primaryImage=data.get("primaryImage", "")
            )

    async def download_stage(
        self, numbered_ids: list[tuple[int, str]]
    ) -> AsyncGenerator[tuple[int, ArtworkMetadata, bytes], None]:
        async with aiohttp.ClientSession() as session:
            async def download_single(image_number: int, object_id: str):
                logging.info(f"Загрузка изображения {image_number} начата")
                metadata = await self.fetch_metadata(session, object_id)
                if metadata is None:
                    return None

                async with session.get(metadata.primaryImage) as response:
                    image_bytes = await response.read()

                fname = f"{image_number}_{metadata.objectID}_original.jpg"
                original_path = os.path.join(self.paths.download_dir, fname)

                async with aiofiles.open(original_path, "wb") as file:
                    await file.write(image_bytes)

                logging.info(f"Загрузка изображения {image_number} завершена")
                return (image_number, metadata, image_bytes)

            tasks = [asyncio.create_task(download_single(num, oid)) for num, oid in numbered_ids]
            for task in asyncio.as_completed(tasks):
                yield await task

    async def processing_stage(
        self, source: AsyncGenerator[tuple[int, ArtworkMetadata, bytes], None]
    ) -> AsyncGenerator[ProcessedImage, None]:
        loop = asyncio.get_running_loop()
        tasks = []

        async for image_number, metadata, image_bytes in source:
            task = loop.run_in_executor(self.pool, process_image, image_bytes, metadata, image_number)
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            yield await task

    async def save_stage(self, source: AsyncGenerator[ProcessedImage, None]) -> None:
        async for processed in source:
            image_number = processed.image_number
            metadata = processed.metadata
            save_tasks = []

            for name, image in processed.results.items():
                fname = f"{image_number}_{metadata.objectID}_{name}.jpg"
                filename = os.path.join(self.paths.output_dir, fname)
                logging.info(f"Сохранение изображения {image_number}: {name}")
                save_tasks.append(self.save_image(image, filename))

            await asyncio.gather(*save_tasks)
            logging.info(f"Сохранение изображения {image_number} завершено")

    async def save_image(self, image: np.ndarray, filename: str) -> None:
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Ошибка кодирования")

        async with aiofiles.open(filename, "wb") as file:
            await file.write(buffer.tobytes())

    async def run(self, numbered_ids: list[tuple[int, str]]) -> None:
        download_gen = self.download_stage(numbered_ids)
        processing_gen = self.processing_stage(download_gen)
        await self.save_stage(processing_gen)
