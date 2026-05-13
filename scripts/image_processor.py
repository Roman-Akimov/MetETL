import csv
import logging
import random
from time import perf_counter

from .async_pipeline import AsyncImagePipeline, ImageProcessorPaths


class ImageProcessor:
    """Главный управляющий класс приложения"""

    def __init__(self, paths: ImageProcessorPaths) -> None:
        self.pipeline = AsyncImagePipeline(paths)

    def get_random_painting_ids(self, count: int) -> list[str]:
        """Получение случайных ID картин из CSV файла"""
        painting_ids = []

        with open("data/MetObjects.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("Classification") == "Paintings" and row.get("Is Public Domain") == "True":
                    object_id = row.get("Object ID")
                    if object_id:
                        painting_ids.append(object_id)

        if len(painting_ids) < count:
            raise ValueError("Недостаточно картин для выборки")

        return random.sample(painting_ids, count)

    def build_numbered_ids(self, count: int) -> list[tuple[int, str]]:
        """Подготовка пронумерованного списка ID"""
        ids = self.get_random_painting_ids(count)
        numbered_ids = list(enumerate(ids, start=1))

        logging.info("Список изображений подготовлен")
        for number, object_id in numbered_ids:
            logging.info(f"Изображение {number}: ID {object_id}")

        return numbered_ids

    async def process_images_async(self, count: int) -> None:
        """Асинхронная обработка изображений"""
        logging.info("Асинхронная версия")
        numbered_ids = self.build_numbered_ids(count)

        start = perf_counter()
        await self.pipeline.run(numbered_ids)
        total_time = perf_counter() - start

        logging.info("Асинхронное выполнение завершено")
        logging.info(f"Время асинхронного выполнения: {total_time:.2f} сек")

    async def process_images_sequential(self, count: int) -> None:
        """Последовательная обработка изображений"""
        logging.info("Последовательная версия")
        numbered_ids = self.build_numbered_ids(count)

        start = perf_counter()
        for item in numbered_ids:
            await self.pipeline.run([item])
        total_time = perf_counter() - start

        logging.info("Последовательное выполнение завершено")
        logging.info(f"Время последовательного выполнения: {total_time:.2f} сек")

    async def compare_versions(self, count: int) -> None:
        """Сравнение времени работы версий"""
        logging.info("Начало сравнения")

        sequential_start = perf_counter()
        await self.process_images_sequential(count)
        sequential_time = perf_counter() - sequential_start

        async_start = perf_counter()
        await self.process_images_async(count)
        async_time = perf_counter() - async_start

        logging.info("Результаты")
        logging.info(f"Последовательно: {sequential_time:.2f} сек")
        logging.info(f"Асинхронно: {async_time:.2f} сек")

        if async_time > 0:
            speedup = sequential_time / async_time
            logging.info(f"Ускорение: {speedup:.2f}x")
