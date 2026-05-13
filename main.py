import asyncio
import logging
import sys

from scripts.async_pipeline import ImageProcessorPaths
from scripts.image_processor import ImageProcessor

logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main() -> None:
    image_count = 3

    if len(sys.argv) > 1:
        try:
            image_count = int(sys.argv[1])
            if image_count <= 0:
                raise ValueError
        except ValueError:
            logging.error("Количество изображений должно быть > 0")
            return

    paths = ImageProcessorPaths(
        download_dir="paintings/originals/",
        output_dir="paintings/process/"
    )
    processor = ImageProcessor(paths)
    logging.info(f"Количество изображений: {image_count}\n")
    try:
        await processor.compare_versions(image_count)
    except Exception as error:
        logging.error(f"\nОшибка: {error}")

    logging.info("\nЗавершили.")

if __name__ == "__main__":
    asyncio.run(main())
