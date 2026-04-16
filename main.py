import logging
import sys
import os
from scripts.image_processor import ImageProcessor, ImageProcessorPaths

# Добавление пути к скриптам для корректного импорта модулей
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    # Настройка путей с разделением по папкам для оригиналов и результатов
    paths = ImageProcessorPaths(
        download_dir="paintings/stocks/",
        output_dir="paintings/processed/"
    )
    processor = ImageProcessor(paths)
    try:
        # Загрузка случайного объекта
        artwork = processor._download_random_image()
        print(f"\nКартина: {artwork.metadata.title}")

        # Grayscale
        gray_manual = artwork.to_grayscale(use_opencv=False)
        processor.save_artwork(gray_manual, suffix="_grayscale_manual")

        gray_cv = artwork.to_grayscale(use_opencv=True)
        processor.save_artwork(gray_cv, suffix="_grayscale_opencv")

        # Smoothing
        smooth_manual = artwork.smooth(kernel_size=5, use_opencv=False)
        processor.save_artwork(smooth_manual, suffix="_smooth_manual")

        smooth_cv = artwork.smooth(kernel_size=5, use_opencv=True)
        processor.save_artwork(smooth_cv, suffix="_smooth_opencv")

        # Edge Detection
        edges_manual = artwork.detect_edges(use_opencv=False)
        processor.save_artwork(edges_manual, suffix="_edges_manual")

        edges_cv = artwork.detect_edges(use_opencv=True)
        processor.save_artwork(edges_cv, suffix="_edges_opencv")

        # Gamma Correction
        gamma_manual = artwork.gamma_correction(gamma=2.2, use_opencv=False)
        processor.save_artwork(gamma_manual, suffix="_gamma_manual")

        gamma_cv = artwork.gamma_correction(gamma=2.2, use_opencv=True)
        processor.save_artwork(gamma_cv, suffix="_gamma_opencv")

        print("\nОбработка завершена. Результаты в paintings/processed/")

    except Exception as e:
        logging.error(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
