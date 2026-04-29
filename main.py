import logging
import sys
import os
from scripts.image_processor import ImageProcessor, ImageProcessorPaths

sys.path.append(os.path.join(os.getcwd(), 'scripts'))
logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    paths = ImageProcessorPaths(
        download_dir="paintings/stocks/",
        output_dir="paintings/processed/"
    )
    processor = ImageProcessor(paths)
    try:
        artwork = processor.download_random_image()
        print(f"\nКартина: {artwork.metadata.title}")
        # серый
        gray_manual = artwork.to_grayscale(use_opencv=False)
        processor.save_artwork(gray_manual, suffix="_grayscale_manual")
        gray_cv = artwork.to_grayscale(use_opencv=True)
        processor.save_artwork(gray_cv, suffix="_grayscale_opencv")
        # сглаживание
        smooth_manual = artwork.smooth(kernel_size=5, use_opencv=False)
        processor.save_artwork(smooth_manual, suffix="_smooth_manual")
        smooth_cv = artwork.smooth(kernel_size=5, use_opencv=True)
        processor.save_artwork(smooth_cv, suffix="_smooth_opencv")
        # края
        edges_manual = artwork.detect_edges(use_opencv=False)
        processor.save_artwork(edges_manual, suffix="_edges_manual")
        edges_cv = artwork.detect_edges(use_opencv=True)
        processor.save_artwork(edges_cv, suffix="_edges_opencv")
        # гамма
        gamma_manual = artwork.gamma_correction(gamma=2.2, use_opencv=False)
        processor.save_artwork(gamma_manual, suffix="_gamma_manual")
        gamma_cv = artwork.gamma_correction(gamma=2.2, use_opencv=True)
        processor.save_artwork(gamma_cv, suffix="_gamma_opencv")

        # чб + края
        new1_1 = artwork.to_grayscale(use_opencv=True)
        processor.save_artwork(new1_1, suffix="_NEW_11_gray")
        new1_2 = artwork.detect_edges(use_opencv=True).to_grayscale()
        processor.save_artwork(new1_2, suffix="_NEW_22__ed-gray")
        res1 = new1_1 + new1_2
        processor.save_artwork(res1, suffix="_res1_BW+EDGES")

        print("\nЗавершено")

    except Exception as e:
        logging.error(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
