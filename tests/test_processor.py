import unittest
from pathlib import Path
from akimovcode.images.async_pipeline import ImageProcessorPaths


class TestProcessorConfig(unittest.TestCase):
    def test_paths_creation(self):
        """Проверяем, что пути инициализируются корректно"""
        paths = ImageProcessorPaths(download_dir="test_in", output_dir="test_out")
        self.assertEqual(paths.download_dir, "test_in")
        self.assertEqual(paths.output_dir, "test_out")

    def test_directory_logic(self):
        """Проверка логики Path """
        path = Path("images") / "test"
        self.assertEqual(str(path).replace("\\", "/"), "images/test")


if __name__ == "__main__":
    unittest.main()
