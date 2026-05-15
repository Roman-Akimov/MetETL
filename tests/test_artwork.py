import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from akimovcode.images.artwork import ColorArtwork, ArtworkMetadata, GrayscaleArtwork


class TestArtwork(unittest.TestCase):
    def setUp(self):
        """Подготовка данных перед каждым тестом"""
        self.meta = ArtworkMetadata(objectID="1", title="Test", primaryImage="http://test.com/1.jpg")
        # Создаем маленькое тестовое изображение 4x4 пикселя
        self.img = np.zeros((4, 4, 3), dtype=np.uint8)
        self.img[0, 0] = [255, 255, 255]
        self.artwork = ColorArtwork(self.img, self.meta)

    def test_metadata_integrity(self):
        """Проверка сохранности метаданных"""
        self.assertEqual(self.artwork.metadata.objectID, "1")
        self.assertEqual(self.artwork.metadata.title, "Test")

    def test_to_grayscale(self):
        """Проверка конвертации в оттенки серого"""
        gray_art = self.artwork.to_grayscale(use_opencv=False)
        # Проверяем, что размерность изменилась (стала 2D)
        self.assertEqual(gray_art.image.ndim, 2)
        # Проверяем, что вернулся правильный класс
        self.assertIsInstance(gray_art, GrayscaleArtwork)

    def test_gamma_correction_validity(self):
        """Проверка, что гамма-коррекция не меняет размер изображения"""
        corrected = self.artwork.gamma_correction(2.2)
        self.assertEqual(corrected.image.shape, self.img.shape)


class TestImageProcessorMock(unittest.TestCase):
    @patch('akimovcode.images.image_processor.requests.get')
    def test_api_call_mocking(self, mock_get):
        """Пример использования Mock для имитации ответа API"""
        # поддельный ответ сервера
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"objectID": 123, "title": "Mock Title"}
        mock_get.return_value = mock_response
        self.assertEqual(mock_response.json()["objectID"], 123)


if __name__ == "__main__":
    unittest.main()
