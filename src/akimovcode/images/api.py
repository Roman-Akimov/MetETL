import csv
import random
import requests
import os
import json
import logging
from akimovcode.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

OBJECT_FILE = 'data/MetObjects.csv'
REQUEST = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
PAINTINGS_DIR = 'paintings'

# 1
object_id = []
try:
    with open(OBJECT_FILE, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get('Classification') == 'Paintings':
                if row.get('Object ID'):
                    object_id.append(row['Object ID'])

    logger.info(f'Найдено {len(object_id)} картин в файле {OBJECT_FILE}')

except FileNotFoundError:
    logger.error(f'Файл {OBJECT_FILE} не найден')
    exit()

except Exception as e:
    logger.error(f'Ошибка при чтении файла: {e}')
    exit()

if not object_id:
    logger.error('Не найдено ни одного объекта с классификацией "Paintings"')
    exit()

random_object_id = random.choice(object_id)
logger.info(f'Выбран случайный объект ID: {random_object_id}')

# 2
try:
    resp = requests.get(f'{REQUEST}{random_object_id}')
    resp.raise_for_status()
    object_data = resp.json()
    image_url = object_data.get('primaryImage')
    logger.info(f'Получены данные для объекта {random_object_id}')
except requests.exceptions.RequestException as e:
    logger.error(f'Ошибка при запросе: {e}')
    exit()

# 3 скачивание
os.makedirs(PAINTINGS_DIR, exist_ok=True)
image_path = os.path.join(PAINTINGS_DIR, f'{random_object_id}.jpg')
json_path = os.path.join(PAINTINGS_DIR, f'{random_object_id}.json')

if image_url:
    try:
        image_get = requests.get(image_url)
        image_get.raise_for_status()
        with open(image_path, mode='wb') as image:
            image.write(image_get.content)
        logger.info(f'Изображение сохранено: {image_path}')
    except requests.exceptions.RequestException as e:
        logger.error(f'Ошибка при скачивании изображения: {e}')
        exit()
else:
    logger.warning(f'Нет URL изображения для объекта {random_object_id}')

try:
    with open(json_path, mode='w', encoding='utf-8') as meta:
        json.dump(object_data, meta, indent=4, ensure_ascii=False)
    logger.info(f'Метаданные сохранены: {json_path}')
except Exception as e:
    logger.error(f'Ошибка при сохранении метаданных: {e}')
    exit()

logger.info('Операция завершена успешно')
