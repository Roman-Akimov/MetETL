
import csv
import random
import requests
import os
import json


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


except FileNotFoundError:
    print(f'Файл {OBJECT_FILE} не найден')


except Exception as e:
    print('Ошибка при чтении файла', {e})


random_object_id = random.choice(object_id)

# 2

try:
    resp = requests.get(f'{REQUEST}{random_object_id}')
    resp.raise_for_status()
    object_data = resp.json()
    image_url = object_data.get('primaryImage')



except requests.exceptions.RequestException:
    print('Ошибка при запроcе')
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
    
    except requests.exceptions.RequestException:
        print('Ошибка при скачивании изображения')
        exit()


with open(json_path, mode='w') as meta:
    json.dump(object_data, meta, indent=4)
