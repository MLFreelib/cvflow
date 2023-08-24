# cvflow

![Tests](https://github.com/MLFreelib/cvflow/workflows/Tests/badge.svg)

# Введение

CVFlow - набор алгоритмов на основе компьютерного зрения, предоставляющий,
как запуск готовых примеров, так и настройку собственных.

## Установка

1) Для установки всех необходимых библиотек требуется python, чтобы его установить, скачайте и
проследуйте одной из инструкций 
   1) [Oфициальный сайт python](https://www.python.org/downloads/)
   2) [Официальный сайт Anaconda](https://www.anaconda.com/download)

2) Если в качестве установки была выбрана Anaconda, то проследуйте следующей инструкции для создания среды для работы
[Создание среды](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

3) После установки python/Anaconda откройте консоль и введите команду(В случае Anaconda необходимо вводить в созданной 
выше среде):

       pip install -r requirements.txt

4) Для использования алгоритмов на видеокарте введите команду указанную на сайте в соответствии вашим
требованиям [Официальный сайт PyTorch](https://pytorch.org/get-started/locally/)
    

## Список алгоритмов

<details open>
<summary><b>Определение положения объектов в пространстве и расстояния между ними</b></summary>

### Об алгоритме

    Данный алгоритм на основе двух снимков одной сцены с разных ракурсов определяет расстояние между выделенными 
    пользователем объектами. Особенностью реализации в библиотеке является то, что алгоритм направлен на работу с 
    небольшими объектами. В библиотеке представлено несколько реализаций на основе двух моделей MobileStereoNet и 
    CDN-GaNet.

### Результаты

1) **Расстояние от камер до объектов**


- MobileStereoNet

![MobileStereoNet distance from cameras to objects](readmedata/MobileStereoNetObjectDistance.png)

- CDN-GaNET

![MobileStereoNet distance from cameras to objects](readmedata/CDN-GaNETToObjectDistance.png)

- Полученные измерения

| №   | Высота объекта  | Расстояние до объекта, Метры | Реальное расстояние до объекта, Метры | Ошибка, Миллиметры |
|-----|-----------------|------------------------------|---------------------------------------|--------------------|
| 1   | MobileStereoNet | 0.63                         | 0.6                                   | 30                 |
| 2   | MobileStereoNet | 0.64                         | 0.6                                   | 40                 |
| 3   | MobileStereoNet | 0.68                         | 0.7                                   | 20                 |
| 4   | MobileStereoNet | 0.35                         | 0.25                                  | 100                |
| 5   | CDN-GaNET       | 0.62                         | 0.6                                   | 20                 |
| 6   | CDN-GaNET       | 0.56                         | 0.6                                   | 40                 |
| 7   | CDN-GaNET       | 0.75                         | 0.7                                   | 50                 |
| 8   | CDN-GaNET       | 0.4                          | 0.25                                  | 150                |

---

2) **Расстояние между объектами**

- Пример
  ![Distance between objects](readmedata/DistanceBetweenObjects.jpg)


- Полученные измерения

| №   | Расстояние до объектов, Метры | Предсказанное расстояние между объектами, Миллиметры | Реальное расстояние между объекта, Миллиметры | Ошибка, Миллиметры |
|-----|-------------------------------|------------------------------------------------------|-----------------------------------------------|--------------------|
| 1   | 0.7                           | 113                                                  | 101                                           | 12                 |
| 2   | 0.7                           | 93                                                   | 101                                           | 7                  |
| 3   | 0.6                           | 105                                                  | 100                                           | 5                  |
| 4   | 0.6                           | 94                                                   | 100                                           | 6                  |
| 5   | 0.6                           | 95                                                   | 100                                           | 5                  |

---

3) **Размер объектов**

- Пример

![ObjectsSize](readmedata/ObjectsSize.png)

- Полученные измерения

| №   | Расстояние до объектов, Метры | Предсказанный размер объекта, Миллиметры | Реальный размер объекта, Миллиметры | Ошибка, Миллиметры |
|-----|-------------------------------|------------------------------------------|-------------------------------------|--------------------|
| 1   | 0.3                           | 153                                      | 150                                 | 3                  |
| 2   | 0.35                          | 149                                      | 150                                 | 1                  |
| 3   | 0.25                          | 148                                      | 150                                 | 2                  |
| 4   | 0.3                           | 104                                      | 100                                 | 4                  |
| 5   | 0.35                          | 102                                      | 100                                 | 2                  |
| 6   | 0.25                          | 97                                       | 100                                 | 3                  |

---

### Запуск алгоритма
  Запустите файл  stereo_tracking_example/run.py для демонстрации примера. В качестве аргументов необходимо указать путь к файлам источников данных через запятую (не менее двух). Можно использовать данные из папки tests/test_data. 

  Пример команды для запуска:
```python run.py --videofile top_l.mov,top_r.mov```

</details>

<details open>
<summary><b>Поиск дефектов и прочих образований на материале</b></summary>

### Об алгоритме

    За основу алгоритма взята архитектура SSD300, в которой последний слой классификации был удален,
    а перед слоями по предсказанию сдвигов добавлен слой для для векторного представления каждой
    ограничивающей рамки и на основе функции ошибки TripletLoss создается векторное пространство, которое 
    позволяет без обучения обнаруживать дефекты на различных материалах.

Примеры детекции

![flaw_wood](readmedata/flaw_wood.jpg)
![flaw_steel](readmedata/flaw_steel.jpg)

</details>


<details open>
<summary><b>Распознавание номера автомобилей</b></summary>

### Об алгоритме

    Алгормитм по распозаванию номеров состоит из двух моделей: YOLOv8n для детекции номеров и
    RCNN для распознавания текстовых последовательностей.

### Результаты

1) Детекция, YOLOv8n

- Пример детекции

![Plate detection](readmedata/PlateDetection.png)

2) Распознавание, RCNN

- Пример распознавания

![Plate detection](readmedata/plates_recogn.jpg)

- Измерения (автомобили)

| Задача                | Train | Test | Valid |
|-----------------------|-------|------|-------|
| Детекция номеров      | 0.99     | 0.98    | 0.98  | 
| Цельный алгоритм  (детекция + распознавание номеров    | 0.99     | 0.96    | 0.96  |

- Измерения (поезда)
  
| Задача                | Train | Test | Valid |
|-----------------------|-------|------|-------|
| Детекция номеров      | 0.99    | 0.93    | 0.95  | 
| Цельный алгоритм      | 0.98     | 0.89    | 0.90  |

### Запуск алгоритма

</details>


<details open>
<summary><b>Распозавание номера вагона поезда</b></summary>

### Об алгоритме

    Разработанный алгоритм обучен на реальных данных в различных условиях окружающей среды и предназначен

для обнаружения и распознавания номеров на вагонах поездов. Для алгоритма использовались архитектура из
распознавания номеров машин.

### Результаты

- Пример детекции

![Plate detection](readmedata/TrainCarsPlateDetection.png)

### Запуск алгоритма

</details>


<details open>
<summary><b>Распознавание QR и штрихкодов</b></summary>

### Об алгоритме

    На основе OpenCV и zbar построен алгоритм по детекции QR-кодов и их расшифровке.

### Результат

Пример работы алгоритма

![qrcode_example](readmedata/qrcode.jpg)

### Запуск алгоритма

</details>


<details open>
<summary><b>Классификация трранспортных средств</b></summary>

### Об алгоритме

    Разработанный алгоритм позволяет классифицировать транспортные средства по типу кузова и, таким образом, определять
    их размеры. Это может быть необходимо при анализе поставок на заводах, фильтрации т/с, которые могут угрожать 
    безопасности персонала или при проектировке контрольно-пропускных пунктов, однако использование этого модуля не 
    ограничивается приведёнными ситуациями. Процесс работы модуля продемонстрирован на рисунке 8.1. Точность классификации 
    при IoU 0.5 составила 0,962. 

Пример определения размеров кузова

![car_class1](readmedata/car_classification.jpg)
![car_class2](readmedata/car_classification2.jpg)


### Запуск алгоритма

</details>


<details open>
<summary><b>Гранулометрия</b></summary>

### Об алгоритме

    Гранулометрический анализ – распределение камней руды по крупности, характеризующееся процентным выходом от массы 
    или количества кусков руды. Алгоритм построен на основе архитектуры UNet, что позволяет достичь
    высокой точности в определении размеров камней на основе масок.

Пример работы алгоритма и сравнение с классическим методом водораздела

![unet](readmedata/gran_unet.jpg)

Измерения


|                |          |         | Точность | Полнота |
|----------------|----------|---------|----------|---------|
| <b>Вычисленное | <b>Камни | <b>Фон  |          |         |
| <b>Камни       | TP = 947 | FP = 56 | 94,4%    | 94,9%   |
| <b>Фон         | FN = 23  | TN = 0  |          |         |


Пример распределения камней на конвейере

![distribution](readmedata/distrib.jpg)

### Запуск алгоритма

1.Скачайте веса и подготовьте источники данных для определения размеров.Поместите веса рядом с файлом примера.
2. Запустите файл примера run.py, указав в качестве параметров источники данных.
Пример команды запуска c видео: 
```python run.py --videofile video.mp4```

</details>

<details open>
<summary><b>Определения скорости объектов по видеопотоку</b></summary>

### Об алгоритме

    Модуль использует несколько различных алгоритмов компьютерного зрения для подсчёта скорости объекта: нейронную 
    сеть yolov5, состоящую из 367 слоёв и содержащую свыше 46 миллионов параметров, трекер объектов SORT для 
    отслеживания обнаруженных объектов и сравнения положений их центроидов, а также алгоритм расчёта скорости, 
    использующий данные о положениях центроидов, заранее известные размеры объектов (ширина или длина кузова),
    также частоту кадров, размер кадра и свободный параметр, необходимый для более точной калибровки системы. 

Пример работы:

![speed_algo](readmedata/speed.jpg)

Измерения


|                                      | Измерения |
|--------------------------------------|-----------|
| <b>Средняя погрешность               | < 5 км/ч  |
| <b>Скорость работы детектора         | 0.037 c   |
| <b> Полная скорость работы алгоритма | 0.07 c    | 
Измерения проводились на RTX 3090.

### Запуск алгоритма

</details>

## Веса

    Веса находятся по сссылке в папке в соответствие названию алгоритма: 
<b>[ссылка на веса](https://statanly.com/info/weight)
    
