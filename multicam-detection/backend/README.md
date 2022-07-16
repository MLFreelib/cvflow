## Backend

Backend сервис предназначен для обработки запросов, приходящих с [frontend](../frontend).
Запуск сервиса происходит после запуска [frontend](../frontend)
, в противном случае произойдет ошибка.

### Сборка сервиса
Для сборки и запуска сервиса используются [docker](https://www.docker.com/) и [docker-compose](https://docs.docker.com/compose/).
Убедитесь, что данные приложения установлены на вашем устройстве.

Чтобы собрать сервис, из [текущей дериктории](/), запустите команду слудующего вида:
```shell
docker-compose build --build-arg DEVICE=... --build-arg CUDA_VERSION=...
```
Для корректной сборки требуется указать аргументы сборки.
#### DEVICE
Указывает тип девайса, который будет использоваться для работы.
Можно указать либо `cpu`, либо `cuda`.
<br>
Девайс `cuda` указывается только в том случае, если таковой имеется на машине, 
где производится сборка сервиса.

#### CUDA_VERSION
Указывает версию девайса `cuda`.
<br>
Данный аргумент указывается только в том случае, если аргумент сборки `DEVICE` был установлен на `cuda`.
В противном случае этот аргумент не имеет никакого эффекта.
<br>
В текущей версии проекта корректная работа достигается **только** в случае указания версии из
[списка](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), доступной для
`torch 1.10`.
<br>
Если версия Вашей `cuda` больше, чем те, что присутствуют в 
[списке](https://detectron2.readthedocs.io/en/latest/tutorials/install.html),
укажите версию ниже, из тех, что присутствуют в нем.

#### Примеры
Если для работы будет спользована `cpu`, соберите проект при помощи команды:
```shell
docker-compose build --build-arg DEVICE=cpu
```

Если для рабоыт будет использована `cuda`, например, версии `11.1` или выше,
собрать проект можно при помощи команды:
```shell
docker-compose build --build-arg DEVICE=cuda --build-arg CUDA_VERSION=11.1
```

Сборку сервиса требуется производить только при первом запуску проекта. 
При последующих запусках пропскайте данный пункт.

### Запуск сервиса
После завершения сборки, или при последующих запусках, для старта сервиса запустите команду следующего вида:
```shell
docker-compose run [--gpus=all] backend [args ...]
```
Для корректного запуска серсиса, требуется указать аргументы запуска.

#### --host или -H
Задает хост, на котором запущен [frontend](../frontend).
<br>
Если оба сервиса запускаются на одной машине, достаточно в этом аргументе указать `localhost`.

Обязытельный: Да


#### --port или -P
Задает порт, на котором запущен [frontend](../frontend).
<br>
В общем случае это значение поля `FRONTEND_PORT` из [.env](../frontend/.env)

Обязательный: Да

#### --secure или -S
Если этот флаг установлен, то сервис будет общаться с [frontend](../frontend)
по защищенному соединению (https и wss), если таковая возможность активирована на [frontend](../frontend).

Обязательный: Нет

#### --api
Указывает версию api сервиса [frontend](../frontend).
<br>
На текущий момент существует только версия `1`, поэтому стоит оставлять этот аргумент без изменения.

Обязательный: Нет
<br>
Значение по умолчанию: 1

#### --stride_detection
Указывает промежуток в количестве фреймов, через которые каждый раз будет производиться детекция
(см. `--detector`)

Обязательный: Нет
<br>
Значение по умолчанию: 10

#### --stride_send
Указывает промежуток в количестве фреймов, через которые каждый раз будут отсылаться данные на
[frontend](../frontend).

Обязательный: Нет
<br>
Значение по умолчанию: 60

#### --devices
Указывает список девайсов, на которых будут производиться вычисления.
<br>
Значения указываются в виде `cpu` или `cuda:x`, где `x` - номер девайса с `cuda`.
<br>
Возможно указать несколько девайсов через пробел. 
В таком случае вся работа будет распределена между ними.

Обязательный: Нет
<br>
Значение по умолчанию: cpu

#### --detector
Указывает детектор, который будет использоваться для обнаружения объектов во время
работы програмы.
<br>
Доступны следующие варианты:
```
yolov5n
yolov5s
yolov5m
yolov5l
yolov5x
COCO-Detection/fast_rcnn_R_50_FPN_1x
COCO-Detection/faster_rcnn_R_101_C4_3x
COCO-Detection/faster_rcnn_R_101_DC5_3x
COCO-Detection/faster_rcnn_R_101_FPN_3x
COCO-Detection/faster_rcnn_R_50_C4_1x
COCO-Detection/faster_rcnn_R_50_C4_3x
COCO-Detection/faster_rcnn_R_50_DC5_1x
COCO-Detection/faster_rcnn_R_50_DC5_3x
COCO-Detection/faster_rcnn_R_50_FPN_1x
COCO-Detection/faster_rcnn_R_50_FPN_3x
COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x
COCO-Detection/fcos_R_50_FPN_1x
COCO-Detection/retinanet_R_101_FPN_3x
COCO-Detection/retinanet_R_50_FPN_1x
COCO-Detection/retinanet_R_50_FPN_3x
COCO-Detection/rpn_R_50_C4_1x
COCO-Detection/rpn_R_50_FPN_1x
COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x
COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x
COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x
COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x
COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x
COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x
COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x
COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x
COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_giou
COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x
COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x
```
Для большей информации о детекторах, смотри 
[YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/)
и
[Detectron2](https://github.com/facebookresearch/detectron2/tree/main/configs).

Обязательный: Да

#### --detector_weights
Задает путь до кастомных весов, которые будут использоваться в детектове (см. `--detector`).
<br>
Если Вы собираетесь использовать стандартные веса предобученной модели,
не задавайте данный аргумент.

Обязателбный: Нет

#### --detector_threshold
Задает порог, при котором детектор будет считать детекцию правильной.

Обязательный: Нет
<br>
Значение по умолчанию: 0.65

#### --bbox_expander или --expander
Включает модель BBoxExpander.
<br>
Если данный аргумент не задан, то данная модель не будет использоваться при работе сервиса.

Обязателный: Нет

#### --bbox_expander_weights или --expander_weights
Задает путь до кастомных весов, которые будут использоваться в моделе BBoxExpander.
<br>
Если Вы собираетесь использовать стандартные веса предобученной модели,
не задавайте данный аргумент

Обязательный: Нет
<br>
Значение по умолчанию: data/bbox_expander/weight/model_final.pth

#### --fps
Задает количество кадров в секунду видеофрагментов, которые будет отправляться 
на [frontend](../frontend).

Обязательный: Нет
<br>
Значение по умолчанию: 30

#### --person_radius
Задает радиус точки проекции, которая будет отображаться как проекции человека
на [frontend](../frontend).

Обязательный: Нет
<br>
Значение по умолчанию: 0.2

#### --person_mean_height
Задает среднюю высоту человека.

Обязательный: Нет
<br>
Значение по умолчанию: 1.65

#### --person_mean_velocity
Задает средне значение скорости человека в используюмом расстоянии в секунду.

Обязательный: Нет
<br>
Значение по умолчанию: 0.7

#### --car_radius
Задает радиус точки проекции, которая будет отображаться как проекции машины
на [frontend](../frontend).

Обязательный: Нет
<br>
Значение по умолчанию: 0.75

#### --car_mean_height
Задает среднюю высоту машины.

Обязательный: Нет
<br>
Значение по умолчанию: 2.15

#### --car_mean_velocity
Задает среднее значение скорости машины в используемом расстоянии в секунду.

Обязательный: Нет
<br>
Значение по умолчанию: 1.8

#### --decay_point_time
Задает промежуток времени в секундах, через который старые неиспользующиеся точки проекций забываются.

Обязательный: Нет
<br>
Значение по умолчанию: 2

#### --download
Указывает путь загрузки видеофайлов, которая будет использоваться для их загрузки
при работе с таковыми.

Обязательный: Нет
<br>
Значение по умолчанию: data/video

#### --processes
Задает количество процессов, которые будут обрабатывать параллельно поток видео.
<br>
Для каждого девайса, указанного в `--devices` будет создано количество процессов,
указанное в данном аргуменете.

Обязательный: Нет
<br>
Значение по умолчанию: 1

#### --logger_type или --logger
Указывает тип выводимых сообщений логгера.
<br>
Можно указать следующие значения
```
INFO
DEBUG
ERROR
```

Обязательный: Нет
<br>
Значение по умолчанию: INFO

#### Дополнительно
Если Вы собираетесь использовать девайсы `cuda` (см. `--devices`), для корректной работы требуется
указать параметр `--gpus=all` (см. Примеры)

#### Примеры
```shell
docker-compose run backend --host localhost --port 4000 --detector COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x
```

```shell
docker-compose run --gpus=all backend --host localhost --port 4000 --detector yolov5x --devices cuda:0 cuda:1
```

### Команды backend
[frontend](../frontend) общается с данным сервисам при помощи команд по websocker соединению.
Здесь представлен список команд, которе готов обрабатывать [backend](/)
<br>
Команды приходят в формате `json`. Тип команды задан в поле `command`.

#### START_SESSION

```shell
{
  "command": "START_SESSION",
  "sessionId": ... // ID сессии
}
```
Получает нотификацию от [frontend](../frontend) о том, что сессия с ID `sessionId` была запущена.

#### APPEND_VIDEO

```shell
{
  "command": "APPEND_VIDEO",
  "videoId": ..., // ID видео
  "sessionId": ..., // ID сессии
  "uri": ..., // Пусть до видео
  "streamingType": ... // Тип стриминга
}
```
Добавляет видео к списку обрабатываемых.
<br>
Команда дожна приходить после команды START_SESSION.
<br>
`sessionId` у данный команды и команды `START_SESSION` должны совпадать.

Типа стриминга ожидается, либо `CAMERA` для обработки видео с камер в реальном времени,
либо `FILE` для обработки видео из файла.
<br>
Если тип стриминга `CAMERA`, то в `uri` содержится ссылка до видео потока камеры.
<br>
Если тип стриминга `FILE`, то в `uri` содержится ссылка для скачивания видеофайла.

#### SET_CALIBRATION

```shell
{
  "command": "SET_CALIBRATION",
  "videoId": ... // ID видео,
  "sessionId": ..., // ID сессии
  "calibrationPointList": [{
    "xScreen": ..., // Координата X на изображении
    "yScreen": ..., // Координата Y на изображении
    "xWorld": ..., // Координата X в пространстве
    "yWorld": ..., // Координата Y в пространтсве
    "zWorld": ... // Координата Z в пространстве
  },
  ...
  ]
}
```
Определяет точки калибрации камры с ID `videoId`. 
<br>
`sessionId` у данный команды и команды `START_SESSION` должны совпадать.
<br>
Данная команда должна приходить после того, как видео с ID `videoId` было
добавлено командой `APPEND_VIDEO`.

#### START_STREAMING

```shell
{
  "command": "START_STREAMING",
  "sessionId": ... // ID сессии
}
```
Запускает обработку видео файлов, полученый в командах `APPEND_VIDEO`.
<br>
`sessionId` у данный команды и команды `START_SESSION` должны совпадать.

#### STOP_SESSION

```shell
{
  "command": "STOP_SESSION",
  "sessionId": ... // ID сессии
}
```
Останавливает сессию с ID `sessionId` и все, что с ней связано.
`sessionId` должна совпадать с тем, что пришло в `START_SESSION`.

### BBoxExpander
Для работы с моделью BBoxExpander, потребуется скачать веса по [ссыке](https://disk.yandex.ru/d/N24WGesxaRgd_Q) 
и разместить данный файл в папку, которую Вы собираетесь указать в `--bbox_expander_weights`.
<br>
Изначально это папка `data/bbox_expander/weight/model_final.pth`, поэтому, 
если Вы не собираетесь изменять данный алгумент, поместите веса туда.
