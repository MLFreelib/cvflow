### Модель для локализации дефектов

## Запуск

Для запуска необходимо в качестве дополнительнго аргумента ввести
путь к примерам дефектов. Формат хранения файлов должен быть следующим:  

    ../Directory  
        --templates  
        --boxes

Где в templates хранятся изображения с дефектами, а в boxes файлы с разметкой,
где каждый файл в формате .txt и содержит название дефекта и координаты дефекта.   
Пример: 

    Crack 0.1 0.2 0.2 0.4
    Resin 0.15 0.1 0.3 0.2

Пример запуска:

        python3 run.py --videofile {file_path},{file_path} --temppath {directory with templates} --w {path_to_weights}

## Обучение

Для обучения потребуется запустить команду

    python3 train.py --data {path to data}

Формат данных:

    ../Directory
        --train
            --images
            --bboxes

        --valid
            --images
            --bboxes

Формат хранения данных соответствует формату данных для запуска модели.