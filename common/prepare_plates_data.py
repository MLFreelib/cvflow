import cv2
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import os
import shutil
import argparse


def move_to_folder(filename, to_folder, dataset_folder='datasets/russian_plates'):
    # move val images, labels, annotations to 'val' folder
    label_filename = filename[:filename.find('.')] + '.txt'
    annot_filename = filename[:filename.find('.')] + '.annot'
    try:
        os.rename(os.path.join(dataset_folder, 'images/train/', filename),
        os.path.join(dataset_folder, f'images/{to_folder}/', filename))
    except FileNotFoundError:
        print(f'Unable to move {filename} as it does not exist')
    
    try:
        os.rename(os.path.join(dataset_folder, 'labels/train/', label_filename),
        os.path.join(dataset_folder, f'labels/{to_folder}/', label_filename))
    except FileNotFoundError:
        print(f'Unable to move {label_filename} as it does not exist')
    
    try:
        os.rename(os.path.join(dataset_folder, 'annotations/train/', annot_filename),
        os.path.join(dataset_folder, f'annotations/{to_folder}/', annot_filename))
    except FileNotFoundError:
        print(f'Unable to move {annot_filename} as it does not exist')


def prepare_russian_plates_bboxes(raw_data_folder='raw_data', save_folder='datasets/russian_plates'):
    with open(os.path.join(raw_data_folder, 'train.json')) as f:
        # load labels
        train_labels = json.load(f)
        
    # iterate over labels and transform bboxes
    for train_label in tqdm(train_labels):
        rows = []
        rows_annot = []
        
        image_path = os.path.join(save_folder, 'images', train_label['file'])
        
        try:
            img = cv2.imread(image_path)
            image_height, image_width, _ = img.shape

        except AttributeError:
            print('Corrupted image', image_path)
            continue
        
        nums = train_label['nums']
        generic_name = train_label['file'][train_label['file'].find('/') + 1:train_label['file'].find('.')]

        for i, num in enumerate(nums):
            # bbox here is in fact 4 points (not exactly rectangle)
            x_coordinates, y_coordinates = zip(*num['box'])
            # min coordinates
            x_min, y_min = min(x_coordinates), min(y_coordinates)
            # max coordinates
            x_max, y_max = (max(x_coordinates), max(y_coordinates))
            # image height
            height = y_max - y_min
            # image width
            width = x_max - x_min
            
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            # find a crop for an image (will be used by OCR model)
            crop_img = img[y_min:y_min+height, x_min:x_min+width]
            # return bbox in yolo format
            # yolo requires center coordinates to be normalized, bbox w/h as well
            x_center /= image_width
            width /= image_width
            y_center /= image_height
            height /= image_height
            # append annotation text
            rows_annot.append(num['text'])
            # append bbox coordinates, '0' stands for 'plate' class
            rows.append(' '.join(['0', str(x_center), str(y_center), str(width), str(height)]))

            crop_name = generic_name + f'_{i+1}.jpg'
            
            try:
                # save crop
                cv2.imwrite(os.path.join(save_folder, f'crops/train/{crop_name}'), crop_img) 
            except Exception as e:
                print('Corrupted crop', crop_name)
                continue
        
        # names for labels and annotations
        label_name = generic_name + '.txt'
        annot_name = generic_name + '.annot'
        
        with open(os.path.join(save_folder, 'labels/train/',  label_name), 'w') as f:
            # write bboxes
            for row in rows:
                f.write(row + '\n')
            
        with open(os.path.join(save_folder, 'annotations/train/', annot_name), 'w') as f:
            # write annotations
            for row in rows_annot:
                f.write(row + '\n')
                
    # get all image names
    names = os.listdir(os.path.join(save_folder,'images/train/'))
    # split train - test - val
    train_names, dev_names = train_test_split(names, random_state=42, test_size=0.2)
    test_names, val_names = train_test_split(dev_names, random_state=42, test_size=0.5)
    
    for test_name in test_names:
        move_to_folder(test_name, 'test')

    for val_name in val_names:
        move_to_folder(val_name, 'val')


def prepare_trains_plates_bboxes(data_folder, trains_json_path, save_folder='datasets/trains_plates'):
    with open(trains_json_path) as f:
        data = json.load(f)
        
    for i in range(len(data)):
        x = data[i]
        path = x['file_upload']
        # path fixes
        path = path[path.find('-') + 1:]
        path = path.replace('split-video.com', '(split-video.com)')
        annotations = x['annotations'][0] # we had only one annotator
        
        try:
            img = cv2.imread(os.path.join(data_folder, path))
            image_height, image_width, _ = img.shape
        except AttributeError:
            print('Corrupted image', path)
            continue
        
        rows = []
        rows_annot = []

        generic_name = str(i)

        for j,bbox in enumerate(annotations['result']):
            # prepare bboxes for yolo format
            values = bbox['value']
            x = values['x'] / 100#* a.size[0])
            y = values['y'] / 100# * a.size[1])
            w = values['width'] / 100# * a.size[0])
            h = values['height'] / 100# * a.size[1])
            # calculate center coordinates
            x_center = x + w / 2
            y_center = y + h / 2
            
            # hopefully where is an annotation
            annot_text = bbox['meta']['text'][0] if 'meta' in bbox else ''
            # crop
            crop_img = img[y:y+h, x:x+w]
            crop_name = generic_name + f'_{j+1}.jpg'

            if annot_text:
                # if we have annotation, then consider the bbox valid
                rows_annot.append(annot_text)
                rows.append(' '.join(['0', str(x_center), str(y_center), str(w), str(h)]))
                cv2.save(os.path.join(save_folder, f'crops/train/{crop_name}'), crop_img)
        
        # names for label and annotation
        label_name = generic_name + '.txt'
        annot_name = generic_name + '.annot'
        
        with open(os.path.join(save_folder, 'labels/train/', label_name), 'w') as f:
            for row in rows:
                f.write(row + '\n')
            
        with open(os.path.join(save_folder, 'annotations/train/', annot_name, 'w')) as f:
            for row in rows_annot:
                f.write(row + '\n')
                
        shutil.copyfile(os.path.join(data_folder, path), os.path.join(save_folder, f'images/train/{str(i)}.jpg'))
        
    names = os.listdir('datasets/trains_plates/images/train/')
    train_names, test_names = train_test_split(names, random_state=42, test_size=0.2)
    test_names, val_names = train_test_split(test_names, random_state=42, test_size=0.5)
    
    # move test images/labels/annotations to 'test' folder
    for test_name in test_names:
        move_to_folder(test_name, 'test')

    # move val images/labels/annotations to 'val' folder
    for val_name in val_names:
        move_to_folder(val_name, 'val')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trains_json_path', type=str, default='Разметка2/data/project-4-at-2022-08-24-14-36-42298d34.json')
    parser.add_argument('--trains_data_path', type=str, default='Разметка2')
    args = parser.parse_args()
    
    prepare_russian_plates_bboxes()
    prepare_trains_plates_bboxes(args.trains_data_path, args.trains_json_path)