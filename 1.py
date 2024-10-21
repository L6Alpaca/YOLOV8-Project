import wandb
import os
import yaml
from shutil import copytree, ignore_patterns, copy
import xml.etree.ElementTree as ET
from ultralytics import YOLO

# Log in to wandb with API key
wandb.login(key='236420adb4c482d7a7965b7ddf6962385d55471a')

root_path = './kaggle/input/fruit-images-for-object-detection/'
os.listdir(root_path)
train_data_path = os.path.join(root_path, 'train_zip/train')
test_data_path = os.path.join(root_path, 'test_zip/test')

'''All.xml and.jpg file names'''
train_data_description = os.listdir(train_data_path)
test_data_description = os.listdir(test_data_path)

'''train_annotaion_file_paths and test_annotation_file_paths contains all.xml file paths
   train_image_file_paths and test_image_file_paths contains all.jpg file paths'''
train_annotation_file_paths = [os.path.join(train_data_path, i) for i in train_data_description if '.xml' in i]
train_image_file_paths = [os.path.join(train_data_path, i) for i in train_data_description if '.jpg' in i]

test_annotation_file_paths = [os.path.join(test_data_path, i) for i in test_data_description if '.xml' in i]
test_image_file_paths = [os.path.join(test_data_path, i) for i in test_data_description if '.jpg' in i]




# Create a validation set from the training data (e.g., 20% of training data)
import random
validation_size = int(0.2 * len(train_image_file_paths))
validation_image_file_paths = random.sample(train_image_file_paths, validation_size)
train_image_file_paths = [img_path for img_path in train_image_file_paths if img_path not in validation_image_file_paths]

validation_annotation_file_paths = [os.path.join(train_data_path, os.path.basename(img_path).replace('.jpg', '.xml')) for img_path in validation_image_file_paths]
train_annotation_file_paths = [img_path for img_path in train_annotation_file_paths if img_path not in validation_annotation_file_paths]
#print(len(validation_annotation_file_paths),len(train_annotation_file_paths))
#print(f'length of training Data {len(train_image_file_paths)}, length of test data {len(test_image_file_paths)}')

'''Creating required directories to labels'''

for i in ['train/labels','test/labels', 'val/labels']:
    if not os.path.isdir(os.path.join('./kaggle/working/object_detection/',i)):
        os.makedirs(os.path.join('./kaggle/working/object_detection/',i))


'''Copying all images to required directories'''
for i in train_image_file_paths:
    copy(i,'./kaggle/working/object_detection/train/images/')
for i in test_image_file_paths:
    copy(i,'./kaggle/working/object_detection/test/images/')
for i in validation_image_file_paths:
    copy(i,'./kaggle/working/object_detection/val/images/')


def convert_xml_to_txt(label_path, xml_file, class_dict, destination_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    x = root.find('filename').text
    txt_file = x.replace('.jpg', '.txt')
    txt_file = os.path.join(destination_file, txt_file)
    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_dict:
                continue
            class_id = class_dict[class_name]
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)

            x_center = (xmin + xmax) / 2 / image_width if image_width!= 0 else (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2 / image_height if image_height!= 0 else (ymin + ymax) / 2
            width = (xmax - xmin) / image_width if image_width!= 0 else (xmax - xmin)
            height = (ymax - ymin) / image_height if image_height!= 0 else (ymax - ymin)

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            f.write(line)


class_dict = {'apple': 0, 'banana': 1, 'orange': 2}


dest_path = './kaggle/working/object_detection/train/labels/'
for i in train_annotation_file_paths:
    convert_xml_to_txt(train_annotation_file_paths, i, class_dict, dest_path)

dest_path = './kaggle/working/object_detection/test/labels/'
for i in test_annotation_file_paths:
    convert_xml_to_txt(test_annotation_file_paths, i, class_dict, dest_path)

dest_path = './kaggle/working/object_detection/val/labels/'
for i in validation_annotation_file_paths:
    convert_xml_to_txt(validation_annotation_file_paths, i, class_dict, dest_path)
#done


'''yaml file path, you can check the file'''
yaml_path = "./data.yaml"
model = YOLO('yolov8n.pt')

'''Training model'''
model.to(device="cuda")
model.train(data=yaml_path, epochs=20, batch=16, patience = 5)