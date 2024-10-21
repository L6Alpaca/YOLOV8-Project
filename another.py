import wandb
import os
import yaml
from shutil import copytree, ignore_patterns, copy
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image


save_dir = "./kaggle/working/object_detection/val/images2"

img_dir = "./kaggle/working/object_detection/val/images"
img_dir = os.listdir(img_dir)
img_path = "./kaggle/working/object_detection/val/images"
for img in img_dir:
    path = os.path.join(img_path,img)
    img1 = cv2.imread(path)
    img1 = Image.fromarray(img1, 'RGB')
    img1.save(os.path.join(save_dir,img))
    #cv2.imwrite(os.path.join(save_dir,img),img1)

'''plt.imshow(img1)
plt.show()'''

