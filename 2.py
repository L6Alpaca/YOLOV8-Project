import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("./runs/detect/train3/weights/best.pt")

'''Load the image'''
img = cv2.imread('./kaggle/input/fruit-images-for-object-detection/test_zip/test/mixed_25.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = model(img)
#print(len(result[0].boxes))
result_img = result[0].plot()  # Plot results on the image
plt.figure(figsize=(10, 10))
plt.imshow(result_img)
plt.show()

'''metrics = model.val()
#map——Mean Average Precision 平均精度均值
print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")    #置信度0.5~0.95——0.6419243962600525
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")  #置信度0.5——0.9895777628885968
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")  #置信度0.7——0.7682203103123189'''