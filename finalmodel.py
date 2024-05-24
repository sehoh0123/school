import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from IPython.display import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
#from imageio import imread
import cv2
from sklearn.model_selection import train_test_split
import zipfile
import yaml
from ultralytics import YOLO
from tensorflow.keras.models import Model

#이미지 데이터 경로 설정
image_dir = Path('image data/train data')
#이미지 경로 및 라벨링, 데이터셋화
filepaths = list(image_dir.glob(r'**/*.jpeg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
#경로와 라벨 결합
train_df = pd.concat([filepaths, labels], axis=1)
#이미지 데이터 경로 설정
image_dir = Path('image data/test data')
#이미지 경로 및 라벨링, 데이터셋화
filepaths = list(image_dir.glob(r'**/*.jpeg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')
#경로와 라벨 결합
test_df = pd.concat([filepaths, labels], axis=1)
# Drop GT images
#image_df = image_df[image_df['Label'].apply(lambda x: x[-2:] != 'GT')]
#훈련 데이터와 테스트 데이터 스플릿
#train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

#훈련 데이터가 적어 과대적합이 일어날 수 있기 때문에 여러 랜덤한 변환을 적용하여 샘플을 늘림
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2, #검증 데이터 비율
    width_shift_range=0.2, #수평 이동
    height_shift_range=0.2, #수직 이동
    shear_range=0.2, #전단 적용
    horizontal_flip=True, #수평으로 뒤집기
    fill_mode='nearest' #픽셀 채우기
)
#데이터 증식은 훈련데이터에만 적용
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

#훈련, 검증, 테스트 데이터에 데이터 증식 및 전처리 적용(ex.픽셀 맞춤, 배치)
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    batch_size=len(train_df['Label']),
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=len(train_df['Label']),
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=len(test_df['Label']),
    shuffle=False
)
Found 172 validated image filenames belonging to 6 classes.
Found 42 validated image filenames belonging to 6 classes.

Found 93 validated image filenames belonging to 6 classes.

model_yolo = YOLO('best_final.pt')

results_train = model_yolo.predict(source=train_images.filepaths, save=True)
results_val = model_yolo.predict(source=val_images.filepaths, save=True)
results_test = model_yolo.predict(source=test_images.filepaths, save=True)
results_lst=[results_train,results_val,results_test]

0: 416x416 1 excavator, 4 piles, 1: 416x416 1 excavator, 2: 416x416 1 excavator, 3: 416x416 1 excavator, 4: 416x416 1 excavator, 1 plant, 5: 416x416 1 basin, 1 brick, 1 excavator, 1 mixed, 1 steel, 6: 416x416 1 dump truck, 1 excavator, 7: 416x416 1 excavator, 8: 416x416 1 excavator, 9: 416x416 1 excavator, 2 plants, 10: 416x416 1 excavator, 11: 416x416 1 pile, 14 steels, 12: 416x416 1 excavator, 13: 416x416 1 excavator, 14: 416x416 2 piles, 15: 416x416 1 pile, 16: 416x416 1 pile, 17: 416x416 2 piles, 1 plant, 18: 416x416 1 pile, 19: 416x416 (no detections), 20: 416x416 1 pile, 21: 416x416 2 piles, 22: 416x416 1 excavator, 1 pile, 23: 416x416 1 excavator, 1 pile, 24: 416x416 3 piles, 1 plant, 9 steels, 25: 416x416 1 dump truck, 3 piles, 26: 416x416 2 piles, 27: 416x416 1 plant, 28: 416x416 17 plants, 29: 416x416 1 excavator, 8 plants, 30: 416x416 4 plants, 31: 416x416 21 plants, 32: 416x416 5 plants, 33: 416x416 4 plants, 1 plastic barrel, 12 woods, 34: 416x416 3 plants, 35: 416x416 6 plants, 36: 416x416 1 excavator, 9 plants, 37: 416x416 1 bricks, 1 excavator, 5 plants, 38: 416x416 1 plant, 4 woods, 39: 416x416 1 excavator, 4 plants, 9 woods, 40: 416x416 2 plants, 20 woods, 41: 416x416 1 excavator, 2 plants, 42: 416x416 1 excavator, 1 plant, 43: 416x416 5 plants, 44: 416x416 3 plants, 11 woods, 45: 416x416 1 bricks, 2 concrete bags, 19 plants, 1 plastic barrel, 46: 416x416 2 plants, 47: 416x416 16 plants, 48: 416x416 1 brick, 2 brickss, 12 concrete bags, 1 excavator, 2 plants, 1 plastic barrel, 49: 416x416 5 bricks, 1 plant, 50: 416x416 11 bricks, 1 plant, 51: 416x416 1 excavator, 1 plant, 52: 416x416 4 bricks, 2 brickss, 3 plants, 1 plastic barrel, 53: 416x416 1 excavator, 2 plants, 54: 416x416 2 plants, 55: 416x416 5 plants, 56: 416x416 1 excavator, 1 plant, 57: 416x416 1 excavator, 3 plants, 2 woods, 58: 416x416 2 plants, 59: 416x416 8 bricks, 7 plants, 60: 416x416 4 plants, 61: 416x416 1 excavator, 4 plants, 62: 416x416 1 bricks, 2 plants, 63: 416x416 1 Trowel, 1 basin, 5 bricks, 3 brickss, 1 plastic barrel, 64: 416x416 2 brickss, 65: 416x416 2 basins, 2 brickss, 1 concrete bag, 1 plastic barrel, 66: 416x416 1 bricks, 67: 416x416 1 basin, 3 bricks, 7 brickss, 1 plastic barrel, 68: 416x416 1 Trowel, 1 bricks, 1 plastic barrel, 69: 416x416 1 basin, 16 bricks, 1 bricks, 70: 416x416 1 basin, 30 bricks, 3 brickss, 1 plastic barrel, 71: 416x416 3 brickss, 1 plastic barrel, 72: 416x416 1 bricks, 4 plastic barrels, 1 wood, 73: 416x416 1 brick, 2 brickss, 1 plastic barrel, 74: 416x416 1 bricks, 75: 416x416 1 bricks, 76: 416x416 1 basin, 3 brickss, 2 plastic barrels, 77: 416x416 6 bricks, 3 brickss, 2 steels, 78: 416x416 1 bricks, 79: 416x416 2 brickss, 1 stud, 80: 416x416 1 bricks, 14 steels, 81: 416x416 5 brickss, 82: 416x416 2 brickss, 1 plastic barrel, 83: 416x416 1 basin, 10 bricks, 3 brickss, 84: 416x416 1 brick, 1 bricks, 1 concrete bag, 1 plastic barrel, 85: 416x416 1 brick, 4 brickss, 1 steel, 86: 416x416 1 concrete bag, 1 plastic barrel, 87: 416x416 1 basin, 1 brick, 4 brickss, 1 plastic barrel, 88: 416x416 1 basin, 2 brickss, 89: 416x416 2 basins, 8 bricks, 5 brickss, 90: 416x416 1 basin, 9 bricks, 2 brickss, 1 plastic barrel, 91: 416x416 1 basin, 1 bricks, 2 concrete bags, 1 plastic barrel, 92: 416x416 1 basin, 2 brickss, 1 plastic barrel, 93: 416x416 2 basins, 4 bricks, 5 brickss, 94: 416x416 3 basins, 21 bricks, 3 brickss, 11 concrete bags, 95: 416x416 11 bricks, 1 bricks, 1 concrete bag, 2 steels, 96: 416x416 1 bricks, 97: 416x416 2 brickss, 3 steels, 98: 416x416 1 Trowel, 99: 416x416 1 wood, 100: 416x416 4 plastic barrels, 2 square cans, 101: 416x416 (no detections), 102: 416x416 1 Trowel, 103: 416x416 1 plastic barrel, 1 square can, 104: 416x416 1 Trowel, 1 plastic barrel, 105: 416x416 1 board, 106: 416x416 1 roller, 1 square can, 107: 416x416 1 long handle float, 1 roller, 1 square can, 108: 416x416 (no detections), 109: 416x416 1 board, 3 plastic barrels, 2 woods, 110: 416x416 4 plastic barrels, 111: 416x416 1 roller, 1 square can, 112: 416x416 1 square can, 113: 416x416 1 plastic barrel, 114: 416x416 1 plastic barrel, 115: 416x416 (no detections), 116: 416x416 (no detections), 117: 416x416 1 plastic barrel, 118: 416x416 1 plastic barrel, 119: 416x416 3 square cans, 120: 416x416 5 plastic barrels, 121: 416x416 1 Trowel, 2 woods, 122: 416x416 1 roller, 123: 416x416 1 bricks, 1 roller, 2 square cans, 124: 416x416 (no detections), 125: 416x416 3 plastic barrels, 126: 416x416 2 rollers, 127: 416x416 1 roller, 128: 416x416 2 rollers, 2 square cans, 129: 416x416 1 roller, 1 square can, 130: 416x416 1 roller, 1 square can, 131: 416x416 1 square can, 132: 416x416 2 rollers, 2 square cans, 133: 416x416 14 woods, 134: 416x416 1 plastic barrel, 2 rollers, 1 square can, 1 steel, 135: 416x416 2 rainboots, 136: 416x416 1 concrete bag, 1 long handle float, 1 stud, 137: 416x416 1 long handle float, 1 plasterer shoe, 138: 416x416 2 Trowels, 2 basins, 3 plastic barrels, 2 rainboots, 139: 416x416 2 basins, 2 plastic barrels, 6 steels, 140: 416x416 5 plastic barrels, 141: 416x416 2 Trowels, 142: 416x416 2 plasterer shoes, 143: 416x416 1 Trowel, 2 plasterer shoes, 144: 416x416 1 plastic barrel, 145: 416x416 2 Trowels, 1 rainboot, 146: 416x416 1 Trowel, 147: 416x416 2 Trowels, 148: 416x416 1 board, 1 long handle float, 1 plastic barrel, 149: 416x416 1 plastic barrel, 150: 416x416 1 long handle float, 2 rainboots, 151: 416x416 1 long handle float, 1 rainboot, 5 studs, 152: 416x416 2 Trowels, 153: 416x416 3 Trowels, 3 plastic barrels, 154: 416x416 1 long handle float, 2 rainboots, 4 studs, 155: 416x416 1 Trowel, 156: 416x416 1 long handle float, 3 rainboots, 157: 416x416 1 basin, 1 bricks, 3 plastic barrels, 158: 416x416 1 Trowel, 159: 416x416 3 Trowels, 1 basin, 2 plastic barrels, 1 rainboot, 3 studs, 160: 416x416 1 wood, 161: 416x416 1 long handle float, 1 mixed, 162: 416x416 1 long handle float, 1 rainboot, 163: 416x416 1 long handle float, 1 plasterer shoe, 1 plastic barrel, 164: 416x416 1 long handle float, 2 plasterer shoes, 165: 416x416 1 long handle float, 2 rainboots, 166: 416x416 1 long handle float, 2 rainboots, 167: 416x416 1 Trowel, 1 rainboot, 168: 416x416 1 Trowel, 169: 416x416 (no detections), 170: 416x416 1 long handle float, 2 plasterer shoes, 171: 416x416 2 Trowels, 1 long handle float, 2 rainboots, 4018.0ms
Speed: 0.9ms preprocess, 23.4ms inference, 0.4ms postprocess per image at shape (1, 3, 416, 416)
Results saved to runs\detect\predict4

0: 416x416 4 boards, 1 concrete bag, 2 woods, 1: 416x416 2 studs, 2 woods, 2: 416x416 13 boards, 1 wood, 3: 416x416 5 boards, 1 steel, 3 woods, 4: 416x416 3 boards, 13 woods, 5: 416x416 25 woods, 6: 416x416 6 woods, 7: 416x416 (no detections), 8: 416x416 3 studs, 1 wood, 9: 416x416 1 board, 2 brickss, 4 woods, 10: 416x416 6 woods, 11: 416x416 (no detections), 12: 416x416 1 plastic barrel, 1 wood, 13: 416x416 3 studs, 14: 416x416 1 board, 1 stud, 2 woods, 15: 416x416 2 boards, 2 studs, 16: 416x416 2 boards, 1 wood, 17: 416x416 7 woods, 18: 416x416 2 boards, 1 concrete bag, 5 studs, 19: 416x416 1 board, 3 studs, 20: 416x416 1 stud, 1 wood, 21: 416x416 7 boards, 1 stud, 13 woods, 22: 416x416 3 boards, 1 stud, 23: 416x416 3 studs, 24: 416x416 1 board, 4 woods, 25: 416x416 4 studs, 1 wood, 26: 416x416 1 stud, 1 wood, 27: 416x416 9 boards, 1 wood, 28: 416x416 1 stud, 1 wood, 29: 416x416 1 board, 1 plastic barrel, 3 woods, 30: 416x416 3 boards, 2 woods, 31: 416x416 2 boards, 25 woods, 32: 416x416 2 excavators, 9 piles, 33: 416x416 1 excavator, 34: 416x416 1 excavator, 1 mixed, 4 steels, 35: 416x416 1 dump truck, 1 excavator, 36: 416x416 1 excavator, 1 pile, 37: 416x416 1 excavator, 1 steel, 38: 416x416 2 piles, 39: 416x416 2 boards, 40: 416x416 1 excavator, 41: 416x416 1 excavator, 1439.2ms
Speed: 2.1ms preprocess, 34.3ms inference, 0.6ms postprocess per image at shape (1, 3, 416, 416)
Results saved to runs\detect\predict4

0: 416x416 3 boards, 11 woods, 1: 416x416 2 studs, 7 woods, 2: 416x416 12 boards, 2 studs, 3: 416x416 (no detections), 4: 416x416 10 boards, 2 woods, 5: 416x416 2 boards, 6: 416x416 1 board, 1 wood, 7: 416x416 9 woods, 8: 416x416 3 boards, 1 wood, 9: 416x416 1 board, 5 studs, 3 woods, 10: 416x416 1 stud, 8 woods, 11: 416x416 6 studs, 12: 416x416 4 boards, 1 wood, 13: 416x416 9 concrete bags, 2 studs, 14: 416x416 3 boards, 1 stud, 15: 416x416 4 boards, 1 wood, 16: 416x416 13 woods, 17: 416x416 7 boards, 1 wood, 18: 416x416 1 excavator, 2 piles, 19: 416x416 1 pile, 20: 416x416 1 excavator, 21: 416x416 1 excavator, 22: 416x416 2 excavators, 23: 416x416 1 excavator, 1 pile, 24: 416x416 1 excavator, 25: 416x416 1 excavator, 26: 416x416 1 excavator, 27: 416x416 4 dump trucks, 1 excavator, 28: 416x416 1 excavator, 29: 416x416 1 excavator, 1 plant, 30: 416x416 1 excavator, 31: 416x416 1 pile, 32: 416x416 3 piles, 33: 416x416 1 pile, 34: 416x416 1 excavator, 35: 416x416 1 excavator, 3 plants, 36: 416x416 1 plant, 37: 416x416 4 plants, 38: 416x416 5 plants, 5 woods, 39: 416x416 13 plants, 4 woods, 40: 416x416 1 excavator, 1 plant, 41: 416x416 4 plants, 42: 416x416 1 excavator, 2 plants, 43: 416x416 5 plants, 44: 416x416 1 excavator, 6 plants, 45: 416x416 3 plants, 1 wood, 46: 416x416 2 plants, 47: 416x416 8 plants, 48: 416x416 1 excavator, 9 plants, 49: 416x416 1 excavator, 1 plant, 50: 416x416 2 plants, 1 wood, 51: 416x416 29 bricks, 1 bricks, 1 concrete bag, 52: 416x416 3 bricks, 2 brickss, 1 plastic barrel, 53: 416x416 1 bricks, 54: 416x416 7 bricks, 1 bricks, 55: 416x416 1 bricks, 2 woods, 56: 416x416 7 bricks, 22 brickss, 6 concrete bags, 7 plastic barrels, 57: 416x416 2 Trowels, 1 bricks, 1 plant, 58: 416x416 1 basin, 2 brickss, 1 plastic barrel, 59: 416x416 1 bricks, 3 steels, 60: 416x416 1 basin, 2 brickss, 61: 416x416 4 brickss, 1 plastic barrel, 62: 416x416 1 basin, 3 brickss, 63: 416x416 1 basin, 6 brickss, 8 concrete bags, 2 plastic barrels, 64: 416x416 1 basin, 29 bricks, 1 bricks, 65: 416x416 1 basin, 1 brick, 3 brickss, 1 plastic barrel, 66: 416x416 1 plant, 2 plastic barrels, 1 roller, 1 square can, 67: 416x416 1 roller, 2 square cans, 2 steels, 68: 416x416 1 square can, 1 wood, 69: 416x416 (no detections), 70: 416x416 (no detections), 71: 416x416 1 square can, 72: 416x416 1 roller, 1 square can, 73: 416x416 1 roller, 1 square can, 74: 416x416 3 square cans, 75: 416x416 1 plastic barrel, 1 roller, 1 square can, 76: 416x416 1 roller, 77: 416x416 1 roller, 78: 416x416 1 roller, 1 square can, 79: 416x416 (no detections), 80: 416x416 1 board, 81: 416x416 (no detections), 82: 416x416 5 steels, 1 wood, 83: 416x416 1 basin, 84: 416x416 1 long handle float, 2 rainboots, 85: 416x416 3 Trowels, 2 rainboots, 86: 416x416 1 Trowel, 1 plastic barrel, 3 rainboots, 87: 416x416 1 Trowel, 88: 416x416 (no detections), 89: 416x416 1 rainboot, 90: 416x416 2 bricks, 1 steel, 91: 416x416 1 long handle float, 4 plasterer shoes, 92: 416x416 1 long handle float, 2 plasterer shoes, 1 stud, 4775.7ms
Speed: 1.1ms preprocess, 51.4ms inference, 1.0ms postprocess per image at shape (1, 3, 416, 416)
Results saved to runs\detect\predict4

truck=['dump truck', 'mixed']
plastering_tool=['Trowel', 'rainboot', 'plasterer shoe']
brick=['brick','bricks']
concrete=['concrete bag', 'basin']
liquid_material=['plastic barrel', 'square can']
merge_lst=[truck, plastering_tool, brick, concrete, liquid_material]
merge_lst_name=['truck', 'plastering_tool', 'brick', 'concrete', 'liquid_material']

class_name=[]
for i in range (len(model_yolo.names)):
  class_name.append(model_yolo.names[i])

for results in results_lst:
  #클래스 개수, 면적 리스트
  lst=[]
  for result in results:
    area_tensor=[]
    for i in range (len(result.boxes.cls)):
      area=result.boxes.xywh[i,2]*result.boxes.xywh[i,3]
      area=area.cpu().numpy().tolist()
      area_tensor.append(area)
    area_tensor = tf.convert_to_tensor(area_tensor)
    merged_tensor = np.vstack((result.boxes.cls.cpu(), area_tensor))
    lst.append(merged_tensor)
    #lst[i]는 사진 한장의 클래스와 면적에 대한 array
  #클래스 별 개수, 면적 리스트
  lst_final=[]
  lst_flat=[]
  for num1 in range (len(results)):
    class_info = {}

    # Iterate over the tensors to count occurrences and calculate total area
    #lst[result][0]는 클래스 정보
    #lst[result][1]는 면적
    for cls, area in zip(lst[num1][0], lst[num1][1]):
        if cls not in class_info:
            class_info[cls] = {
                'Count': 0,
            }
        class_info[cls]['Count'] += 1

    # Add missing classes with count 0 and area 0
    for cls in range(0,len(model_yolo.names)):
        if cls not in class_info:
            class_info[cls] = {
                'Count': 0
            }

    # Create a data frame from the class information
    df = pd.DataFrame(class_info).T.reset_index().rename(columns={'index': 'Class'})
    # Sort the data frame by class
    df = df.sort_values('Class')
    df['Class_name'] = class_name
    df['Count'].reset_index(drop=True)
    df = df.reset_index(drop=True)
    #객체 병합
    for num2 in merge_lst:
      count=0
      for num3 in num2:
        count += df['Count'][df.loc[df['Class_name'] == num3].index[0]]
        df = df.drop(df.loc[df['Class_name'] == num3].index[0])
      new_row = pd.DataFrame([{'Class': len(df), 'Count': count, 'Class_name': merge_lst_name[merge_lst.index(num2)]}])
      #df = df.append(new_row, ignore_index=True)
      df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values('Class_name')
    #데이터프레임->텐서
    tensor1 = tf.convert_to_tensor(df['Count'].reset_index(drop=True), dtype=tf.float32)
    flatten_tensor = tf.reshape(tensor1, [-1])
    lst_final.append(tensor1)
    lst_flat.append(flatten_tensor)
  if results==results_train:
    lst_train=lst_final
  elif results==results_val:
    lst_val=lst_final
  elif results==results_test:
    lst_test=lst_final

    tfidf = pd.read_csv('final_tfidf.csv', encoding='utf-8-sig', index_col=0)
    tfidf_tensor = tf.convert_to_tensor(tfidf)

    lst = [lst_train, lst_val, lst_test]

    # 개수 0인 객체는 tfidf점수는 0
    lst_train_merge = []
    lst_val_merge = []
    lst_test_merge = []
    for lst_1 in lst:
        for result in range(len(lst_1)):
            var = tf.Variable(tfidf_tensor)
            for cls in range(0, len(tfidf.columns)):
                if lst_1[result][cls] == 0:
                    for i in range(6):
                        var[i, cls].assign(0)
            merged_tensor2 = np.vstack((lst_1[result], var))
            if lst_1 == lst_train:
                lst_train_merge.append(merged_tensor2)
            elif lst_1 == lst_val:
                lst_val_merge.append(merged_tensor2)
            elif lst_1 == lst_test:
                lst_test_merge.append(merged_tensor2)

lst_merge=[lst_train_merge,lst_val_merge,lst_test_merge]

#개수*tfidf
lst_train_final=[]
lst_val_final=[]
lst_test_final=[]
for lst in lst_merge:
  for n in range (len(lst)):
    for i in range (1,len(lst[0])):
      for k in range (len(tfidf.columns)):
        lst[n][i][k]=lst[n][0][k]*lst[n][i][k]
    tensor = np.delete(lst[n], [0], axis=0)
    if lst==lst_train_merge:
      lst_train_final.append(tensor.reshape(-1))
    elif lst==lst_val_merge:
      lst_val_final.append(tensor.reshape(-1))
    elif lst==lst_test_merge:
      lst_test_final.append(tensor.reshape(-1))

images_train, labels_train = next(train_images)
images_val, labels_val = next(val_images)
images_test, labels_test = next(test_images)

object_train =np.array(lst_train_final)
object_test =np.array(lst_test_final)
object_val =np.array(lst_val_final)

# Load the pretained model
#lagorithm - neural network - deep learning
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False

inputs = pretrained_model.input
dropout = tf.keras.layers.Dropout(rate=0.2)(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(dropout)
x = tf.keras.layers.Dense(128, activation='relu')(x) #relu is the activation function for neurla network task

outputs = tf.keras.layers.Dense(6, activation='softmax')(x) #softmax is the activation function for classiciation task

model_org = tf.keras.Model(inputs=inputs, outputs=outputs)

model_org.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_org.summary()

callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
checkpointer = ModelCheckpoint(filepath="weights_org.hdf5", verbose=1, save_best_only=True)

history_org = model_org.fit(x=images_train,
    y=labels_train,
    validation_data = (images_val, labels_val),
    epochs=40,
    callbacks=[
        callback
    ]
)

model_org.load_weights('weight_org.hdf5')

pd.DataFrame(history_org.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()

pd.DataFrame(history_org.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()

results_org = model_org.evaluate(images_test, labels_test, verbose=0)

print("    Test Loss: {:.5f}".format(results_org[0]))
print("Test Accuracy: {:.2f}%".format(results_org[1] * 100))

  Test Loss: 0.62999
Test Accuracy: 75.27%

results_org
[0.6299936175346375, 0.7526881694793701]

# Predict the label of the test_images
pred_org = model_org.predict(images_test)
pred_org_final = np.argmax(pred_org,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred_org_final = [labels[k] for k in pred_org_final]

# Display the result
print(f'The first 5 predictions: {pred_org_final[:5]}')

3/3 [==============================] - 1s 231ms/step
The first 5 predictions: ['carpentry', 'carpentry', 'masonry', 'carpentry', 'carpentry']

#landscaping 결과 값
pred_org[[index for index, label in enumerate(labels_test) if np.array_equal(label, np.array([0, 0, 1, 0, 0, 0]))]]
array([[ 0.00065917,     0.47874,     0.51857,  0.00076172,  5.3606e-05,    0.001211],
       [ 0.00074776,     0.62091,     0.35035,   0.0017384,  0.00058413,    0.025672],
       [  6.925e-06,   0.0051969,     0.98524,   0.0015767,  0.00027434,   0.0077075],
       [ 0.00097892,     0.20975,     0.66966,     0.10541,   0.0017994,    0.012395],
       [ 0.00043098,     0.12602,     0.86855,  0.00084161,  9.8225e-05,   0.0040566],
       [  0.0011254,     0.94952,    0.040847,     0.00678,   0.0005434,   0.0011817],
       [ 0.00022052,      0.0442,     0.87072,    0.022448,   0.0014184,    0.060994],
       [ 0.00019543,     0.91939,    0.077441,  0.00028785,  0.00013543,   0.0025501],
       [ 2.7987e-05,    0.020384,     0.97197,   0.0036227,  0.00013443,   0.0038604],
       [ 0.00027785,     0.48776,     0.50695,   0.0022356,  0.00012124,   0.0026532],
       [ 6.2452e-05,    0.014855,       0.967,   0.0015499,  0.00040549,    0.016127],
       [ 0.00020175,    0.010034,     0.97649,    0.012506,  2.3142e-05,  0.00074509],
       [  0.0032336,    0.025084,     0.85183,    0.084615,    0.010555,    0.024682],
       [ 2.7285e-06,   0.0060405,     0.99265,  6.4226e-05,  8.5309e-05,   0.0011569],
       [ 8.4205e-05,     0.25906,     0.73215,   0.0062019,  2.9587e-05,   0.0024755],
       [ 0.00053976,    0.016278,     0.96404,   0.0032664,   0.0020887,     0.01379]], dtype=float32)

from sklearn.metrics import confusion_matrix, classification_report
y_test = list(test_df.Label)
print(classification_report(y_test, pred_org_final))
              precision    recall  f1-score   support

   carpentry       0.92      0.67      0.77        18
   earthwork       0.84      0.94      0.89        17
 landscaping       0.93      0.81      0.87        16
     masonry       0.67      0.93      0.78        15
    painting       0.83      0.36      0.50        14
  plastering       0.50      0.77      0.61        13

    accuracy                           0.75        93
   macro avg       0.78      0.75      0.74        93
weighted avg       0.80      0.75      0.75        93


from sklearn.metrics import confusion_matrix
import seaborn as sns

cf_matrix = confusion_matrix(y_test, pred_org_final, normalize='true')
plt.figure(figsize = (10,6))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)))
plt.title('Normalized Confusion Matrix')
plt.show()
model_org.save(filepath='weight_org.hdf5')

# Load the pretained model
#lagorithm - neural network - deep learning
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False

#1280+84
image_input = pretrained_model.input #이미지
object_input = Input(shape=(84,))#lst_merge[0].input #객체 탐지, 표준시방서 데이터(84,)
output1 = tf.keras.layers.Concatenate(axis=1)([pretrained_model.output, object_input]) #(1280+84,)
dropout = tf.keras.layers.Dropout(rate=0.2)(output1)

x = tf.keras.layers.Dense(128, activation='relu')(dropout)
x = tf.keras.layers.Dense(128, activation='relu')(x) #relu is the activation function for neurla network task
output = tf.keras.layers.Dense(6, activation='softmax')(x) #softmax is the activation function for classiciation task

model = tf.keras.Model(inputs=[image_input, object_input], outputs=output)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

history = model.fit(x=[images_train, object_train],
    y=labels_train,
    validation_data = ([images_val, object_val], labels_val),
    epochs=20,
    batch_size=8,
    callbacks=[
        callback
    ]
)
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Accuracy")
plt.show()
pd.DataFrame(history.history)[['loss','val_loss']].plot()
plt.title("Loss")
plt.show()

model.load_weights('weight.hdf5')
model.save(filepath='weight.hdf5')

results = model.evaluate([images_test, object_test], labels_test, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
    Test Loss: 0.59618
Test Accuracy: 80.65%

# Predict the label of the test_images
pred = model.predict([images_test, object_test])
pred_final = np.argmax(pred,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred_final = [labels[k] for k in pred_final]

# Display the result
print(f'The first 5 predictions: {pred_final[:5]}')
3/3 [==============================] - 1s 225ms/step
The first 5 predictions: ['carpentry', 'carpentry', 'carpentry', 'carpentry', 'carpentry']

'carpentry'

from sklearn.metrics import confusion_matrix, classification_report
y_test = list(test_df.Label)
print(classification_report(y_test, pred_final))
              precision    recall  f1-score   support

   carpentry       0.94      0.89      0.91        18
   earthwork       0.79      0.88      0.83        17
 landscaping       0.86      0.75      0.80        16
     masonry       0.92      0.80      0.86        15
    painting       0.73      0.57      0.64        14
  plastering       0.63      0.92      0.75        13

    accuracy                           0.81        93
   macro avg       0.81      0.80      0.80        93
weighted avg       0.82      0.81      0.81        93


from sklearn.metrics import confusion_matrix
import seaborn as sns

cf_matrix = confusion_matrix(y_test, pred_final, normalize='true')
plt.figure(figsize = (10,6))
sns.heatmap(cf_matrix, annot=True, xticklabels = sorted(set(y_test)), yticklabels = sorted(set(y_test)))
plt.title('Normalized Confusion Matrix')
plt.show()
# Display 25 picture of the dataset with their labels
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(test_df.Filepath.iloc[i]))
    ax.set_title(f"True: {test_df.Label.iloc[i]}\nPredicted: {pred_final[i]}")
plt.tight_layout()
plt.show()
model.save("saved_model.h5")

#landscaping 결과 값
pred_org[[index for index, label in enumerate(labels_test) if np.array_equal(label, np.array([0, 0, 1, 0, 0, 0]))]]
(16, 6)

vector.index(1)
2

vector=[0, 0, 0, 1, 0, 0]
ind=[index for index, label in enumerate(labels_test) if np.array_equal(label, np.array(vector))]
category_gt=labels[vector.index(1)]
category_pd=[]
category_pd_org=[]
filename=[]
pred_gt=[]
pred_pd=[]
pred_gt_org=[]
pred_pd_org=[]
for i in ind:
    category_pd_org.append(pred_org_final[i])
    category_pd.append(pred_final[i])
    filename.append(test_images.filenames[i])
    pred_gt.append(pred[i,vector.index(1)])
    pred_pd.append(np.max(pred[i]))
    pred_gt_org.append(pred_org[i,vector.index(1)])
    pred_pd_org.append(np.max(pred_org[i]))

df={'index':ind,'filename':filename,'category_gt':category_gt,'category_pd_org':category_pd_org, 'category_pd':category_pd,
    'org_pd':pred_pd_org, 'org_gt':pred_gt_org, 'propose_pd':pred_pd,'propose_gt':pred_gt}

df=pd.DataFrame(df)

df.to_csv('masonry.csv')

path=test_df.Filepath.iloc[28]
display(Image(filename=path, width=600))
#객체탐지 결과
path=test_df.Filepath.iloc[15]
filename = '/content/runs/detect/predict/'+os.path.basename(path)
display(Image(filename=filename, width=600))
df=pd.DataFrame(lst_test[34][0], index=tfidf.columns).transpose().sum().sum()
df
total = df.sum().sum()
a=pd.DataFrame(test_df.Label)
labels_tp = {v: k for k, v in labels.items()}
a['predict_org'] = pred_org_final
a['predict'] = pred_final
a['result'] = 0
a['object_num']=0
a=a.reset_index(drop=True)
for i in range(93):
  a['object_num'][i]= pd.DataFrame(lst_test[i][0], index=tfidf.columns).transpose().sum().sum()
  if pred[i][labels_tp[a['Label'][i]]]>pred_org[i][labels_tp[a['Label'][i]]]:
    a['result'][i]=1
  else:
    pass
a
a.to_csv('a.csv', index=False)
pred[12]
pred_org[12]
labels

def check_feature_org(num_layer, id):
  object_data=object_test[id].reshape(1, 84)
  feature_layer = model.get_layer(num_layer)
  feature_layer_org = model_org.get_layer(num_layer)
  # Create a new model that will output the feature maps from the chosen layer
  feature_map_model = Model(inputs=model.input, outputs=feature_layer.output)
  feature_map_model_org = Model(inputs=model_org.input, outputs=feature_layer_org.output)
  # Load and preprocess your image data
  input_img=test_df.Filepath.iloc[id]
  img = tf.keras.preprocessing.image.load_img(input_img, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  #img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
  img_array = tf.expand_dims(img_array, axis=0)
  # Get the feature maps for your image
  feature_map = feature_map_model.predict([img_array, object_data])[0]
  feature_map_org = feature_map_model_org.predict(img_array)[0]
  print(feature_map.shape)

  plt.figure(figsize=(10, 10))
  for i in range(feature_map_org.shape[-1]):
      plt.subplot((feature_map_org.shape[-1]//10)+1, 10, i + 1)  # Assumes 192 channels, adjust the subplot arrangement based on the actual number of channels
      plt.imshow(feature_map_org[:, :, i], cmap='viridis')  # Choose a suitable colormap
      plt.axis('off')
  plt.show()
def check_feature(num_layer, id):
  object_data=object_test[id].reshape(1, 84)
  feature_layer = model.get_layer(num_layer)
  #feature_layer_org = model_org.get_layer(num_layer)
  # Create a new model that will output the feature maps from the chosen layer
  feature_map_model = Model(inputs=model.input, outputs=feature_layer.output)
  #feature_map_model_org = Model(inputs=model_org.input, outputs=feature_layer_org.output)
  # Load and preprocess your image data
  input_img=test_df.Filepath.iloc[id]
  img = tf.keras.preprocessing.image.load_img(input_img, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  #img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
  img_array = tf.expand_dims(img_array, axis=0)
  # Get the feature maps for your image
  feature_map = feature_map_model.predict([img_array, object_data])[0]
  #feature_map_org = feature_map_model_org.predict(img_array)[0]
  print(feature_map.shape)
  #print(feature_map.shape[0]*25/56, feature_map.shape[3])
        #*25/56,((feature_map.shape[3]//10)+1)*feature_map.shape[0]/112*5)
  #print(((feature_map.shape[2]//10)+1))

  plt.figure(figsize=(int(feature_map.shape[0]*25/56),int(((feature_map.shape[2]//10)+1)*feature_map.shape[0]/112*5)))
  for i in range(feature_map.shape[-1]):
      plt.subplot((feature_map.shape[-1]//10)+1, 10, i + 1)  # Assumes 192 channels, adjust the subplot arrangement based on the actual number of channels
      plt.imshow(feature_map[:, :, i], cmap='viridis')  # Choose a suitable colormap
      plt.axis('off')
  plt.show()
plt.figure()
plt.imshow(tf.keras.preprocessing.image.load_img(test_df.Filepath.iloc[13], target_size=(224, 224)))
plt.colorbar()
plt.show()
check_feature('Conv1', 41)
check_feature('Conv1', 4)
def check_feature_down(num_layer, id):
  object_data=object_test[id].reshape(1, 84)
  feature_layer = model.get_layer(num_layer)
  #feature_layer_org = model_org.get_layer(num_layer)
  # Create a new model that will output the feature maps from the chosen layer
  feature_map_model = Model(inputs=model.input, outputs=feature_layer.output)
  #feature_map_model_org = Model(inputs=model_org.input, outputs=feature_layer_org.output)
  # Load and preprocess your image data
  input_img=test_df.Filepath.iloc[id]
  img = tf.keras.preprocessing.image.load_img(input_img, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  #img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
  img_array = tf.expand_dims(img_array, axis=0)
  # Get the feature maps for your image
  feature_map = feature_map_model.predict([img_array, object_data])[0]
  #feature_map_org = feature_map_model_org.predict(img_array)[0]
  #print(feature_map.shape)
  #print(feature_map.shape[0]*25/56, feature_map.shape[3])
        #*25/56,((feature_map.shape[3]//10)+1)*feature_map.shape[0]/112*5)
  #print(((feature_map.shape[2]//10)+1))

  plt.figure(figsize=(int(feature_map.shape[0]*25/56),int(((feature_map.shape[2]//10)+1)*feature_map.shape[0]/112*5)))
  for i in range(feature_map.shape[-1]):
      plt.subplot((feature_map.shape[-1]//10)+1, 10, i + 1)  # Assumes 192 channels, adjust the subplot arrangement based on the actual number of channels
      plt.imshow(feature_map[:, :, i], cmap='viridis')  # Choose a suitable colormap
      plt.axis('off')
  #plt.show()
  plt.savefig('/content/{}.png'.format(id))
for index in range(93):
  check_feature_down('block_2_depthwise', index)
import os
import zipfile

# Replace the list below with the actual file paths of your images.
image_file_paths = []
for i in range(93):
  image_file_paths.append('/content/{}.png'.format(i))

# Create a directory to save the compressed images (if it doesn't exist).
output_directory = 'images1'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the name of the ZIP file to create.
zip_filename = 'images1.zip'

# Compress the images and add them to the ZIP file.
with zipfile.ZipFile(os.path.join(output_directory, zip_filename), 'w') as zipf:
    for idx, image_path in enumerate(image_file_paths):
        # Extract the filename from the image path
        image_name = os.path.basename(image_path)

        # Add the compressed image to the ZIP file
        zipf.write(image_path, image_name)

        # Delete the temporary compressed image file (optional)
        os.remove(image_path)

check_feature('block_16_depthwise', 2)
