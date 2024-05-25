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

#데이터세트 만들기 위한 생성기 정의
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

#앞서 만든 데이터세트 생성기로 데이터세트 생성
#훈련, 검증 데이터에 데이터 증식 및 전처리 적용(ex.픽셀 맞춤, 배치)
#테스트 데이터세트 생성
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
