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
