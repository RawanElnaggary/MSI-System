import math
import os
import shutil
from PIL import Image
import random
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img

# To get the needed augmented images per class
def round_up (n):
    if n < 10:
        return 10
    digits = len(str(n))
    base = 10 ** (digits - 1)
    return math.ceil(n / base) * base

initialDataset = "InitialDataset"
finalDataset = "FinalDataset"

# Check if final folder exists and delete it then create a new one
if os.path.exists(finalDataset):
    shutil.rmtree(finalDataset)
os.mkdir(finalDataset)

# To store the number of good images in each folder
foldersImgsCount = {}

# Loop through the initial folder to copy uncorrupted images to the final folder
for folder in os.listdir(initialDataset):
    imgsCount = 0
    initialFolder = os.path.join(initialDataset, folder)
    if not os.path.isdir(initialFolder):
        continue

    finalFolder = os.path.join(finalDataset, folder)
    os.makedirs(finalFolder, exist_ok=True)

    for file in os.listdir(initialFolder):
        initialFile = os.path.join(initialFolder, file)
        try:
            with Image.open(initialFile) as img:
                img.verify()
            with Image.open(initialFile) as img:
                img.load()
            # If image is corrupted don't copy it
            shutil.copy(initialFile, finalFolder)
            imgsCount += 1
        except Exception:
            continue

    foldersImgsCount[folder] = imgsCount

# To calculate the needed images in each folder later
totalTrainingImgs = 0

# Split data into 80% in a training folder and 20% in a testing folder
for folder in os.listdir(finalDataset):
    currentFolder = os.path.join(finalDataset, folder)

    trainImgsCount = math.floor(foldersImgsCount[folder] * 0.8)
    foldersImgsCount[folder] = trainImgsCount
    totalTrainingImgs += trainImgsCount
    trainingImgs = random.sample(os.listdir(currentFolder), trainImgsCount)

    trainFolder = os.path.join(currentFolder, "Train")
    os.makedirs(trainFolder, exist_ok=True)
    for imgName in trainingImgs:
        imgPath = os.path.join(currentFolder, imgName)
        shutil.move(imgPath, trainFolder)

    testFolder = os.path.join(currentFolder, "Test")
    os.makedirs(testFolder, exist_ok=True)
    for imgName in os.listdir(currentFolder):
        imgPath = os.path.join(currentFolder, imgName)
        if os.path.isdir(imgPath):
            continue
        shutil.move(imgPath, testFolder)

# Get the number of images needed in each training folder
totalTrainingImgsNeeded = totalTrainingImgs + math.ceil(totalTrainingImgs * 0.4) # Target is at least 40% more
imgsPerClass = math.ceil(totalTrainingImgsNeeded / len(foldersImgsCount))

if max(foldersImgsCount.values()) > imgsPerClass:
    targetCount = round_up(max(foldersImgsCount.values()))
else:
    targetCount = round_up(imgsPerClass)

# Keras augmentation settings
imgAug = ImageDataGenerator (
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.15,
    horizontal_flip = True,
    brightness_range = [0.5, 1.5],
    channel_shift_range = 15,
    shear_range = 0.15,
    fill_mode = "nearest"
)

# Apply augmentation for each training folder till it reaches the maximum size needed
for folder in os.listdir(finalDataset):
    classFolder = os.path.join(finalDataset, folder, "Train")
    if not os.path.isdir(classFolder):
        continue

    images = os.listdir(classFolder)
    imagesCount = len(images)
    neededImagesCount = targetCount - imagesCount

    if neededImagesCount <= 0:
        continue

    # If the needed images count don't exceed the images count in the folder then augment random unrepeated images
    if neededImagesCount <= imagesCount:
        chosenImages = random.sample(images, neededImagesCount)
        for imgName in chosenImages:
            imgPath = os.path.join(classFolder, imgName)
            img = load_img(imgPath)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for batch in imgAug.flow(x,
                                      batch_size = 1,
                                      save_to_dir = classFolder,
                                      save_prefix="aug",
                                      save_format="jpg"):
                break

    # If the needed images count exceeds the images count in the folder then augment all the images multiple times
    # And if there is a remainder then randomly select some  unrepeated images
    else:
        augTimes = neededImagesCount // imagesCount
        remainder = neededImagesCount % imagesCount

        for imgName in images:
            imgPath = os.path.join(classFolder, imgName)
            img = load_img(imgPath)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            for _ in range(augTimes):
                for batch in imgAug.flow(x,
                                         batch_size = 1,
                                         save_to_dir = classFolder,
                                         save_prefix = f"{imgName}_aug",
                                         save_format = "jpg"):
                    break

        if remainder > 0:
            chosenImages = random.sample(images, remainder)
            for imgName in chosenImages:
                imgPath = os.path.join(classFolder, imgName)
                img = load_img(imgPath)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)

                for batch in imgAug.flow(x,
                                         batch_size = 1,
                                         save_to_dir = classFolder,
                                         save_prefix = "aug",
                                         save_format = "jpg"):
                    break
