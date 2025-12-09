import os
import shutil
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img
import random

initialDataset = "InitialDataset"
finalDataset = "FinalDataset"

# Check if final folder exists and delete it then create a new one
if os.path.exists(finalDataset):
    shutil.rmtree(finalDataset)
os.mkdir(finalDataset)

# Loop through the initial folder to copy uncorrupted images to the final folder
for folder in os.listdir(initialDataset):
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
        except Exception:
            continue

# Keras augmentation settings
imgAug = ImageDataGenerator (
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.15,
    horizontal_flip = True,
    brightness_range = [0.7, 1.3],
    shear_range = 0.05,
    fill_mode = "nearest"
)

# Apply augmentation for each folder till it reaches the maximum size needed
for folder in os.listdir(finalDataset):
    classFolder = os.path.join(finalDataset, folder)
    if not os.path.isdir(classFolder):
        continue

    images = os.listdir(classFolder)
    imagesCount = len(images)
    neededImagesCount = 500 - imagesCount

    if neededImagesCount <= 0:
        continue

    # If the needed images count don't exceed the images count in the folder then augment random unrepeated images
    if neededImagesCount <= imagesCount:
        chosenImages = random.sample(images, k = neededImagesCount)
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
            chosenImages = random.sample(images, k = remainder)
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
