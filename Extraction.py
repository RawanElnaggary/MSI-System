import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.eval()

featureExtractor = nn.Sequential (
    *list(resnet.children())[:-1],
    nn.Flatten()
)

preprocess = transforms.Compose ([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_resnet_features (img_path):
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = featureExtractor(img).numpy().squeeze()
    return features

def load_dataset (path, split):
    X, y = [], []
    classNames = sorted(os.listdir(path))

    for label, className in enumerate(classNames):
        splitFolder = os.path.join(path, className, split)
        if not os.path.isdir(splitFolder):
            continue

        for imgName in os.listdir(splitFolder):
            imgPath = os.path.join(splitFolder, imgName)
            try:
                X.append(extract_resnet_features(imgPath))
                y.append(label)
            except Exception:
                pass

    return np.array(X), np.array(y)

if __name__ == "__main__":
    datasetPath = "FinalDataset"

    XTrain, YTrain = load_dataset(datasetPath, "Train")
    np.save("TrainingXFeatures.npy", XTrain)
    np.save("TrainingYLabels.npy", YTrain)

    XTest, YTest = load_dataset(datasetPath, "Test")
    np.save("TestingXFeatures.npy", XTest)
    np.save("TestingYLabels.npy", YTest)
