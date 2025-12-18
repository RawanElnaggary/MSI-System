import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# Load the ResNet50 model
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet.eval() # Put the model in evaluation mode (only extracting features)

featureExtractor = nn.Sequential (
    *list(resnet.children())[:-1], # Take feature layers (All layers except the classification layer)
    nn.Flatten() # Convert the output (2D/3D) into 1D feature vector
)

preprocess = transforms.Compose ([
    transforms.Resize((224, 224)), # ResNet50 expects
    transforms.ToTensor(), # Convert the image into PyTorch tensor (multidimensional array for calculations)
    transforms.Normalize (
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # Standardize the color values
])


def extract_resnet_features (img_path):
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0) # Apply preprocessing and add a batch dimension (PyTorch expects)
    with torch.no_grad(): # No gradient calculation
        features = featureExtractor(img).numpy().squeeze() # Convert to numpy array and remove the batch dimension
    return features # 1D vector


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
