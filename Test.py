import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def predict(dataFilePath, bestModelPath):
    """
    Args:
        dataFilePath (str): Path to folder containing images
        bestModelPath (str): Path to trained SVC model (.pkl)

    Returns:
        list: Predictions
    """

 
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )
    resnet.eval()

    featureExtractor = nn.Sequential(
        *list(resnet.children())[:-1],
        nn.Flatten()
    )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

   
    model = joblib.load(bestModelPath)
    scaler = joblib.load("SVCScaler.pkl")

    THRESHOLD = 0.50
    predictions = []

   
    for img_name in sorted(os.listdir(dataFilePath)):
        img_path = os.path.join(dataFilePath, img_name)

        try:
            # Load & preprocess image
            img = Image.open(img_path).convert("RGB")
            img = preprocess(img).unsqueeze(0)

            # Feature extraction
            with torch.no_grad():
                features = featureExtractor(img).numpy().squeeze()

            features = np.array(features).reshape(1, -1)

            # Scale features
            features_scaled = scaler.transform(features)

            # Prediction with rejection
            probs = model.predict_proba(features_scaled)
            max_prob = probs.max(axis=1)[0]
            pred = model.predict(features_scaled)[0]

            if max_prob < THRESHOLD:
                predictions.append(-1)
            else:
                predictions.append(int(pred))

        except Exception:
            predictions.append(-1)

    return predictions

preds = predict("D:\\Ahmed\\FourthYear\\FirstTerm\\Machine_Learning\\Assignment\\dataset\\paper", "SVCModel.pkl")
print(preds)