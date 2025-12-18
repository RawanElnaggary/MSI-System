import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load Features
XTrain = np.load("TrainingXFeatures.npy")
XTest = np.load("TestingXFeatures.npy")

YTrain = np.load("TrainingYLabels.npy")
YTest  = np.load("TestingYLabels.npy")

# Scale Features
scaler = Normalizer(norm='l2')
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled  = scaler.transform(XTest)


# Rejection Threshold
THRESHOLD = 0.40

def predict_with_rejection (model, X, threshold):
    probs = model.predict_proba(X)
    preds = model.predict(X)

    finalPreds = []
    for p, sample_probs in zip(preds, probs):
        max_conf = max(sample_probs)
        if max_conf < threshold:
            finalPreds.append(-1)   # Unknown class
        else:
            finalPreds.append(p)

    return np.array(finalPreds)


# Train KNN Model
KNNModel = KNeighborsClassifier (
    n_neighbors = 9,
    weights = 'uniform',
    metric = 'cosine',
)
KNNModel.fit(XTrainScaled, YTrain)

# Predictions with rejection
trainPred = predict_with_rejection(KNNModel, XTrainScaled, THRESHOLD)
testPred = predict_with_rejection(KNNModel, XTestScaled, THRESHOLD)

trainAcc = accuracy_score(YTrain, trainPred) * 100
testAcc = accuracy_score(YTest, testPred) * 100

print("\nK Value: 9")
print(f"Training Accuracy: {trainAcc:.4f}%")
print(f"Testing  Accuracy: {testAcc:.4f}%")
print("Difference:", trainAcc - testAcc)

joblib.dump(KNNModel, "KNNModel.pkl")
joblib.dump(scaler, "KNNScaler.pkl")
