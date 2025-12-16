import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load Features
XTrain = np.load("TrainingXFeatures.npy")
XTest = np.load("TestingXFeatures.npy")

YTrain = np.load("TrainingYLabels.npy")
YTest = np.load("TestingYLabels.npy")

# Scale Features
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled = scaler.transform(XTest)

# Rejection Threshold
THRESHOLD = 0.50

def predict_with_rejection (model, X, threshold):
    probs = model.predict_proba(X)
    maxProb = probs.max(axis = 1)
    preds = model.predict(X)

    finalPreds = []
    for p, conf in zip(preds, maxProb):
        if conf < threshold:
            finalPreds.append(-1)     # Unknown class
        else:
            finalPreds.append(p)
    return np.array(finalPreds)


# Train SVC Model
SVCModel = SVC (
    kernel = 'rbf',
    C = 0.9,
    gamma = 'scale',
    probability = True,
    random_state = 42
)
SVCModel.fit(XTrainScaled, YTrain)

# Predictions with rejection
trainPred = predict_with_rejection(SVCModel, XTrainScaled, THRESHOLD)
testPred  = predict_with_rejection(SVCModel, XTestScaled, THRESHOLD)

trainAcc = accuracy_score(YTrain, trainPred) * 100
testAcc = accuracy_score(YTest, testPred) * 100

print("\nC Value: 0.9")
print(f"Training Accuracy: {trainAcc:.4f}%")
print(f"Testing  Accuracy: {testAcc:.4f}%")
print("Difference:", trainAcc - testAcc)

joblib.dump(SVCModel, "SVCModel.pkl")
joblib.dump(scaler, "SVCScaler.pkl")
