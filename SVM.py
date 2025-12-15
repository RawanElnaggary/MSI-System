import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
THRESHOLD = 0.5   # Can be tuned (0.5 → 0.75)

def predict_with_rejection (model, X, threshold):
    probs = model.predict_proba(X)
    maxProb = probs.max(axis = 1)
    preds = model.predict(X)
    finalPreds = []
    for p, conf in zip(preds, maxProb):
        if conf < threshold:
            finalPreds.append(6)     # Unknown class
        else:
            finalPreds.append(p)
    return np.array(finalPreds)

CValues = [ 0.35]#0.259,0.45, 2.3, 0.4, 2.588,2.259] 

for C in CValues:
    # Train SVC Model
    SVCmodel = SVC (
        kernel = 'rbf',
        C = C,
        gamma = 'scale', 
        probability = True
    )
    SVCmodel.fit(XTrainScaled, YTrain)

    # Predictions with rejection
    trainPred = predict_with_rejection(SVCmodel, XTrainScaled, THRESHOLD)
    testPred  = predict_with_rejection(SVCmodel, XTestScaled, THRESHOLD)

    trainAcc = accuracy_score(YTrain, trainPred) * 100
    testAcc = accuracy_score(YTest, testPred) * 100

    print("\nC Value:", C)
    print(f"Training Accuracy: {trainAcc:.4f}%")
    print(f"Testing  Accuracy: {testAcc:.4f}%")
    print("Difference:", trainAcc - testAcc)
    print("\n=======================================")

import joblib

joblib.dump(SVCmodel, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
