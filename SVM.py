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

# Train SVM Model
SVCmodel = SVC (
    kernel = 'rbf',
    C = 10,
    gamma = 'scale',
    probability = True
)
SVCmodel.fit(XTrainScaled, YTrain)

# Rejection Threshold
THRESHOLD = 0.60   # Can be tuned (0.5 → 0.75)

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

# Evaluate
trainPred = predict_with_rejection(SVCmodel, XTrainScaled, THRESHOLD)
testPred  = predict_with_rejection(SVCmodel, XTestScaled, THRESHOLD)

train_acc = accuracy_score(YTrain, trainPred)
test_acc  = accuracy_score(YTest, testPred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")
