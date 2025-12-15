import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Features
XTrain = np.load("TrainingXFeatures.npy")
XTest = np.load("TestingXFeatures.npy")

YTrain = np.load("TrainingYLabels.npy")
YTest  = np.load("TestingYLabels.npy")

# Scale Features
scaler = StandardScaler()
XTrainScaled = scaler.fit_transform(XTrain)
XTestScaled  = scaler.transform(XTest)

# Rejection Threshold
THRESHOLD = 0.60

def predict_with_rejection (model, X, threshold):
    probs = model.predict_proba(X)
    preds = model.predict(X)

    final_preds = []
    for p, sample_probs in zip(preds, probs):
        max_conf = max(sample_probs)
        if max_conf < threshold:
            final_preds.append(6)   # Unknown class
        else:
            final_preds.append(p)

    return np.array(final_preds)

KValues = [3, 5, 7, 9, 11, 13]

print("\nDistance and Euclidean:")
print("=======================================")
for K in KValues:
    # Train KNN Model
    KNNModel = KNeighborsClassifier (
        n_neighbors = K,
        weights = 'distance',
        metric = 'euclidean',
    )
    KNNModel.fit(XTrainScaled, YTrain)

    # Predictions with rejection
    trainPred = predict_with_rejection(KNNModel, XTrainScaled, THRESHOLD)
    testPred = predict_with_rejection(KNNModel, XTestScaled, THRESHOLD)

    trainAcc = accuracy_score(YTrain, trainPred) * 100
    testAcc = accuracy_score(YTest, testPred) * 100

    print("\nK Value:", K)
    print(f"Training Accuracy: {trainAcc:.4f}%")
    print(f"Testing  Accuracy: {testAcc:.4f}%")
    print("Difference:", trainAcc - testAcc)
    print("\n---------------------------------------")


print("\nDistance and Cosine:")
print("=======================================")
for K in KValues:
    # Train KNN Model
    KNNModel = KNeighborsClassifier (
        n_neighbors = K,
        weights = 'distance',
        metric = 'cosine',
    )
    KNNModel.fit(XTrainScaled, YTrain)

    # Predictions with rejection
    trainPred = predict_with_rejection(KNNModel, XTrainScaled, THRESHOLD)
    testPred = predict_with_rejection(KNNModel, XTestScaled, THRESHOLD)

    trainAcc = accuracy_score(YTrain, trainPred) * 100
    testAcc = accuracy_score(YTest, testPred) * 100

    print("\nK Value:", K)
    print(f"Training Accuracy: {trainAcc:.4f}%")
    print(f"Testing  Accuracy: {testAcc:.4f}%")
    print("Difference:", trainAcc - testAcc)
    print("\n---------------------------------------")


print("\nUniform and Euclidean:")
print("=======================================")
for K in KValues:
    # Train KNN Model
    KNNModel = KNeighborsClassifier (
        n_neighbors = K,
        weights = 'uniform',
        metric = 'euclidean',
    )
    KNNModel.fit(XTrainScaled, YTrain)

    # Predictions with rejection
    trainPred = predict_with_rejection(KNNModel, XTrainScaled, THRESHOLD)
    testPred = predict_with_rejection(KNNModel, XTestScaled, THRESHOLD)

    trainAcc = accuracy_score(YTrain, trainPred) * 100
    testAcc = accuracy_score(YTest, testPred) * 100

    print("\nK Value:", K)
    print(f"Training Accuracy: {trainAcc:.4f}%")
    print(f"Testing  Accuracy: {testAcc:.4f}%")
    print("Difference:", trainAcc - testAcc)
    print("\n---------------------------------------")


print("\nUniform and Cosine:")
print("=======================================")
for K in KValues:
    # Train KNN Model
    KNNModel = KNeighborsClassifier (
        n_neighbors = K,
        weights = 'uniform',
        metric = 'cosine',
    )
    KNNModel.fit(XTrainScaled, YTrain)

    # Predictions with rejection
    trainPred = predict_with_rejection(KNNModel, XTrainScaled, THRESHOLD)
    testPred = predict_with_rejection(KNNModel, XTestScaled, THRESHOLD)

    trainAcc = accuracy_score(YTrain, trainPred) * 100
    testAcc = accuracy_score(YTest, testPred) * 100

    print("\nK Value:", K)
    print(f"Training Accuracy: {trainAcc:.4f}%")
    print(f"Testing  Accuracy: {testAcc:.4f}%")
    print("Difference:", trainAcc - testAcc)
    print("\n---------------------------------------")
