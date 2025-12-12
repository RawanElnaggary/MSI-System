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


THRESHOLD = 0.60

def predict_with_rejection_knn(model, X, threshold):
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

k_values = [5, 7, 9]

print("\n===== K-NN Evaluation =====\n")

for K in k_values:
    print(f"➡ Testing k = {K}")

    knn = KNeighborsClassifier(
        n_neighbors=K,
        weights='distance'
    )
    knn.fit(XTrainScaled, YTrain)

    # Predictions with rejection
    train_pred = predict_with_rejection_knn(knn, XTrainScaled, THRESHOLD)
    test_pred  = predict_with_rejection_knn(knn, XTestScaled, THRESHOLD)

    # Accuracy
    train_acc = accuracy_score(YTrain, train_pred)
    test_acc  = accuracy_score(YTest, test_pred)

    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Testing  Accuracy: {test_acc:.4f}\n")


