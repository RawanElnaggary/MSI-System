import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load Features
X = np.load("XFeatures.npy")
Y = np.load("YLabels.npy")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM Model
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)
model.fit(X_train_scaled, y_train)

# Rejection Threshold
THRESHOLD = 0.60   # you can tune this (0.5 → 0.75)

def predict_with_rejection(model, X, threshold):
    probs = model.predict_proba(X)
    max_prob = probs.max(axis=1)
    preds = model.predict(X)

    final_preds = []
    for p, conf in zip(preds, max_prob):
        if conf < threshold:
            final_preds.append(6)     # Unknown class
        else:
            final_preds.append(p)
    return np.array(final_preds)

# Evaluate
train_pred = predict_with_rejection(model, X_train_scaled, THRESHOLD)
test_pred  = predict_with_rejection(model, X_test_scaled, THRESHOLD)

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test, test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")
