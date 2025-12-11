import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load Features
X = np.load("XFeatures.npy")
Y = np.load("YLabels.npy")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# Scale Features (IMPORTANT)
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

# Evaluate
# ============================
train_pred = model.predict(X_train_scaled)
test_pred  = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test, test_pred)

print("SVM Classifier Results")
print("====================================")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy : {test_acc:.4f}")


