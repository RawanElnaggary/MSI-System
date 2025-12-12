import cv2
import numpy as np
import joblib
from Extraction import extract_color_histogram, extract_lbp
from sklearn.preprocessing import StandardScaler

# ============================
# Load Model + Scaler
# ============================
model = joblib.load("SVM_Model.pkl")
scaler = joblib.load("scaler.pkl")   # MAKE SURE you saved the scaler (see note below)

# ============================
# Class Names
# ============================
CLASSES = ["Glass", "Paper", "Cardboard", "Plastic", "Metal", "Trash", "Unknown"]

# ============================
# Feature Extraction (Same as Training)
# ============================
def extract_features_live(image):
    image = cv2.resize(image, (128, 128))
    
    colorFeatures = extract_color_histogram(image)
    lbpFeatures   = extract_lbp(image)

    features = np.concatenate([colorFeatures, lbpFeatures])
    return features.reshape(1, -1)

# ============================
# Open the Camera
# ============================
cap = cv2.VideoCapture(0)

print("Camera running... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera Error!")
        break

    # Extract features from current frame
    features = extract_features_live(frame)

    # Scale like training
    features = scaler.transform(features)

    # Predict class
    pred = model.predict(features)[0]
    label = CLASSES[pred]

    # Draw label on frame
    cv2.putText(
        frame, f"Prediction: {label}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2
    )

    cv2.imshow("Material Classification - Live", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
