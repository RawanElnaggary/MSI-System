import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler

def extract_color_histogram (image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    histogram = cv2.calcHist([hsv], [0, 1, 2], None,
                             [8, 8, 8],
                             [0, 180, 0, 256, 0, 256])

    # Normalize the histogram, convert to 1D vector
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram


def extract_lbp (image):
    # Convert to grayScale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    LBP = local_binary_pattern(gray, 24, 3, method="uniform")

    histogram, _ = np.histogram (
        LBP.ravel(),
        bins=np.arange(0, 27),
        range=(0, 26)
    )

    # Convert histogram values to float
    histogram = histogram.astype("float")
    # Normalize the histogram (converts into a probability distribution)
    histogram /= histogram.sum()
    return histogram


def extract_features (path):
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Could not read image {path}")
        return None

    image = cv2.resize(image, (128, 128))

    colorFeatures = extract_color_histogram(image)
    lbpFeatures = extract_lbp(image)

    features = np.concatenate([colorFeatures, lbpFeatures])
    return features


def load_dataset (datasetPath):
    x, y = [], []

    classes = sorted(os.listdir(datasetPath))
    label_map = {c: idx for idx, c in enumerate(classes)}

    for c in classes:
        folder = os.path.join(dataset_path, c)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
                feature = extract_features(os.path.join(folder, file))
                if feature is not None:
                    x.append(feature)
                    y.append(label_map[c])

    return np.array(x), np.array(y)

if __name__ == "__main__":
    dataset_path = "FinalDataset"
    X, Y = load_dataset(dataset_path)

    # print("Original X shape:", X.shape)
    # print("y shape:", y.shape)

    # Scale features
    scaler = StandardScaler()
    XScaled = scaler.fit_transform(X)

    # print("Scaled X shape:", X_scaled.shape)

    # Save to files
    np.save("XFeatures.npy", XScaled)
    np.save("YLabels.npy", Y)

    # print("Features and labels saved: X_scaled.npy, y.npy")
