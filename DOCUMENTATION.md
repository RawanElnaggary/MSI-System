# Detailed Documentation for MSI System Files

## Table of Contents
1. [Overview](#overview)
2. [Agumentation.py Detailed Explanation](#agumentationpy-detailed-explanation)
3. [Extraction.py Detailed Explanation](#extractionpy-detailed-explanation)

---

## Overview

This Material Stream Identification (MSI) System is designed to classify different types of recyclable materials (Cardboard, Glass, Metal, Paper, Plastic, and Trash) using machine learning. The system consists of two main preprocessing steps:

1. **Agumentation.py**: Prepares and augments the image dataset
2. **Extraction.py**: Extracts features from images for machine learning models

---

## Agumentation.py Detailed Explanation

### Purpose
This script performs **data augmentation** on an image dataset to ensure each class has a balanced number of training samples (500 images per class). Data augmentation is a technique used to artificially expand the size of a training dataset by creating modified versions of existing images.

### Workflow Overview

```
InitialDataset/ → [Validation] → FinalDataset/ → [Augmentation] → Balanced Dataset (500 images/class)
```

### Detailed Code Breakdown

#### 1. Import Libraries (Lines 1-6)
```python
import os              # File and directory operations
import shutil          # High-level file operations (copy, delete)
from PIL import Image  # Image verification and validation
from keras_preprocessing.image import ImageDataGenerator  # Augmentation engine
from keras_preprocessing.image import img_to_array, load_img  # Image utilities
import random          # Random selection for augmentation
```

#### 2. Initialize Dataset Paths (Lines 8-9)
```python
initialDataset = "InitialDataset"  # Source folder with original images
finalDataset = "FinalDataset"      # Destination folder for processed images
```

#### 3. Clean and Recreate Final Dataset Folder (Lines 11-14)
```python
if os.path.exists(finalDataset):
    shutil.rmtree(finalDataset)  # Delete existing folder to start fresh
os.mkdir(finalDataset)            # Create new empty folder
```

**Why?** Ensures a clean slate for each run, preventing duplicate or stale data.

#### 4. Copy Valid Images from Initial to Final Dataset (Lines 16-35)

```python
for folder in os.listdir(initialDataset):
    # Iterate through each class folder (Cardboard, Glass, Metal, Paper, Plastic, Trash)
```

**Image Validation Process:**
- **Line 28-29**: `img.verify()` - Checks if file is a valid image without loading full data
- **Line 30-31**: `img.load()` - Actually loads the image to ensure it's not corrupted
- **Line 33**: If both checks pass, copy the image to FinalDataset
- **Line 34-35**: If any exception occurs (corrupted image), skip it

**Purpose**: Filters out corrupted or invalid images that could cause errors during training.

#### 5. Configure Image Augmentation Parameters (Lines 37-47)

```python
imgAug = ImageDataGenerator (
    rotation_range = 15,              # Rotate images randomly by ±15 degrees
    width_shift_range = 0.15,         # Shift images horizontally by ±15%
    height_shift_range = 0.15,        # Shift images vertically by ±15%
    zoom_range = 0.15,                # Zoom in/out by ±15%
    horizontal_flip = True,           # Randomly flip images horizontally
    brightness_range = [0.7, 1.3],    # Adjust brightness (70% to 130%)
    shear_range = 0.05,               # Apply shearing transformation (±5%)
    fill_mode = "nearest"             # Fill empty pixels with nearest pixel value
)
```

**Why These Parameters?**
- **Rotation**: Handles objects at different angles
- **Shifts**: Simulates objects at different positions
- **Zoom**: Handles objects at different distances
- **Flip**: Increases variety, objects can appear from different sides
- **Brightness**: Handles different lighting conditions
- **Shear**: Simulates perspective changes
- **Fill Mode**: Handles empty pixels created by transformations

#### 6. Apply Augmentation to Reach 500 Images Per Class (Lines 49-111)

**Strategy Overview:**
```
If current images < 500:
    needed = 500 - current
    
    Case 1: needed ≤ current images
        → Augment random selection of images (once each)
    
    Case 2: needed > current images
        → Augment ALL images multiple times
        → Handle remainder with random selection
```

**Case 1: Needed Images ≤ Current Images (Lines 63-76)**
```python
if neededImagesCount <= imagesCount:
    chosenImages = random.sample(images, k = neededImagesCount)
```
- Randomly select `neededImagesCount` images (without repetition)
- Augment each selected image once
- Example: Have 450 images, need 50 more → randomly pick 50 and augment them

**Case 2: Needed Images > Current Images (Lines 80-111)**
```python
augTimes = neededImagesCount // imagesCount  # How many times to augment ALL images
remainder = neededImagesCount % imagesCount   # Extra images needed
```

Example calculation:
- Have 200 images, need 300 more
- `augTimes = 300 // 200 = 1` (augment all 200 images once = 200 new images)
- `remainder = 300 % 200 = 100` (still need 100 more)
- Randomly select 100 images and augment them once more

**Lines 84-96**: Augment all images `augTimes` times
**Lines 98-111**: Handle remainder by randomly selecting and augmenting additional images

#### 7. Augmentation Flow (Lines 71-76, 91-96, 106-111)
```python
for batch in imgAug.flow(x,
                         batch_size = 1,
                         save_to_dir = classFolder,
                         save_prefix = "aug",
                         save_format = "jpg"):
    break  # Generate only one augmented image
```

- `imgAug.flow()`: Generates augmented images on-the-fly
- `batch_size = 1`: Generate one image at a time
- `save_to_dir`: Save augmented images directly to class folder
- `save_prefix`: Prefix for augmented image filenames (e.g., "aug_12345.jpg")
- `break`: Exit after generating one image (flow is a generator that could produce infinite images)

### Summary of Agumentation.py

**Input**: `InitialDataset/` folder with varying numbers of images per class
**Output**: `FinalDataset/` folder with exactly 500 valid images per class
**Key Benefits**:
- Removes corrupted images
- Balances dataset (equal samples per class)
- Increases dataset diversity through augmentation
- Prevents overfitting by providing varied training examples

---

## Extraction.py Detailed Explanation

### Purpose
This script **extracts features** from images to create numerical representations that machine learning models can process. It combines two complementary feature extraction techniques: **Color Histograms** and **Local Binary Patterns (LBP)**.

### Workflow Overview

```
FinalDataset/ → [Load Images] → [Resize] → [Extract Features] → [Normalize] → XFeatures.npy, YLabels.npy
                                              ↓
                                    [Color Histogram + LBP]
```

### Detailed Code Breakdown

#### 1. Import Libraries (Lines 1-5)
```python
import os                                    # File operations
import cv2                                   # OpenCV for image processing
import numpy as np                           # Numerical operations
from skimage.feature import local_binary_pattern  # LBP feature extraction
from sklearn.preprocessing import StandardScaler   # Feature normalization
```

#### 2. Extract Color Histogram Features (Lines 7-17)

```python
def extract_color_histogram(image):
```

**What is a Color Histogram?**
A color histogram represents the distribution of colors in an image. It counts how many pixels have each color value.

**Step-by-Step Process:**

**Line 9**: Convert BGR to HSV
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```
- OpenCV uses BGR (Blue-Green-Red) by default
- HSV (Hue-Saturation-Value) is better for color analysis because:
  - **Hue**: The actual color (0-180° in OpenCV)
  - **Saturation**: Color intensity (0-256)
  - **Value**: Brightness (0-256)
- HSV separates color information from lighting, making it more robust

**Lines 11-13**: Calculate 3D Histogram
```python
histogram = cv2.calcHist([hsv], [0, 1, 2], None,
                         [8, 8, 8],
                         [0, 180, 0, 256, 0, 256])
```
- `[hsv]`: Input image
- `[0, 1, 2]`: Use all three channels (H, S, V)
- `[8, 8, 8]`: Number of bins for each channel
  - Hue: divided into 8 bins (0-180 → 8 ranges)
  - Saturation: divided into 8 bins (0-256 → 8 ranges)
  - Value: divided into 8 bins (0-256 → 8 ranges)
  - Total: 8 × 8 × 8 = **512 features**
- `[0, 180, 0, 256, 0, 256]`: Value ranges for H, S, V

**Why 8 bins?** Balances detail vs. computational efficiency. More bins = more detail but more data.

**Line 16**: Normalize and Flatten
```python
histogram = cv2.normalize(histogram, histogram).flatten()
```
- **Normalize**: Scale values to [0, 1] range (makes images of different sizes comparable)
- **Flatten**: Convert 3D histogram (8×8×8) to 1D vector (512 values)

**Output**: 512-dimensional vector representing color distribution

#### 3. Extract Local Binary Pattern (LBP) Features (Lines 20-36)

```python
def extract_lbp(image):
```

**What is LBP?**
Local Binary Pattern is a texture descriptor that captures local patterns in an image by comparing each pixel with its neighbors.

**How LBP Works:**
1. For each pixel, compare it with surrounding pixels (8 neighbors in a circle)
2. If neighbor > center pixel: mark as 1, else: mark as 0
3. This creates a binary pattern (e.g., 11010001)
4. Convert binary to decimal (e.g., 11010001 → 209)
5. Create histogram of these patterns across the entire image

**Step-by-Step Process:**

**Line 22**: Convert to Grayscale
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
- LBP works on intensity patterns, not color
- Grayscale simplifies processing and focuses on texture

**Line 24**: Calculate LBP
```python
LBP = local_binary_pattern(gray, 24, 3, method="uniform")
```
- `24`: Number of neighboring points to consider
- `3`: Radius of the circle (in pixels)
- `method="uniform"`: Only consider "uniform" patterns
  - Uniform patterns have at most 2 transitions from 0→1 or 1→0
  - Example: 00001111 is uniform (1 transition), 01010101 is not (4 transitions)
  - Uniform patterns capture fundamental texture properties
  - Reduces features from 2^24 to 26 patterns + 1 non-uniform class = 27 total

**Lines 26-30**: Create Histogram
```python
histogram, _ = np.histogram(
    LBP.ravel(),           # Flatten LBP image to 1D
    bins=np.arange(0, 27),  # 26 bins (0-25) for uniform patterns
    range=(0, 26)          # Value range
)
```

**Lines 33-35**: Normalize
```python
histogram = histogram.astype("float")
histogram /= histogram.sum()  # Convert to probability distribution (sums to 1)
```

**Output**: 26-dimensional vector representing texture patterns

#### 4. Combine Features (Lines 39-51)

```python
def extract_features(path):
```

**Line 40-43**: Load and validate image
```python
image = cv2.imread(path)
if image is None:
    print(f"Warning: Could not read image {path}")
    return None
```

**Line 45**: Resize to standard size
```python
image = cv2.resize(image, (128, 128))
```
- Ensures all images have the same dimensions
- 128×128 is a good balance between detail and processing speed
- Consistent size required for feature extraction

**Lines 47-50**: Extract and combine features
```python
colorFeatures = extract_color_histogram(image)  # 512 features
lbpFeatures = extract_lbp(image)                # 26 features
features = np.concatenate([colorFeatures, lbpFeatures])  # 538 features total
```

**Why combine both?**
- **Color Histogram**: Captures "what colors are present" (good for materials with distinct colors)
- **LBP**: Captures "what textures are present" (good for materials with distinct patterns)
- Together: More robust classification (e.g., distinguishing metal from glass)

**Output**: 538-dimensional feature vector per image

#### 5. Load Entire Dataset (Lines 54-70)

```python
def load_dataset(datasetPath):
```

**Lines 57-58**: Create label mapping
```python
classes = sorted(os.listdir(datasetPath))  # ['Cardboard', 'Glass', 'Metal', ...]
label_map = {c: idx for idx, c in enumerate(classes)}  # {'Cardboard': 0, 'Glass': 1, ...}
```
- Converts class names to numeric labels (ML models need numbers, not strings)
- Sorted to ensure consistent ordering across runs

**Lines 60-68**: Process all images
```python
for c in classes:
    folder = os.path.join(dataset_path, c)
    for file in os.listdir(folder):
        feature = extract_features(os.path.join(folder, file))
        if feature is not None:
            x.append(feature)      # Feature vector
            y.append(label_map[c]) # Numeric label
```

**Line 70**: Convert to NumPy arrays
```python
return np.array(x), np.array(y)
```
- X shape: (N, 538) where N = total number of images
- Y shape: (N,) containing class labels 0-5

#### 6. Main Execution (Lines 72-89)

```python
if __name__ == "__main__":
```
This block runs only when script is executed directly (not imported).

**Line 73-74**: Load dataset
```python
dataset_path = "FinalDataset"
X, Y = load_dataset(dataset_path)
```

**Lines 80-81**: Feature Scaling
```python
scaler = StandardScaler()
XScaled = scaler.fit_transform(X)
```

**What is StandardScaler?**
Standardizes features by removing the mean and scaling to unit variance:
```
scaled_value = (value - mean) / standard_deviation
```

**Why scale?**
- Different features have different ranges (color: 0-1, LBP: 0-1)
- Scaling ensures all features contribute equally to ML models
- Many ML algorithms perform better with normalized features
- Results in mean=0, standard deviation=1 for each feature

**Lines 86-87**: Save processed data
```python
np.save("XFeatures.npy", XScaled)  # Scaled feature matrix
np.save("YLabels.npy", Y)          # Labels
```

**Why .npy format?**
- NumPy's native binary format
- Fast to load and save
- Preserves exact numerical values
- Compact file size

### Summary of Extraction.py

**Input**: `FinalDataset/` with balanced image dataset
**Output**: 
- `XFeatures.npy`: (N, 538) array of scaled feature vectors
- `YLabels.npy`: (N,) array of class labels

**Feature Vector Composition** (538 dimensions total):
- 512 dimensions: Color histogram (HSV, 8×8×8 bins)
- 26 dimensions: Local Binary Patterns (texture)

**Key Benefits**:
- Converts images to numerical features ML models can process
- Combines color and texture information for robust classification
- Standardizes features for optimal model performance
- Produces reusable features (train different models without re-extracting)

---

## How These Files Work Together

### Complete Pipeline:

```
1. Agumentation.py:
   InitialDataset (unbalanced) 
   → Remove corrupted images 
   → Augment to 500 images/class 
   → FinalDataset (balanced)

2. Extraction.py:
   FinalDataset 
   → Extract color + texture features 
   → Scale features 
   → XFeatures.npy + YLabels.npy

3. Machine Learning (KNN.py, SVM.py):
   Load XFeatures.npy + YLabels.npy 
   → Train classifier 
   → Predict material type
```

### Data Flow:

```
Raw Images → Validated & Augmented Images → Feature Vectors → ML Model → Classification
  (Initial)        (Agumentation.py)         (Extraction.py)    (KNN/SVM)     (Output)
```

---

## Technical Concepts Explained

### Why Data Augmentation?
- **Problem**: Limited training data can cause overfitting
- **Solution**: Create variations of existing images
- **Result**: Model learns to recognize materials from different angles, lighting, and positions

### Why Feature Extraction?
- **Problem**: Raw images are high-dimensional (128×128×3 = 49,152 values)
- **Solution**: Extract meaningful patterns (color + texture = 538 values)
- **Result**: Faster training, better generalization, less memory

### Why Both Color and Texture?
- **Example**: Glass and Plastic might have similar colors but different textures
- **Example**: Metal and Paper might have similar textures but different colors
- **Combination**: More discriminative features for accurate classification

---

## Expected Results

After running both scripts:

1. **FinalDataset/** contains exactly **3,000 images** (6 classes × 500 images)
2. **XFeatures.npy**: 3,000 × 538 matrix of scaled features
3. **YLabels.npy**: 3,000 labels (0=Cardboard, 1=Glass, 2=Metal, 3=Paper, 4=Plastic, 5=Trash)

These files are ready for training machine learning classifiers (KNN, SVM) to identify material types automatically.

---

## Common Questions

**Q: Why 500 images per class?**
A: Balances dataset to prevent bias toward classes with more samples. Ensures fair learning.

**Q: Why HSV instead of RGB for color?**
A: HSV separates color from brightness, making it more robust to lighting changes.

**Q: What if I have more than 500 images in a class?**
A: Current script doesn't reduce; modify line 59 to randomly sample if `imagesCount > 500`.

**Q: Can I change augmentation parameters?**
A: Yes! Adjust lines 38-46. Increase ranges for more variety, decrease for subtle changes.

**Q: Why resize to 128×128?**
A: Balances detail preservation and computational efficiency. Can be adjusted in Extraction.py line 45.

**Q: Can I use different feature extraction methods?**
A: Yes! Could add HOG, SIFT, or deep learning features. Concatenate with existing features.

---

## Potential Improvements

1. **Agumentation.py**:
   - Add vertical flip augmentation
   - Include random cropping
   - Add Gaussian noise for robustness
   - Implement class weight balancing option

2. **Extraction.py**:
   - Add edge detection features (Canny, Sobel)
   - Include shape descriptors (moments, contours)
   - Try deep learning features (CNN embeddings)
   - Add parallel processing for faster feature extraction

3. **General**:
   - Add data validation checks
   - Implement logging for debugging
   - Create configuration files for parameters
   - Add progress bars for long operations

---

**End of Documentation**
