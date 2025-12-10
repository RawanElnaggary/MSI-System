# An Automated Material Stream Identification (MSI) System

## Overview
This system classifies recyclable materials (Cardboard, Glass, Metal, Paper, Plastic, and Trash) using machine learning and image processing techniques.

## Project Structure
```
MSI-System/
├── InitialDataset/     # Raw, unprocessed images
├── FinalDataset/       # Processed and augmented images
├── Agumentation.py     # Data augmentation script
├── Extraction.py       # Feature extraction script
├── KNN.py             # K-Nearest Neighbors classifier
├── SVM.py             # Support Vector Machine classifier
├── Main.py            # Main execution script
└── Documentation/     # Detailed explanations
```

## Documentation

### 📚 Complete Guides Available:

1. **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete overview of the entire pipeline
   - How Agumentation.py and Extraction.py work together
   - Data flow and processing pipeline
   - Technical concepts explained

2. **[AGUMENTATION_EXPLAINED.md](AGUMENTATION_EXPLAINED.md)** - Detailed guide to Agumentation.py
   - Image validation and cleaning
   - Data augmentation techniques
   - Balancing dataset to 500 images per class
   - Step-by-step workflow with examples

3. **[EXTRACTION_EXPLAINED.md](EXTRACTION_EXPLAINED.md)** - Detailed guide to Extraction.py
   - Color histogram feature extraction (HSV)
   - Local Binary Pattern (LBP) texture features
   - Feature scaling and normalization
   - Output format (XFeatures.npy, YLabels.npy)

## Quick Start

### 1. Data Augmentation
```bash
python Agumentation.py
```
This will:
- Remove corrupted images from InitialDataset/
- Copy valid images to FinalDataset/
- Augment images to ensure 500 images per class

### 2. Feature Extraction
```bash
python Extraction.py
```
This will:
- Extract color and texture features from FinalDataset/
- Scale features for optimal ML performance
- Save features to XFeatures.npy and labels to YLabels.npy

### 3. Train and Classify
```bash
python Main.py
```

## Material Classes
The system identifies 6 types of materials:
1. **Cardboard** - Corrugated paper material
2. **Glass** - Transparent/translucent containers
3. **Metal** - Aluminum cans, steel containers
4. **Paper** - Newspapers, documents, packaging
5. **Plastic** - Bottles, containers, packaging
6. **Trash** - Non-recyclable waste

## Features
- ✅ Automatic data augmentation and balancing
- ✅ Robust feature extraction (color + texture)
- ✅ Multiple ML classifiers (KNN, SVM)
- ✅ Handles corrupted images gracefully
- ✅ Comprehensive documentation

## Requirements
```
Python 3.x
opencv-python (cv2)
numpy
scikit-learn
scikit-image
keras
Pillow (PIL)
```

## Pipeline Overview
```
Raw Images → Validation → Augmentation → Feature Extraction → ML Training → Classification
(Initial)   (Agumentation.py)            (Extraction.py)      (KNN/SVM)    (Prediction)
```

For detailed explanations of each step, see the documentation files linked above.
