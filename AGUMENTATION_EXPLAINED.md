# Agumentation.py - Detailed Explanation

## Overview
This script performs **data augmentation** on an image dataset to ensure balanced training data for a Material Stream Identification (MSI) system. It processes raw images, removes corrupted files, and generates augmented versions to reach exactly 500 images per material class.

---

## Purpose
- **Validate Images**: Remove corrupted or unreadable images
- **Balance Dataset**: Ensure equal representation (500 images per class)
- **Augment Data**: Create diverse variations to improve model generalization

---

## Material Classes
The system works with 6 material types:
1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

---

## Complete Workflow

```
┌─────────────────┐
│ InitialDataset/ │ (Unbalanced, may have corrupted images)
│  ├─ Cardboard/  │
│  ├─ Glass/      │
│  ├─ Metal/      │
│  ├─ Paper/      │
│  ├─ Plastic/    │
│  └─ Trash/      │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│ Image Validation   │ Check each image with PIL
│ - Verify format    │ Remove corrupted files
│ - Check integrity  │
└────────┬───────────┘
         │
         ▼
┌─────────────────┐
│ FinalDataset/   │ (Clean images only)
│  ├─ Cardboard/  │
│  ├─ Glass/      │
│  ├─ Metal/      │
│  ├─ Paper/      │
│  ├─ Plastic/    │
│  └─ Trash/      │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ Count Images/Class   │
│ Calculate needed:    │
│ 500 - current_count  │
└────────┬─────────────┘
         │
         ▼
    ┌────┴────┐
    │ Needed? │
    └─┬────┬──┘
      │    │
  NO  │    │ YES
      │    │
      ▼    ▼
   [Skip] ┌──────────────────────┐
          │ Apply Augmentation   │
          │ - Rotation           │
          │ - Shift              │
          │ - Zoom               │
          │ - Flip               │
          │ - Brightness change  │
          └──────────┬───────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ FinalDataset/       │
          │ (500 images/class)  │
          │ Total: 3000 images  │
          └─────────────────────┘
```

---

## Code Breakdown by Section

### Section 1: Setup (Lines 1-14)

```python
import os
import shutil
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img
import random

initialDataset = "InitialDataset"
finalDataset = "FinalDataset"

# Clean slate - remove old final dataset
if os.path.exists(finalDataset):
    shutil.rmtree(finalDataset)
os.mkdir(finalDataset)
```

**What it does:**
- Imports required libraries
- Sets source and destination folder paths
- Deletes existing `FinalDataset` folder (if exists)
- Creates fresh empty `FinalDataset` folder

**Why delete and recreate?**
Ensures clean state on each run. Prevents mixing old and new data, avoiding duplicates or stale augmented images.

---

### Section 2: Image Validation & Copying (Lines 16-35)

```python
for folder in os.listdir(initialDataset):
    initialFolder = os.path.join(initialDataset, folder)
    if not os.path.isdir(initialFolder):
        continue

    finalFolder = os.path.join(finalDataset, folder)
    os.makedirs(finalFolder, exist_ok=True)

    for file in os.listdir(initialFolder):
        initialFile = os.path.join(initialFolder, file)
        try:
            with Image.open(initialFile) as img:
                img.verify()  # Check file is valid image
            with Image.open(initialFile) as img:
                img.load()    # Ensure image can be loaded
            shutil.copy(initialFile, finalFolder)  # Copy if valid
        except Exception:
            continue  # Skip corrupted images
```

**Step-by-Step Process:**

1. **Loop through each class folder** (Cardboard, Glass, Metal, etc.)
2. **Create corresponding folder** in FinalDataset
3. **For each image file:**
   - **First check**: `img.verify()` - Quick validation without loading full image
   - **Second check**: `img.load()` - Actually load image data to ensure it's readable
   - **If both pass**: Copy image to FinalDataset
   - **If either fails**: Skip (corrupted/invalid image)

**Why two checks?**
- `verify()`: Fast, catches format errors
- `load()`: Catches data corruption that verify() might miss
- Together: Robust validation

**What happens to corrupted images?**
They're silently skipped (not copied to FinalDataset). This prevents errors during later processing.

---

### Section 3: Augmentation Configuration (Lines 37-47)

```python
imgAug = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.15,
    horizontal_flip = True,
    brightness_range = [0.7, 1.3],
    shear_range = 0.05,
    fill_mode = "nearest"
)
```

**Augmentation Techniques Explained:**

| Parameter | Range | Effect | Example |
|-----------|-------|--------|---------|
| `rotation_range` | ±15° | Rotates image randomly | Object tilted left/right |
| `width_shift_range` | ±15% | Shifts image horizontally | Object moved left/right |
| `height_shift_range` | ±15% | Shifts image vertically | Object moved up/down |
| `zoom_range` | ±15% | Zooms in or out | Object closer/farther |
| `horizontal_flip` | True/False | Mirrors image horizontally | Left becomes right |
| `brightness_range` | 0.7-1.3 | Adjusts brightness | Darker (70%) to brighter (130%) |
| `shear_range` | ±5% | Skews image | Parallelogram effect |
| `fill_mode` | "nearest" | Fills empty pixels | Uses nearby pixel colors |

**Visual Examples:**

```
Original:        Rotated (15°):    Shifted:         Flipped:
┌─────────┐      ┌─────────┐       ┌─────────┐      ┌─────────┐
│  ▲      │      │    ▲    │       │         │      │      ▲  │
│ ╱ ╲     │      │   ╱ ╲   │       │    ▲    │      │     ╱ ╲ │
│╱   ╲    │      │  ╱   ╲  │       │   ╱ ╲   │      │    ╱   ╲│
└─────────┘      └─────────┘       └─────────┘      └─────────┘

Zoomed In:       Brightness ↓:     Brightness ↑:    Sheared:
┌─────────┐      ┌─────────┐       ┌─────────┐      ┌─────────┐
│  ▓▓▓    │      │  ░░░    │       │  ███    │      │   ▲     │
│  ▓▓▓    │      │ ░░░░░   │       │ █████   │      │  ╱ ╲╲   │
│  ▓▓▓    │      │░░░░░░   │       │█████    │      │ ╱   ╲╲  │
└─────────┘      └─────────┘       └─────────┘      └─────────┘
```

**Why These Specific Values?**

- **Moderate changes** (15% range): Realistic variations, not too extreme
- **Horizontal flip only**: Vertical flip would make materials look unnatural (upside down)
- **Brightness range**: Simulates different lighting conditions
- **Shear kept small** (5%): Too much shearing distorts materials unrealistically

---

### Section 4: Augmentation Strategy (Lines 49-111)

#### Decision Tree:

```
For each class folder:
    current_count = number of images
    needed = 500 - current_count
    
    IF needed ≤ 0:
        → Skip (already have 500+ images)
    
    ELSE IF needed ≤ current_count:
        → CASE 1: Random selection strategy
    
    ELSE:
        → CASE 2: Full augmentation + remainder strategy
```

---

#### **CASE 1: Needed ≤ Current Images** (Lines 63-76)

```python
if neededImagesCount <= imagesCount:
    chosenImages = random.sample(images, k = neededImagesCount)
    for imgName in chosenImages:
        imgPath = os.path.join(classFolder, imgName)
        img = load_img(imgPath)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        for batch in imgAug.flow(x,
                                  batch_size = 1,
                                  save_to_dir = classFolder,
                                  save_prefix="aug",
                                  save_format="jpg"):
            break
```

**Example Scenario:**
- Current: 450 images
- Needed: 50 images
- Strategy: Randomly pick 50 images (no repeats) and augment each once

**Step-by-Step:**
1. `random.sample()`: Pick 50 different images randomly
2. For each selected image:
   - Load image
   - Convert to array format
   - Reshape for Keras (add batch dimension)
   - Generate ONE augmented version
   - Save to class folder with "aug" prefix
3. Result: 450 original + 50 augmented = 500 total

**Why random selection?**
Ensures diversity. Augmenting the same images repeatedly would reduce variety.

---

#### **CASE 2: Needed > Current Images** (Lines 80-111)

```python
else:
    augTimes = neededImagesCount // imagesCount
    remainder = neededImagesCount % imagesCount
    
    # Augment ALL images multiple times
    for imgName in images:
        imgPath = os.path.join(classFolder, imgName)
        img = load_img(imgPath)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        for _ in range(augTimes):
            for batch in imgAug.flow(x,
                                     batch_size = 1,
                                     save_to_dir = classFolder,
                                     save_prefix = f"{imgName}_aug",
                                     save_format = "jpg"):
                break
    
    # Handle remainder with random selection
    if remainder > 0:
        chosenImages = random.sample(images, k = remainder)
        for imgName in chosenImages:
            # [augmentation code similar to above]
```

**Example Scenario:**
- Current: 150 images
- Needed: 350 images
- Calculation:
  - `augTimes = 350 // 150 = 2` (augment all 150 images twice)
  - `remainder = 350 % 150 = 50` (still need 50 more)

**Strategy:**
1. **Augment ALL 150 images twice**:
   - First round: 150 augmented images
   - Second round: 150 augmented images
   - Total so far: 150 original + 300 augmented = 450 images

2. **Handle remainder (50 images needed)**:
   - Randomly select 50 of the 150 original images
   - Augment each once
   - Total: 450 + 50 = 500 images ✓

**Why this strategy?**
- Maximizes use of all available images
- Ensures even distribution of augmentation
- Handles any count scenario efficiently

---

### Augmentation Flow Details

```python
for batch in imgAug.flow(x,
                         batch_size = 1,
                         save_to_dir = classFolder,
                         save_prefix = "aug",
                         save_format = "jpg"):
    break
```

**Parameters Explained:**

- `x`: Input image as NumPy array with shape (1, height, width, 3)
- `batch_size = 1`: Process one image at a time
- `save_to_dir`: Automatically save augmented image to this folder
- `save_prefix`: Prepend to filename (e.g., "aug_12345.jpg" or "metal_001_aug_67890.jpg")
- `save_format = "jpg"`: Output format

**Why the `break`?**

`imgAug.flow()` is a **generator** that can produce infinite variations. The `break` statement exits after generating exactly one augmented image.

Without `break`:
```python
for batch in imgAug.flow(x, ...):
    pass  # Would generate infinite images! ❌
```

With `break`:
```python
for batch in imgAug.flow(x, ...):
    break  # Generate exactly 1 image ✓
```

---

## Example Execution Walkthrough

### Initial State:
```
InitialDataset/
├─ Cardboard/  (403 images, 2 corrupted)
├─ Glass/      (521 images, 1 corrupted)
├─ Metal/      (189 images, 0 corrupted)
├─ Paper/      (478 images, 3 corrupted)
├─ Plastic/    (512 images, 0 corrupted)
└─ Trash/      (145 images, 1 corrupted)
```

### After Validation (Copying to FinalDataset):
```
FinalDataset/
├─ Cardboard/  (401 valid images)  → Need 99 more
├─ Glass/      (520 valid images)  → Already has 500+, keep as is
├─ Metal/      (189 valid images)  → Need 311 more
├─ Paper/      (475 valid images)  → Need 25 more
├─ Plastic/    (512 valid images)  → Already has 500+, keep as is
└─ Trash/      (144 valid images)  → Need 356 more
```

### Augmentation Strategy per Class:

**Cardboard (401 → 500):**
- Need: 99 images
- Strategy: CASE 1 (99 ≤ 401)
- Action: Randomly select 99 images, augment each once
- Result: 401 + 99 = 500 ✓

**Glass (520 → 520):**
- Need: 0 images
- Strategy: Skip
- Action: None
- Result: 520 images (no augmentation needed)

**Metal (189 → 500):**
- Need: 311 images
- Strategy: CASE 2 (311 > 189)
- Calculation:
  - augTimes = 311 // 189 = 1 (augment all 189 once)
  - remainder = 311 % 189 = 122 (need 122 more)
- Action:
  - Augment all 189 images once → 189 new images
  - Randomly select 122 images, augment each once → 122 new images
- Result: 189 + 189 + 122 = 500 ✓

**Paper (475 → 500):**
- Need: 25 images
- Strategy: CASE 1 (25 ≤ 475)
- Action: Randomly select 25 images, augment each once
- Result: 475 + 25 = 500 ✓

**Plastic (512 → 512):**
- Need: 0 images
- Strategy: Skip
- Action: None
- Result: 512 images

**Trash (144 → 500):**
- Need: 356 images
- Strategy: CASE 2 (356 > 144)
- Calculation:
  - augTimes = 356 // 144 = 2 (augment all 144 twice)
  - remainder = 356 % 144 = 68 (need 68 more)
- Action:
  - Augment all 144 images twice → 288 new images
  - Randomly select 68 images, augment each once → 68 new images
- Result: 144 + 288 + 68 = 500 ✓

### Final State:
```
FinalDataset/
├─ Cardboard/  (500 images)
├─ Glass/      (520 images) ← Could implement max cap
├─ Metal/      (500 images)
├─ Paper/      (500 images)
├─ Plastic/    (512 images) ← Could implement max cap
└─ Trash/      (500 images)

Total: ~3,032 images
```

---

## Key Benefits

### 1. **Balanced Dataset**
- Equal representation for each class
- Prevents model bias toward over-represented classes
- Fair learning opportunity for all material types

### 2. **Data Quality**
- Removes corrupted images automatically
- Only valid, loadable images in final dataset
- Prevents training errors

### 3. **Increased Diversity**
- Multiple variations of same object
- Different angles, positions, lighting
- Model learns robust features, not memorization

### 4. **Prevents Overfitting**
- More training examples
- Model sees variations, not just originals
- Better generalization to new images

### 5. **Efficient Strategy**
- Smart augmentation: only augment what's needed
- Doesn't waste time on classes with enough images
- Balanced approach: distribute augmentation evenly

---

## Common Issues & Solutions

### Issue 1: "PermissionError: Cannot delete FinalDataset"
**Cause**: Folder is open in file explorer or used by another process
**Solution**: Close all programs accessing the folder, then rerun

### Issue 2: Some images not copied
**Cause**: Corrupted or invalid image files
**Solution**: This is expected behavior. Script filters them out automatically

### Issue 3: Augmentation takes too long
**Cause**: Large number of images to augment
**Solutions**:
- Reduce augmentation parameters (smaller ranges)
- Use smaller initial dataset
- Optimize code with multiprocessing

### Issue 4: Augmented images look too similar
**Cause**: Augmentation parameters too small
**Solution**: Increase ranges (e.g., rotation_range = 30 instead of 15)

### Issue 5: Augmented images look unrealistic
**Cause**: Augmentation parameters too large
**Solution**: Decrease ranges to more conservative values

---

## Customization Options

### Change Target Images Per Class
```python
# Line 57: Change 500 to desired number
neededImagesCount = 500 - imagesCount  # Change 500 here
```

### Add More Augmentation Techniques
```python
imgAug = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.15,
    horizontal_flip = True,
    vertical_flip = False,         # Add vertical flip
    brightness_range = [0.7, 1.3],
    shear_range = 0.05,
    channel_shift_range = 0.1,     # Add color shifting
    fill_mode = "nearest",
    preprocessing_function = None  # Add custom function
)
```

### Implement Maximum Cap for Classes
```python
# After line 56
imagesCount = len(images)
if imagesCount > 500:
    # Randomly select 500 images, delete rest
    keep_images = random.sample(images, k=500)
    for img in images:
        if img not in keep_images:
            os.remove(os.path.join(classFolder, img))
    imagesCount = 500
```

### Save Augmentation Log
```python
# At the beginning
log_file = open("augmentation_log.txt", "w")

# In augmentation loops
log_file.write(f"Class: {folder}, Original: {imagesCount}, Needed: {neededImagesCount}\n")

# At the end
log_file.close()
```

---

## Performance Tips

1. **Use SSD for faster I/O**: Image reading/writing is I/O intensive
2. **Process in batches**: Modify to generate multiple augmented images per loop
3. **Parallelize**: Use multiprocessing to process multiple classes simultaneously
4. **Monitor disk space**: Augmentation can quickly fill disk (3000 images × ~100KB each ≈ 300MB)
5. **Use smaller target count for testing**: Start with 100 images/class to test quickly

---

## Summary

**Agumentation.py** is a critical preprocessing step that:
- ✓ Validates and filters image dataset
- ✓ Balances class distribution (500 images per class)
- ✓ Generates diverse augmented images
- ✓ Prepares data for robust machine learning

**Input**: InitialDataset/ (unbalanced, possibly corrupted)
**Output**: FinalDataset/ (balanced, validated, augmented)

This ensures the machine learning model receives high-quality, balanced training data for optimal performance in material classification.

---

**Next Step**: Run `Extraction.py` to extract features from the balanced dataset.
