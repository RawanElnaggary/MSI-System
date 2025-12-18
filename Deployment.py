import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image as PILImage
from collections import deque
from time import time
import os
from tkinter import Tk, filedialog

# Import the existing extraction logic
try:
    from Extraction import extract_resnet_features, featureExtractor, preprocess
    EXTRACTION_AVAILABLE = True
    print("[OK] Imported existing feature extraction logic from Extraction.py")
except ImportError:
    from Extraction import extract_resnet_features, featureExtractor, preprocess
    EXTRACTION_AVAILABLE = False
    print("[WARNING] Could not import Extraction.py - will use fallback extraction")


class MSIClassifier:
    """Real-time material classification using pre-trained models"""

    def __init__(self, model_path, scaler_path, model_type):
        """Initialize classifier with model and feature extractor"""

        print(f"\n{'='*60}")
        print(f"Loading {model_type} Model")
        print(f"{'='*60}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_type = model_type
            print(f"[OK] {model_type} model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Use existing feature extractor from Extraction.py
        if EXTRACTION_AVAILABLE:
            print("[OK] Using feature extractor from Extraction.py")
            self.feature_extractor = featureExtractor
            self.preprocess = preprocess
            self.use_existing_extraction = True
        else:
            # Fallback: create own extractor (only if Extraction.py unavailable)
            print("[...] Loading ResNet50 feature extractor (fallback)...")
            try:
                resnet = models.resnet50(
                    weights=models.ResNet50_Weights.IMAGENET1K_V1)
                resnet.eval()

                self.feature_extractor = nn.Sequential(
                    *list(resnet.children())[:-1],
                    nn.Flatten()
                )

                self.preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                print("[OK] Feature extractor loaded (fallback)")
                self.use_existing_extraction = False
            except Exception as e:
                raise RuntimeError(f"Failed to load feature extractor: {e}")

        # CRITICAL: Class names MUST match the order in Extraction.py (sorted alphabetically)
        # This is the EXACT order from sorted(os.listdir(path)) in Extraction.py
        self.class_names = [
            "Cardboard",  # 0
            "Glass",      # 1
            "Metal",      # 2
            "Paper",      # 3
            "Plastic",    # 4
            "Trash",      # 5
            "Unknown"     # 6 (for rejection)
        ]

        self.class_colors = {
            0: (80, 160, 255),   # Cardboard - Blue
            1: (255, 180, 0),    # Glass - Gold
            2: (200, 200, 200),  # Metal - Silver
            3: (240, 240, 240),  # Paper - White
            4: (80, 220, 80),    # Plastic - Green
            5: (60, 60, 240),    # Trash - Red
            6: (140, 140, 140)   # Unknown - Gray
        }

        # Threshold from training files
        if model_type == "SVM":
            self.threshold = 0.50  # From SVM.py
        else:  # KNN
            self.threshold = 0.40  # From KNN.py

        self.prediction_history = deque(maxlen=5)
        self.frame_times = deque(maxlen=30)

        # Cache for pre-extracted features
        self.preloaded_features = None
        self.preloaded_labels = None
        self.image_index_map = {}

        print(f"[OK] Classifier ready (threshold: {self.threshold})")
        print(f"{'='*60}\n")

    def extract_features_from_frame(self, frame):
        """
        Extract ResNet50 features from frame using EXISTING logic

        This method uses the exact same extraction process as Extraction.py
        """
        try:
            # Convert frame to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)

            # Use existing preprocessing and extraction (SAME as Extraction.py)
            img_tensor = self.preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                features = self.feature_extractor(img_tensor).numpy().squeeze()

            return features.reshape(1, -1)
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None

    def extract_features_from_path(self, img_path):
        """
        Extract features from image path using EXISTING extraction function

        This reuses extract_resnet_features() from Extraction.py
        """
        if EXTRACTION_AVAILABLE:
            try:
                # Use the EXACT same function from Extraction.py
                features = extract_resnet_features(img_path)
                return features.reshape(1, -1)
            except Exception as e:
                print(f"[ERROR] Feature extraction from path failed: {e}")
                return None
        else:
            # Fallback: load image and extract
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    return None
                return self.extract_features_from_frame(frame)
            except Exception as e:
                print(f"[ERROR] Fallback extraction failed: {e}")
                return None

    def load_precomputed_features(self, dataset_path):
        """
        Load pre-extracted features from .npy files (created by Extraction.py)

        Looks for:
        - TestingXFeatures.npy (created by Extraction.py)
        - TestingYLabels.npy (optional, for validation)
        """
        features_path = "TestingXFeatures.npy"
        labels_path = "TestingYLabels.npy"

        if not os.path.exists(features_path):
            print(f"[INFO] Pre-extracted features not found: {features_path}")
            print(f"[INFO] Run Extraction.py first to create feature files")
            print(f"[INFO] Falling back to real-time extraction")
            return False

        try:
            print(f"[...] Loading pre-extracted features from {features_path}")
            self.preloaded_features = np.load(features_path)

            if os.path.exists(labels_path):
                self.preloaded_labels = np.load(labels_path)
                print(
                    f"[OK] Also loaded ground truth labels from {labels_path}")

            print(
                f"[OK] Loaded {len(self.preloaded_features)} pre-extracted features")

            # Build index mapping: image path -> feature index
            # This must match the EXACT order used in Extraction.py
            image_files = []
            class_names = sorted(os.listdir(dataset_path))

            feature_idx = 0
            for class_name in class_names:
                test_folder = os.path.join(dataset_path, class_name, "Test")
                if not os.path.isdir(test_folder):
                    continue

                for img_name in sorted(os.listdir(test_folder)):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue

                    img_path = os.path.join(test_folder, img_name)
                    self.image_index_map[img_path] = feature_idx
                    image_files.append(img_path)
                    feature_idx += 1

            if len(image_files) != len(self.preloaded_features):
                print(f"[WARNING] Feature count mismatch!")
                print(f"  Expected: {len(image_files)} images")
                print(f"  Got: {len(self.preloaded_features)} features")
                print(
                    f"[INFO] This may happen if dataset changed after running Extraction.py")
                print(f"[INFO] Falling back to real-time extraction")
                self.preloaded_features = None
                self.image_index_map = {}
                return False

            print(
                f"[OK] Indexed {len(self.image_index_map)} images to pre-computed features")
            print(f"[OK] Ready for FAST classification (no re-extraction needed)")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load pre-extracted features: {e}")
            self.preloaded_features = None
            self.image_index_map = {}
            return False

    def get_features(self, input_data):
        """
        Get features for classification

        Args:
            input_data: Either:
                - str: image file path -> check pre-computed, else extract
                - np.ndarray: frame/image -> extract features
                - int: index into pre-computed features

        Returns:
            features: (1, 2048) numpy array
        """
        # Case 1: Image path (string)
        if isinstance(input_data, str):
            # Try pre-computed features first
            if input_data in self.image_index_map:
                idx = self.image_index_map[input_data]
                return self.preloaded_features[idx].reshape(1, -1)

            # Otherwise extract using existing logic
            return self.extract_features_from_path(input_data)

        # Case 2: Feature index (integer)
        elif isinstance(input_data, int):
            if self.preloaded_features is not None:
                if 0 <= input_data < len(self.preloaded_features):
                    return self.preloaded_features[input_data].reshape(1, -1)
            return None

        # Case 3: Image array (frame from camera)
        elif isinstance(input_data, np.ndarray):
            return self.extract_features_from_frame(input_data)

        else:
            print(f"[ERROR] Invalid input type: {type(input_data)}")
            return None

    def classify(self, input_data):
        """
        Classify using pre-computed features OR real-time extraction
        Uses EXACT same logic as KNN.py and SVM.py

        Args:
            input_data: str (path), np.ndarray (frame), or int (index)
        """
        features = self.get_features(input_data)

        if features is None:
            return 6, 0.0, np.zeros(7)

        # Scale features using the SAME scaler from training
        features_scaled = self.scaler.transform(features)

        # Predict using the SAME logic as KNN.py/SVM.py
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features_scaled)[0]
            pred_class = np.argmax(probs)
            max_conf = probs[pred_class]
            all_probs = np.zeros(7)
            all_probs[:len(probs)] = probs
        else:
            pred_class = self.model.predict(features_scaled)[0]
            max_conf = 1.0
            all_probs = np.zeros(7)
            all_probs[pred_class] = 1.0

        # Apply rejection threshold (SAME as training files)
        if max_conf < self.threshold:
            pred_class = 6  # Unknown class
            all_probs[6] = 1.0 - max_conf
            confidence = all_probs[6]
        else:
            confidence = max_conf

        # Temporal smoothing ONLY for live camera (not dataset testing)
        if isinstance(input_data, np.ndarray):  # Camera frame
            self.prediction_history.append(pred_class)
            if len(self.prediction_history) >= 3:
                stable_pred = max(set(self.prediction_history),
                                  key=self.prediction_history.count)
            else:
                stable_pred = pred_class
            return stable_pred, confidence, all_probs
        else:
            # Dataset testing - no smoothing
            return pred_class, confidence, all_probs

    def process_dataset(self, dataset_path):
        """Process a test dataset using pre-computed features when available"""

        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_path}")
        print(f"{'='*60}\n")

        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset path not found: {dataset_path}")
            return False

        # Clear prediction history for dataset testing
        self.prediction_history.clear()

        # Try to load pre-computed features (uses Extraction.py output)
        using_precomputed = self.load_precomputed_features(dataset_path)

        results = []
        total = 0

        # Collect image files (must match Extraction.py order)
        image_files = []
        ground_truth_labels = []
        class_names = sorted(os.listdir(dataset_path))

        # Map class names to indices (MUST match training order - alphabetical)
        class_name_to_idx = {}
        idx = 0
        for class_name in sorted(["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash", "Unknown"]):
            class_name_to_idx[class_name] = idx
            idx += 1

        print(f"[INFO] Class mapping (must match Extraction.py):")
        for name, idx in sorted(class_name_to_idx.items(), key=lambda x: x[1]):
            print(f"  {idx}: {name}")
        print()

        for class_name in class_names:
            test_folder = os.path.join(dataset_path, class_name, "Test")
            if not os.path.isdir(test_folder):
                continue

            # Get the ground truth label index
            true_label = class_name_to_idx.get(class_name, -1)

            if true_label == -1:
                print(f"[WARNING] Unknown class folder: {class_name}")
                continue

            for img_name in sorted(os.listdir(test_folder)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(test_folder, img_name))
                    ground_truth_labels.append(true_label)

        if len(image_files) == 0:
            print("[ERROR] No images found in dataset")
            return False

        print(f"[INFO] Found {len(image_files)} images")

        if using_precomputed:
            print(f"[INFO] Using pre-extracted features from Extraction.py (FAST)")
        else:
            print(f"[INFO] Extracting features in real-time (SLOW)")
            print(f"[TIP] Run Extraction.py first to speed up processing")

        print(f"[INFO] Processing...\n")

        start_time = time()

        correct_predictions = 0
        total_predictions = 0

        for idx, img_path in enumerate(image_files):
            try:
                # Classify using image path (automatically uses pre-computed if available)
                class_id, confidence, all_probs = self.classify(img_path)

                img_name = os.path.basename(img_path)
                true_label = ground_truth_labels[idx]

                # Count correct predictions
                if true_label != -1:
                    total_predictions += 1
                    if class_id == true_label:
                        correct_predictions += 1

                results.append({
                    'filename': img_name,
                    'predicted_class': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': confidence,
                    'true_label': true_label,
                    'true_class_name': self.class_names[true_label] if true_label != -1 else "Unknown",
                    'correct': (class_id == true_label)
                })

                if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
                    elapsed = time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    print(
                        f"[PROGRESS] {idx+1}/{len(image_files)} images processed ({rate:.1f} img/sec)")

                total += 1

            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        elapsed_total = time() - start_time
        avg_rate = total / elapsed_total if elapsed_total > 0 else 0

        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions *
                    100) if total_predictions > 0 else 0

        print(
            f"\n[OK] Processed {total} images in {elapsed_total:.2f}s ({avg_rate:.1f} img/sec)")
        print(
            f"[INFO] Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")

        # Display results in GUI with accuracy
        self.display_results_gui(results, dataset_path, accuracy)

        return True

    def draw_ui(self, frame, class_id, confidence, all_probs, fps):
        """Draw UI overlay on frame"""

        h, w = frame.shape[:2]
        overlay = frame.copy()

        material = self.class_names[class_id]
        color = self.class_colors[class_id]

        cv2.rectangle(overlay, (30, 30), (400, 160), color, 3)

        card_bg = np.zeros((130, 370, 3), dtype=np.uint8)
        card_bg[:] = (20, 20, 20)
        roi = overlay[30:160, 30:400]
        overlay[30:160, 30:400] = cv2.addWeighted(roi, 0.3, card_bg, 0.7, 0)

        cv2.putText(overlay, material, (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.putText(overlay, f"{confidence*100:.1f}%", (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

        if confidence >= 0.75:
            status = "HIGH CONFIDENCE"
            status_color = (100, 255, 100)
        elif confidence >= 0.55:
            status = "MODERATE"
            status_color = (255, 200, 100)
        else:
            status = "LOW CONFIDENCE"
            status_color = (255, 100, 100)

        cv2.putText(overlay, status, (50, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        px, py = w - 310, 30
        cv2.rectangle(overlay, (px, py), (px + 280, py + 340), (80, 80, 80), 2)

        prob_bg = np.zeros((340, 280, 3), dtype=np.uint8)
        prob_bg[:] = (20, 20, 20)
        prob_roi = overlay[py:py+340, px:px+280]
        overlay[py:py+340, px:px +
                280] = cv2.addWeighted(prob_roi, 0.3, prob_bg, 0.7, 0)

        cv2.putText(overlay, "ALL PROBABILITIES", (px + 10, py + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i in range(7):
            y = py + 55 + i * 40

            cv2.putText(overlay, self.class_names[i], (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            bar_y = y + 6
            cv2.rectangle(overlay, (px + 10, bar_y), (px + 210, bar_y + 10),
                          (40, 40, 40), -1)

            fill_width = int(200 * all_probs[i])
            if fill_width > 0:
                cv2.rectangle(overlay, (px + 10, bar_y),
                              (px + 10 + fill_width, bar_y + 10),
                              self.class_colors[i], -1)

            cv2.rectangle(overlay, (px + 10, bar_y), (px + 210, bar_y + 10),
                          (100, 100, 100), 1)

            cv2.putText(overlay, f"{all_probs[i]*100:.0f}%",
                        (px + 220, bar_y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        bar_h = 45
        bar_y = h - bar_h

        bar_bg = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar_bg[:] = (15, 15, 15)
        bar_roi = overlay[bar_y:h, 0:w]
        overlay[bar_y:h, 0:w] = cv2.addWeighted(bar_roi, 0.2, bar_bg, 0.8, 0)

        info_y = bar_y + 28
        cv2.putText(overlay, f"Model: {self.model_type}", (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        fps_color = (100, 255, 100) if fps > 15 else (200, 200, 200)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (200, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)

        cv2.putText(overlay, f"Threshold: {self.threshold:.2f}",
                    (330, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(overlay, "Controls: Q=Quit | S=Save | +=More | -=Less",
                    (w - 450, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return overlay

    def display_results_gui(self, results, dataset_path, accuracy=None):
        """Display dataset results in a scrollable GUI window"""

        width, height = 1000, 700

        # Calculate class distribution
        class_counts = {}
        correct_counts = {}
        for result in results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if result.get('correct', False):
                correct_counts[class_name] = correct_counts.get(
                    class_name, 0) + 1

        total = len(results)

        # Create scrollable results window
        window_name = "Dataset Classification Results"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

        scroll_pos = 0
        max_scroll = max(0, len(results) - 12)
        scroll_delta = 0

        def mouse_callback(event, x, y, flags, param):
            nonlocal scroll_delta
            if event == cv2.EVENT_MOUSEWHEEL:
                scroll_delta = -1 if flags > 0 else 1

        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (25, 25, 25)

            # Header with gradient
            title_height = 100
            for i in range(title_height):
                alpha = 1.0 - (i / title_height) * 0.5
                color_val = int(40 * alpha)
                frame[i, :] = (color_val, color_val, color_val)

            # Title
            cv2.putText(frame, "CLASSIFICATION RESULTS", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            # Subtitle with accuracy
            subtitle = f"Model: {self.model_type} | Total: {total} images"
            if accuracy is not None:
                subtitle += f" | Accuracy: {accuracy:.2f}%"
            cv2.putText(frame, subtitle,
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # Separator
            cv2.line(frame, (20, title_height), (width - 20, title_height),
                     (80, 80, 80), 2)

            # Summary panel
            summary_y = title_height + 20
            summary_height = 220

            cv2.rectangle(frame, (30, summary_y), (width - 30, summary_y + summary_height),
                          (40, 40, 40), -1)
            cv2.rectangle(frame, (30, summary_y), (width - 30, summary_y + summary_height),
                          (80, 80, 80), 2)

            # Accuracy display with better spacing
            if accuracy is not None:
                cv2.putText(frame, "OVERALL ACCURACY", (50, summary_y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Large accuracy percentage (moved down)
                acc_text = f"{accuracy:.2f}%"
                acc_color = (100, 255, 100) if accuracy >= 75 else (
                    255, 200, 100) if accuracy >= 50 else (60, 60, 240)
                cv2.putText(frame, acc_text, (50, summary_y + 90),  # Changed from 70 to 90
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, acc_color, 3)

                # Accuracy bar (adjusted position)
                bar_x, bar_y = 280, summary_y + 60  # Adjusted from 45
                bar_width, bar_height = 650, 30

                # Accuracy bar
                bar_x, bar_y = 280, summary_y + 45
                bar_width, bar_height = 650, 30

                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                              (60, 60, 60), -1)

                # Fill based on accuracy
                fill_width = int(bar_width * (accuracy / 100))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                              acc_color, -1)

                # Border
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                              (120, 120, 120), 2)

                # Separator line
                cv2.line(frame, (40, summary_y + 95), (width - 40, summary_y + 95),
                         (80, 80, 80), 1)

            # Class distribution
            dist_y = summary_y + 110
            cv2.putText(frame, "CLASS DISTRIBUTION", (50, dist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            col1_x, col2_x = 50, 520
            row_y = dist_y + 25
            row_spacing = 27

            for idx, class_name in enumerate(self.class_names):
                count = class_counts.get(class_name, 0)
                percentage = (count / total * 100) if total > 0 else 0

                x_pos = col1_x if idx < 4 else col2_x
                y_pos = row_y + (idx % 4) * row_spacing

                color = self.class_colors[idx]

                # Color indicator
                cv2.circle(frame, (x_pos + 10, y_pos - 5), 6, color, -1)
                cv2.circle(frame, (x_pos + 10, y_pos - 5),
                           6, (200, 200, 200), 1)

                # Text - shorten the display format
                text = f"{class_name}: {count} ({percentage:.1f}%)"
                cv2.putText(frame, text, (x_pos + 25, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

            # Results table header
            table_y = summary_y + summary_height + 20
            cv2.putText(frame, "DETAILED RESULTS", (30, table_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Table headers
            header_y = table_y + 30
            cv2.rectangle(frame, (30, header_y - 20), (width - 30, header_y + 5),
                          (50, 50, 50), -1)

            cv2.putText(frame, "Filename", (50, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "True", (320, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "Predicted", (480, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "Conf.", (640, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "Result", (730, header_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Display results (scrollable)
            result_y = header_y + 20
            row_height = 26

            for idx in range(scroll_pos, min(scroll_pos + 12, len(results))):
                result = results[idx]
                y_pos = result_y + (idx - scroll_pos) * row_height

                # Alternate row colors
                if (idx - scroll_pos) % 2 == 0:
                    cv2.rectangle(frame, (30, y_pos - 16), (width - 30, y_pos + 8),
                                  (35, 35, 35), -1)

                # Filename (truncated if too long)
                filename = result['filename']
                if len(filename) > 25:
                    filename = filename[:22] + "..."
                cv2.putText(frame, filename, (50, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

                # True class
                true_label = result.get('true_label', -1)
                if true_label != -1:
                    true_color = self.class_colors[true_label]
                    cv2.circle(frame, (320, y_pos - 5), 5, true_color, -1)
                    true_name = result.get('true_class_name', 'Unknown')
                    # No truncation
                    cv2.putText(frame, true_name, (335, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

                # Predicted class with color indicator
                class_id = result['predicted_class']
                color = self.class_colors[class_id]
                cv2.circle(frame, (480, y_pos - 5), 5, color, -1)
                pred_name = result['class_name']
                cv2.putText(frame, pred_name, (495, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

                # Confidence
                conf_text = f"{result['confidence']*100:.0f}%"
                cv2.putText(frame, conf_text, (640, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

                # Result indicators
                if result.get('correct', False):
                    cv2.circle(frame, (745, y_pos - 5), 8, (100, 255, 100), -1)
                    cv2.circle(frame, (745, y_pos - 5), 8, (150, 255, 150), 1)
                    cv2.line(frame, (741, y_pos - 4),
                             (743, y_pos - 1), (40, 40, 40), 2)
                    cv2.line(frame, (743, y_pos - 1),
                             (749, y_pos - 9), (40, 40, 40), 2)
                elif true_label != -1:
                    cv2.circle(frame, (745, y_pos - 5), 8, (60, 60, 240), -1)
                    cv2.circle(frame, (745, y_pos - 5), 8, (100, 100, 255), 1)
                    cv2.line(frame, (741, y_pos - 9),
                             (749, y_pos - 1), (30, 30, 30), 2)
                    cv2.line(frame, (749, y_pos - 9),
                             (741, y_pos - 1), (30, 30, 30), 2)

            # Scroll indicator
            if max_scroll > 0:
                scroll_bar_height = 250
                scroll_bar_y = header_y + 30
                scroll_bar_x = width - 25

                # Scroll track
                cv2.rectangle(frame, (scroll_bar_x, scroll_bar_y),
                              (scroll_bar_x + 10, scroll_bar_y + scroll_bar_height),
                              (60, 60, 60), -1)

                # Scroll thumb
                thumb_height = max(
                    20, int(scroll_bar_height * (12 / len(results))))
                thumb_y = scroll_bar_y + \
                    int((scroll_pos / max_scroll) *
                        (scroll_bar_height - thumb_height)) if max_scroll > 0 else scroll_bar_y
                cv2.rectangle(frame, (scroll_bar_x, thumb_y),
                              (scroll_bar_x + 10, thumb_y + thumb_height),
                              (120, 120, 120), -1)

            # Bottom instruction bar
            bar_height = 50
            bar_y = height - bar_height

            bar_bg = np.zeros((bar_height, width, 3), dtype=np.uint8)
            bar_bg[:] = (15, 15, 15)
            frame[bar_y:height, :] = bar_bg

            cv2.putText(frame, "Controls: arrows or Mouse Wheel = Scroll | Q or ESC = Close",
                        (30, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if scroll_delta != 0:
                scroll_pos = min(max_scroll, max(0, scroll_pos + scroll_delta))
                scroll_delta = 0

            cv2.imshow(window_name, frame)

            # Handle input
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == 82:  # Up arrow
                scroll_pos = max(0, scroll_pos - 1)
            elif key == 84:  # Down arrow
                scroll_pos = min(max_scroll, scroll_pos + 1)

        cv2.destroyAllWindows()

    def run(self, camera_id=0):
        """Run real-time classification"""

        print(f"Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_id}")
            if camera_id == 0:
                print("Trying camera 1...")
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    print("[ERROR] No camera available")
                    return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("[OK] Camera ready\n")

        print("="*60)
        print("CONTROLS:")
        print("  Q or ESC : Quit")
        print("  S        : Save screenshot")
        print("  + or =   : Increase threshold")
        print("  - or _   : Decrease threshold")
        print("="*60 + "\n")

        window_name = f"MSI System - {self.model_type}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        screenshot_count = 0
        frame_count = 0

        try:
            while True:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[INFO] Window closed")
                    break

                start_time = time()

                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)

                # Classify frame (uses existing extraction logic)
                class_id, confidence, all_probs = self.classify(frame)

                if len(self.frame_times) > 0:
                    avg_time = sum(self.frame_times) / len(self.frame_times)
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                else:
                    fps = 0

                display = self.draw_ui(
                    frame, class_id, confidence, all_probs, fps)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    print("\n[INFO] Quitting...")
                    break

                elif key == ord('s'):
                    filename = f"screenshot_{screenshot_count:03d}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"[OK] Saved: {filename}")
                    screenshot_count += 1

                elif key == ord('+') or key == ord('='):
                    self.threshold = min(0.95, self.threshold + 0.05)
                    print(f"[INFO] Threshold: {self.threshold:.2f}")

                elif key == ord('-') or key == ord('_'):
                    self.threshold = max(0.30, self.threshold - 0.05)
                    print(f"[INFO] Threshold: {self.threshold:.2f}")

                self.frame_times.append(time() - start_time)
                frame_count += 1

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

        finally:
            cap.release()
            cv2.destroyAllWindows()

            if len(self.frame_times) > 0:
                avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                print(f"\n{'='*60}")
                print(f"SESSION SUMMARY")
                print(f"{'='*60}")
                print(f"Model: {self.model_type}")
                print(f"Frames processed: {frame_count}")
                print(f"Average FPS: {avg_fps:.2f}")
                print(f"Screenshots saved: {screenshot_count}")
                print(f"{'='*60}\n")

        return True


def draw_menu_window(title, options, subtitle="", selected_idx=0):
    """Create a graphical menu window with consistent styling"""

    width, height = 900, 650
    menu_frame = np.zeros((height, width, 3), dtype=np.uint8)
    menu_frame[:] = (25, 25, 25)

    # Title area with gradient
    title_height = 120
    for i in range(title_height):
        alpha = 1.0 - (i / title_height) * 0.5
        color_val = int(40 * alpha)
        menu_frame[i, :] = (color_val, color_val, color_val)

    # Draw title
    cv2.putText(menu_frame, title, (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    if subtitle:
        cv2.putText(menu_frame, subtitle, (50, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Separator line
    cv2.line(menu_frame, (30, title_height), (width - 30, title_height),
             (80, 80, 80), 2)

    # Draw menu options
    start_y = title_height + 80
    spacing = 110

    for idx, option in enumerate(options):
        y_pos = start_y + idx * spacing

        # Option card
        card_x1, card_y1 = 50, y_pos - 50
        card_x2, card_y2 = width - 50, y_pos + 35

        # Highlight selected option
        if idx == selected_idx:
            border_color = (80, 160, 255)
            bg_color = (45, 45, 45)
            border_thickness = 3
        else:
            border_color = (100, 100, 100)
            bg_color = (35, 35, 35)
            border_thickness = 2

        # Draw card
        cv2.rectangle(menu_frame, (card_x1, card_y1), (card_x2, card_y2),
                      bg_color, -1)
        cv2.rectangle(menu_frame, (card_x1, card_y1), (card_x2, card_y2),
                      border_color, border_thickness)

        # Number badge
        badge_size = 45
        badge_x = card_x1 + 35
        badge_y = y_pos - 15

        badge_color = (80, 160, 255) if idx == selected_idx else (60, 60, 60)
        cv2.circle(menu_frame, (badge_x, badge_y), badge_size // 2,
                   badge_color, -1)
        cv2.circle(menu_frame, (badge_x, badge_y), badge_size // 2,
                   (120, 180, 255) if idx == selected_idx else (100, 100, 100), 2)

        # Number text
        num_text = str(idx + 1)
        text_size = cv2.getTextSize(
            num_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        num_x = badge_x - text_size[0] // 2
        num_y = badge_y + text_size[1] // 2

        cv2.putText(menu_frame, num_text, (num_x, num_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Option text
        text_color = (255, 255, 255) if idx == selected_idx else (
            220, 220, 220)
        cv2.putText(menu_frame, option, (badge_x + 50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)

    # Bottom instruction bar
    bar_height = 60
    bar_y = height - bar_height

    bar_bg = np.zeros((bar_height, width, 3), dtype=np.uint8)
    bar_bg[:] = (15, 15, 15)
    menu_frame[bar_y:height, :] = bar_bg

    cv2.putText(menu_frame, "Controls: arrows or Mouse Wheel = Scroll | Enter or Click = Select | Q or ESC = Exit",
                (30, bar_y + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return menu_frame


def show_menu(title, options, subtitle=""):
    """Display menu with mouse and keyboard navigation"""

    window_name = "MSI System - Menu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 650)

    selected_idx = 0
    mouse_y = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_idx, mouse_y
        mouse_y = y

        # Calculate which option is being hovered
        title_height = 120
        start_y = title_height + 80
        spacing = 110

        for idx in range(len(options)):
            option_y = start_y + idx * spacing
            card_y1 = option_y - 50
            card_y2 = option_y + 35

            if card_y1 <= y <= card_y2:
                selected_idx = idx
                break

        # Handle mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            param['clicked'] = True

    click_data = {'clicked': False}
    cv2.setMouseCallback(window_name, mouse_callback, click_data)

    while True:
        menu_frame = draw_menu_window(title, options, subtitle, selected_idx)
        cv2.imshow(window_name, menu_frame)

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None

        # Check for mouse click
        if click_data['clicked']:
            click_data['clicked'] = False
            cv2.destroyAllWindows()
            return selected_idx + 1

        key = cv2.waitKey(50)

        # Arrow key navigation
        if key == 82 or key == 0:  # Up arrow
            selected_idx = (selected_idx - 1) % len(options)
        elif key == 84 or key == 1:  # Down arrow
            selected_idx = (selected_idx + 1) % len(options)

        # Enter key to select
        elif key == 13 or key == 10:  # Enter
            cv2.destroyAllWindows()
            return selected_idx + 1

        # Number keys
        elif ord('1') <= key <= ord('9'):
            choice = key - ord('0')
            if choice <= len(options):
                cv2.destroyAllWindows()
                return choice

        # ESC or Q to exit
        elif key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return None


def select_folder():
    """Open folder selection dialog"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(
        title="Select Dataset Folder",
        initialdir=os.getcwd()
    )

    root.destroy()

    return folder_path if folder_path else None


def main():
    """Main entry point with graphical menu system"""

    while True:
        # Level 1: Model Selection
        model_options = [
            "SVM  (Support Vector Machine)",
            "KNN  (K-Nearest Neighbors)",
            "Exit Application"
        ]

        model_choice = show_menu(
            "MSI SYSTEM",
            model_options,
            "Material Stream Identification - Select Model"
        )

        if model_choice is None or model_choice == 3:
            print("\n[INFO] Exiting application...")
            break

        # Map choice to model files
        if model_choice == 1:
            model_type = "SVM"
            model_file = "SVCModel.pkl"
            scaler_file = "SVCScaler.pkl"
        else:
            model_type = "KNN"
            model_file = "KNNModel.pkl"
            scaler_file = "KNNScaler.pkl"

        # Verify model files
        if not os.path.exists(model_file):
            print(f"\n[ERROR] Model file not found: {model_file}")
            continue

        if not os.path.exists(scaler_file):
            print(f"\n[ERROR] Scaler file not found: {scaler_file}")
            continue

        # Level 2: Mode Selection
        while True:
            mode_options = [
                "Live Camera  (Real-time classification)",
                "Dataset Testing  (Hidden test set)",
                "Back to Model Selection"
            ]

            mode_choice = show_menu(
                f"MSI SYSTEM - {model_type}",
                mode_options,
                "Select Operation Mode"
            )

            if mode_choice is None or mode_choice == 3:
                break

            try:
                # Initialize classifier
                classifier = MSIClassifier(model_file, scaler_file, model_type)

                if mode_choice == 1:
                    # Live camera mode
                    classifier.run(camera_id=0)

                elif mode_choice == 2:
                    # Dataset testing mode
                    print("\n" + "="*60)
                    print("DATASET TESTING MODE")
                    print("="*60)
                    print("\nOpening folder browser...")

                    dataset_path = select_folder()

                    if not dataset_path:
                        print("[INFO] No folder selected")
                        continue

                    print(f"[INFO] Selected: {dataset_path}")

                    # Process dataset (uses pre-computed features if available)
                    classifier.process_dataset(dataset_path)

            except FileNotFoundError as e:
                print(f"\n[ERROR] {e}")
                input("Press Enter to continue...")
                break

            except RuntimeError as e:
                print(f"\n[ERROR] {e}")
                input("Press Enter to continue...")
                break

            except Exception as e:
                print(f"\n[ERROR] Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                input("Press Enter to continue...")
                break

    print("\n[INFO] Program terminated\n")


if __name__ == "__main__":
    main()
