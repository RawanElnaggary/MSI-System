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
import sys
from tkinter import Tk, filedialog


class MSIClassifier:
    """Real-time material classification using pre-trained models"""

    def __init__(self, model_path, scaler_path, model_type):
        """Initialize classifier with model and feature extractor"""

        print(f"\n{'='*60}")
        print(f"Loading {model_type} Model")
        print(f"{'='*60}")

        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        # Load model and scaler
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.model_type = model_type
            print(f"[OK] {model_type} model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Load ResNet50 feature extractor
        print("[...] Loading ResNet50 feature extractor...")
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
            print("[OK] Feature extractor loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load feature extractor: {e}")

        # Configuration
        self.threshold = 0.60
        self.class_names = [
            "Glass", "Paper", "Cardboard",
            "Plastic", "Metal", "Trash", "Unknown"
        ]
        self.class_colors = {
            0: (255, 180, 0),
            1: (240, 240, 240),
            2: (80, 160, 255),
            3: (80, 220, 80),
            4: (200, 200, 200),
            5: (60, 60, 240),
            6: (140, 140, 140)
        }

        # Smoothing and metrics
        self.prediction_history = deque(maxlen=5)
        self.frame_times = deque(maxlen=30)

        print(f"[OK] Classifier ready (threshold: {self.threshold})")
        print(f"{'='*60}\n")

    def extract_features(self, frame):
        """Extract ResNet50 features from frame"""
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)

            img_tensor = self.preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                features = self.feature_extractor(img_tensor).numpy().squeeze()

            return features.reshape(1, -1)
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return None

    def classify(self, frame):
        """Classify frame with confidence-based rejection"""

        features = self.extract_features(frame)
        if features is None:
            return 6, 0.0, np.zeros(7)

        features_scaled = self.scaler.transform(features)

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features_scaled)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            all_probs = np.zeros(7)
            all_probs[:len(probs)] = probs
        else:
            pred_class = self.model.predict(features_scaled)[0]
            confidence = 1.0
            all_probs = np.zeros(7)
            all_probs[pred_class] = 1.0

        if confidence < self.threshold:
            pred_class = 6
            all_probs[6] = 1.0 - confidence
            confidence = all_probs[6]

        self.prediction_history.append(pred_class)
        if len(self.prediction_history) >= 3:
            stable_pred = max(set(self.prediction_history),
                              key=self.prediction_history.count)
        else:
            stable_pred = pred_class

        return stable_pred, confidence, all_probs

    def draw_ui(self, frame, class_id, confidence, all_probs, fps):
        """Draw UI overlay on frame"""

        h, w = frame.shape[:2]
        overlay = frame.copy()

        material = self.class_names[class_id]
        color = self.class_colors[class_id]

        # Main result card
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

        # Probabilities panel
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

        # Bottom status bar
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

    def process_dataset(self, dataset_path):
        """Process a hidden test dataset and display results in console"""

        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_path}")
        print(f"{'='*60}\n")

        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset path not found: {dataset_path}")
            return False

        results = []
        total = 0

        image_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))

        if len(image_files) == 0:
            print("[ERROR] No images found in dataset")
            return False

        print(f"[INFO] Found {len(image_files)} images")
        print(f"[INFO] Processing...\n")

        for idx, img_path in enumerate(image_files, 1):
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"[WARNING] Could not load: {img_path}")
                    continue

                class_id, confidence, all_probs = self.classify(frame)

                img_name = os.path.basename(img_path)
                results.append({
                    'filename': img_name,
                    'predicted_class': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': confidence
                })

                if idx % 10 == 0 or idx == len(image_files):
                    print(
                        f"[PROGRESS] {idx}/{len(image_files)} images processed")

                total += 1

            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"CLASSIFICATION RESULTS - {self.model_type}")
        print(f"{'='*60}")
        print(f"Total Images Processed: {total}")
        print(f"Threshold Used: {self.threshold}")
        print(f"\n{'='*60}")
        print(f"{'Filename':<30} {'Predicted':<15} {'Confidence':<12}")
        print(f"{'='*60}")

        for result in results:
            print(f"{result['filename']:<30} "
                  f"{result['class_name']:<15} "
                  f"{result['confidence']*100:>6.2f}%")

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {self.model_type}")
        print(f"Total images: {total}")
        print(f"Threshold: {self.threshold}")

        class_counts = {}
        for result in results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"\nPredicted Class Distribution:")
        for class_name in self.class_names:
            count = class_counts.get(class_name, 0)
            if count > 0:
                print(
                    f"  {class_name:<12}: {count:>4} ({count/total*100:>5.1f}%)")

        print(f"{'='*60}\n")

        return True

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


def draw_menu_window(title, options, subtitle=""):
    """Create a graphical menu window matching the live camera UI style"""

    # Window dimensions
    width, height = 800, 600
    menu_frame = np.zeros((height, width, 3), dtype=np.uint8)
    menu_frame[:] = (25, 25, 25)

    # Title area with gradient effect
    title_height = 100
    for i in range(title_height):
        alpha = 1.0 - (i / title_height) * 0.5
        color_val = int(40 * alpha)
        menu_frame[i, :] = (color_val, color_val, color_val)

    # Draw title
    cv2.putText(menu_frame, title, (50, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    if subtitle:
        cv2.putText(menu_frame, subtitle, (50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # Draw separator line
    cv2.line(menu_frame, (30, title_height + 10),
             (width - 30, title_height + 10), (80, 80, 80), 2)

    # Draw menu options
    start_y = title_height + 80
    spacing = 90

    for idx, option in enumerate(options):
        y_pos = start_y + idx * spacing

        # Option card background
        card_x1, card_y1 = 50, y_pos - 45
        card_x2, card_y2 = width - 50, y_pos + 30

        # Draw card with border
        cv2.rectangle(menu_frame, (card_x1, card_y1), (card_x2, card_y2),
                      (60, 60, 60), -1)
        cv2.rectangle(menu_frame, (card_x1, card_y1), (card_x2, card_y2),
                      (100, 100, 100), 2)

        # Number badge
        badge_size = 40
        badge_x = card_x1 + 20
        badge_y = y_pos - 20
        cv2.circle(menu_frame, (badge_x, badge_y), badge_size // 2,
                   (80, 160, 255), -1)
        cv2.circle(menu_frame, (badge_x, badge_y), badge_size // 2,
                   (120, 180, 255), 2)

        # Number text
        cv2.putText(menu_frame, str(idx + 1), (badge_x - 12, badge_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Option text
        cv2.putText(menu_frame, option, (badge_x + 50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)

    # Bottom instruction bar
    bar_height = 50
    bar_y = height - bar_height

    bar_bg = np.zeros((bar_height, width, 3), dtype=np.uint8)
    bar_bg[:] = (15, 15, 15)
    menu_frame[bar_y:height, :] = bar_bg

    cv2.putText(menu_frame, "Press the number key to select an option",
                (50, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return menu_frame


def show_menu(title, options, subtitle=""):
    """Display menu and get user selection"""

    window_name = "MSI System - Menu"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    menu_frame = draw_menu_window(title, options, subtitle)

    while True:
        cv2.imshow(window_name, menu_frame)

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return None

        key = cv2.waitKey(100) & 0xFF

        # Check for number keys
        if key >= ord('1') and key <= ord('9'):
            choice = key - ord('0')
            if choice <= len(options):
                cv2.destroyAllWindows()
                return choice

        # ESC or Q to exit
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return None

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

                    # Process dataset
                    success = classifier.process_dataset(dataset_path)

                    if success:
                        input("\nPress Enter to continue...")
                    else:
                        print("\n[ERROR] Dataset processing failed")
                        input("Press Enter to continue...")

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
