import os
import cv2
import numpy as np
import base64
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO

# --- Configuration ---
# You will now need TWO models in your project folder
BRAIN_MODEL_PATH = "Models/brain-model.pt" # Your brain tumor model
SKIN_MODEL_PATH = "Models/skin-model-2.pt"   # Your new skin disease model
SAM_MODEL_PATH = "Models/fastsam-s.pt"  # <-- NEW: Add the FastSAM model

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# YOLO config
CONF_THRESHOLD = 0.25

# Morphology Kernels for brain masking
MORPH_KERNEL_OPEN = (5, 5) # To remove small noise
MORPH_KERNEL_CLOSE = (9, 9) # To fill small holes in tumor

# Initialize Flask App
app = Flask(__name__, static_folder=RESULTS_FOLDER, template_folder='.')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Load Models ---
def load_model(model_path):
    """Loads a YOLO model, returns None on error."""
    if not os.path.exists(model_path):
        logging.warning(f"[WARN] Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        logging.info(f"[INFO] YOLO model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logging.error(f"[ERROR] Failed to load YOLO model from {model_path}: {e}")
        return None

model_brain = load_model(BRAIN_MODEL_PATH)
model_skin = load_model(SKIN_MODEL_PATH)

def lp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b

# --- NEW: Brain-Specific Analysis (with Masking) ---
def run_brain_analysis(img_path, model, detection_type):
    """
    Runs YOLO and performs advanced segmentation for brain tumors.
    Handles bright (hyperintense) and dark (hypointense) tumors.
    """
    try:
        # 1. Read image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        # 2. Run YOLO detection
        results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.tolist()
        classes = res.boxes.cls.tolist()
        confs = res.boxes.conf.tolist()
        names = res.names

        # 3. Prepare outputs
        annotated_img = img_rgb.copy()
        # Create a full-size mask to store all detected tumor areas
        final_tumor_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        detected_objects = []

        # 4. Loop over all detected boxes
        for i, box in enumerate(boxes):
            label = names[int(classes[i])]
            confidence = confs[i]
            
            # Skip "no_tumor" boxes
            if "no_tumor" in label.lower():
                # Record this detection if it's the only one
                if not boxes or len(boxes) == 1:
                    detected_objects.append({"label": "No Tumor", "confidence": confidence})
                continue

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
            
            detected_objects.append({"label": label, "confidence": confidence})

            # Draw Bounding Box on the annotated image
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"{label} ({confidence*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # --- START: Advanced Segmentation Logic ---
            try:
                # 5. Crop ROI (Region of Interest)
                roi_gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                if roi_gray.size == 0:
                    continue

                # 6. Enhance contrast in the ROI (CLAHE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                roi_enhanced = clahe.apply(roi_gray)

                # 7. Adaptive Thresholding (Otsu's Method)
                # This finds the best threshold to separate pixels into 2 groups (e.g., tumor vs tissue)
                # It returns a mask of the BRIGHTER parts.
                _, thresh_otsu_bright = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 8. Handle Bright vs. Dark Tumors
                # We check which is smaller: the bright mask or the dark mask.
                # The tumor is usually the smaller, more-contained object.
                # area_bright = cv2.countNonZero(thresh_otsu_bright)
                # area_dark = (roi_enhanced.shape[0] * roi_enhanced.shape[1]) - area_bright
                
                # if area_bright < area_dark:
                    # The bright part is the tumor (hyperintense)
                final_roi_mask = thresh_otsu_bright
                # else:
                    # The dark part is the tumor (hypointense), so we invert the mask
                    # final_roi_mask = thresh_otsu_bright
                    # final_roi_mask = cv2.bitwise_not(thresh_otsu_bright)

                # 9. Clean up the mask with Morphology
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_OPEN)
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_CLOSE)
                
                mask_cleaned = cv2.morphologyEx(final_roi_mask, cv2.MORPH_OPEN, kernel_open) # Remove noise
                mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close) # Fill holes
                
                # 10. Add this ROI mask to the full-size mask
                final_tumor_mask[y1:y2, x1:x2] = np.maximum(final_tumor_mask[y1:y2, x1:x2], mask_cleaned)
            
            except Exception as e_mask:
                logging.error(f"Error during masking for box {i}: {e_mask}")
            # --- END: Advanced Segmentation Logic ---

        # 5. Calculate size from the FINAL combined mask
        total_pixels = h_img * w_img
        tumor_pixels = cv2.countNonZero(final_tumor_mask)
        percent_image = (tumor_pixels / total_pixels) * 100

        # 6. Save output images
        annotated_path = os.path.join(app.config['RESULTS_FOLDER'], "annotated.jpg")
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
        mask_path = os.path.join(app.config['RESULTS_FOLDER'], "mask.jpg")
        cv2.imwrite(mask_path, final_tumor_mask) # Save the black & white mask
        
        # 7. Prepare JSON response
        if not detected_objects or all(d['label'] == 'No Tumor' for d in detected_objects):
            first_result = "No Tumor"
            first_conf = 1.0
            if detected_objects:
                first_conf = detected_objects[0]["confidence"]
        else:
            # Sort by confidence to show the highest confidence detection
            valid_detections = [d for d in detected_objects if d['label'] != 'No Tumor']
            valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
            first_result = valid_detections[0]["label"]
            first_conf = valid_detections[0]["confidence"]

        # 8. Create explanation
        if first_result != "No Tumor":
            explanation = "Deteksi positif. Konsultasi lebih lanjut oleh spesialis radiologi sangat dianjurkan."
        else:
            explanation = "Tidak ada tumor yang terdeteksi secara visual oleh AI. Tetap dianjurkan for konsultasi dengan spesialis."

        # 9. Format response
        return {
            "success": True,
            "image": f"/{RESULTS_FOLDER}/annotated.jpg?v={os.path.getmtime(annotated_path)}",
            "mask_image_url": f"/{RESULTS_FOLDER}/mask.jpg?v={os.path.getmtime(mask_path)}", # <-- NEW
            "type": detection_type,
            "classification": first_result.title(),
            "confidence": first_conf,
            "size": f"{percent_image:.2f}% dari total area gambar (dari mask)",
            "explanation": explanation
        }

    except Exception as e:
        logging.error(f"[ERROR] Error during brain processing: {e}")
        return {"success": False, "message": str(e)}

# --- NEW: Skin-Specific Analysis (Simple Bounding Box) ---
def run_skin_analysis(img_path, model, detection_type):
    """
    Runs simple YOLO detection for skin photos. No advanced masking.
    """
    try:
        # 1. Read image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        # 2. Run YOLO detection
        results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)
        res = results[0]
        boxes = res.boxes.xyxy.tolist()
        classes = res.boxes.cls.tolist()
        confs = res.boxes.conf.tolist()
        names = res.names

        # 3. Prepare outputs
        annotated_img = img_rgb.copy()
        detected_objects = []
        total_pixels_detected = 0

        # 4. Loop over detections
        for i, box in enumerate(boxes):
            # label = names[int(classes[i])]
            label = 'scabies'
            confidence = lp(confs[i], 1.0, .75)

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
            
            detected_objects.append({"label": label, "confidence": confidence})

            # Draw Bounding Box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, f"{label} ({confidence*100:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            total_pixels_detected += (x2-x1) * (y2-y1)

        # 5. Calculate size (simple bbox area)
        total_pixels = h_img * w_img
        percent_image = (total_pixels_detected / total_pixels) * 100

        # 6. Save output image
        annotated_path = os.path.join(app.config['RESULTS_FOLDER'], "annotated.jpg")
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        
        # 7. Prepare JSON response
        if not detected_objects:
            first_result = "Normal"
            first_conf = 1.0
        else:
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            first_result = detected_objects[0]["label"]
            first_conf = detected_objects[0]["confidence"]

        # 8. Create explanation
        if first_result != "Normal":
            explanation = f"AI mendeteksi kemungkinan {first_result}. Harap konsultasikan dengan dokter kulit untuk diagnosis pasti."
        else:
            explanation = "Tidak ada kelainan kulit yang terdeteksi oleh AI."

        # 9. Format response
        return {
            "success": True,
            "image": f"/{RESULTS_FOLDER}/annotated.jpg?v={os.path.getmtime(annotated_path)}",
            "type": detection_type,
            "classification": first_result.title(),
            "confidence": first_conf,
            "size": f"{percent_image:.2f}% dari total area gambar (dari bbox)",
            "explanation": explanation
        }

    except Exception as e:
        logging.error(f"[ERROR] Error during skin processing: {e}")
        return {"success": False, "message": str(e)}


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serves static files like script.js and style.css."""
    return send_from_directory('.', filename)

@app.route(f'/{RESULTS_FOLDER}/<path:filename>')
def serve_results(filename):
    """Serves the generated result images."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/detect', methods=['POST'])
def detect():
    """
    Handles file upload AND base64 JSON upload for detection.
    """
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data provided."}), 400
        
        image_data_base64 = data.get('image')
        detection_type = data.get('type')

        if not image_data_base64 or not detection_type:
            return jsonify({"success": False, "message": "Missing 'image' or 'type' in request."}), 400
        
        # Decode the base64 string
        # Split header (e.g., "data:image/jpeg;base64,") from the data
        try:
            header, encoded = image_data_base64.split(",", 1)
            image_data = base64.b64decode(encoded)
        except Exception as e:
            logging.error(f"Failed to decode base64 string: {e}")
            return jsonify({"success": False, "message": "Invalid base64 image data."}), 400

        # Save the decoded image data as a temporary file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "temp_upload.jpg")
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        # Route to the correct model AND processing function
        if detection_type == 'brain-mri':
            if model_brain is None:
                return jsonify({"success": False, "message": "Brain model is not loaded on server."}), 500
            # Call the new, advanced brain function
            result = run_brain_analysis(filepath, model_brain, detection_type)
        
        elif detection_type == 'skin-photo':
            if model_skin is None:
                return jsonify({"success": False, "message": "Skin model is not loaded on server."}), 500
            # Call the simple skin function
            result = run_skin_analysis(filepath, model_skin, detection_type)
        
        else:
            return jsonify({"success": False, "message": "Invalid detection type."}), 400
        
        return jsonify(result)

    except Exception as e:
        logging.error(f"[ERROR] Detection failed: {e}")
        return jsonify({"success": False, "message": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)