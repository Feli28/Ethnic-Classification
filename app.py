import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import mediapipe as mp

from feature_extraction import (
    detect_landmarks,
    extract_antropometri,
    extract_antropometri13,
    extract_antropometri22,
    extract_lbp,
    extract_lbp_ycbcr,
    extract_gabor,
)

# =====================================================
# CONFIG
# =====================================================
FACE_SIZE = (510, 510)

# =====================================================
# MEDIAPIPE
# =====================================================
mp_detection = mp.solutions.face_detection
mp_selfie = mp.solutions.selfie_segmentation

face_detector = mp_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5,
)
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

# =====================================================
# LOAD MODEL (19 MODEL)
# =====================================================
@st.cache_resource
def load_models():
    models = {}

    # --- 9 MODEL LAMA ---
    models["LBP"] = {
        "model": joblib.load("models/lbp_svm.pkl"),
        "scaler": joblib.load("models/lbp_scaler.pkl"),
    }
    models["Antropometri 10 Multi Angle"] = {
        "model": joblib.load("models/antro_svm.pkl"),
        "scaler": joblib.load("models/antro_scaler.pkl"),
    }
    models["Antro Frontal Angry"] = {
        "model": joblib.load("models/antropometri_angry_frontal_svm.pkl"),
        "scaler": joblib.load("models/antropometri_angry_frontal_scaler.pkl"),
    }
    models["Antropometri 13 Angry Frontal"] = {
        "model": joblib.load("models/antropometri13_angry_frontal_svm.pkl"),
        "scaler": joblib.load("models/antropometri13_angry_frontal_scaler.pkl"),
    }
    models["Antropometri 22 Angry Frontal"] = {
        "model": joblib.load("models/antro22_angry_frontal_svm.pkl"),
        "scaler": joblib.load("models/antro22_angry_frontal_scaler.pkl"),
    }
    models["LBP YCbCr + Antropometri"] = {
        "model": joblib.load("models/lbp_ycbcr_antro_svm.pkl"),
        "scaler": joblib.load("models/lbp_ycbcr_antro_scaler.pkl"),
    }
    models["LBP YCbCr + Gabor"] = {
        "model": joblib.load("models/gabor_lbp_ycbcr_svm.pkl"),
        "scaler": joblib.load("models/gabor_lbp_ycbcr_scaler.pkl"),
    }
    models["OVA Antro + LBP YCbCr"] = {
        "model": joblib.load("models/OVA_antro_lbp_ycbcr_svm.pkl"),
        "scaler": joblib.load("models/OVA_antro_lbp_ycbcr_scaler.pkl"),
    }
    models["OVA Gabor + LBP YCbCr"] = {
        "model": joblib.load("models/OVA_gabor_lbp_ycbcr_svm.pkl"),
        "scaler": joblib.load("models/OVA_gabor_lbp_ycbcr_scaler.pkl"),
    }

    # --- 6 MODEL KAGGLE SEBELUMNYA ---
    models["Antro 10 Multi-Emotion Frontal"] = {
        "model": joblib.load("models/antro10_multi_emotion_frontal_svm.pkl"),
        "scaler": joblib.load("models/antro10_multi_emotion_frontal_scaler.pkl"),
    }
    models["OVA Antro 10 Multi-Emotion Frontal"] = {
        "model": joblib.load("models/antro10_multi_emotion_frontal_ova_svm.pkl"),
        "scaler": joblib.load("models/antro10_multi_emotion_frontal_ova_scaler.pkl"),
    }
    models["Antro 13 Multi-Emotion Frontal"] = {
        "model": joblib.load("models/antro13_multi_emotion_frontal_svm.pkl"),
        "scaler": joblib.load("models/antro13_multi_emotion_frontal_scaler.pkl"),
    }
    models["OVA Antro 13 Multi-Emotion Frontal"] = {
        "model": joblib.load("models/antro13_multi_emotion_frontal_ova_svm.pkl"),
        "scaler": joblib.load("models/antro13_multi_emotion_frontal_ova_scaler.pkl"),
    }
    models["Antro 22 Multi-Emotion Frontal"] = {
        "model": joblib.load("models/antro22_multi_emotion_frontal_svm.pkl"),
        "scaler": joblib.load("models/antro22_multi_emotion_frontal_scaler.pkl"),
    }
    models["OVA Antro 22 Multi-Emotion Frontal"] = {
        "model": joblib.load("models/antro22_multi_emotion_frontal_ova_svm.pkl"),
        "scaler": joblib.load("models/antro22_multi_emotion_frontal_ova_scaler.pkl"),
    }

    # --- 4 MODEL TAMBAHAN TERBARU DARI GAMBAR ---
    models["Fusi LBP-YCbCr-Gabor Subj-Indep SVM"] = {
        "model": joblib.load("models/lbp_ycbcr_gabor_subj_indep_svm.pkl"),
        "scaler": joblib.load("models/lbp_ycbcr_gabor_subj_indep_scaler.pkl"),
    }
    models["Fusi LBP-YCbCr-Gabor Subj-Indep OVA"] = {
        "model": joblib.load("models/lbp_ycbcr_gabor_subj_indep_ova_svm.pkl"),
        "scaler": joblib.load("models/lbp_ycbcr_gabor_subj_indep_ova_scaler.pkl"),
    }
    models["Antro 22 Frontal Subj-Indep SVM"] = {
        "model": joblib.load("models/antro22_multi_emotion_frontal_subj_indep_svm.pkl"),
        "scaler": joblib.load("models/antro22_multi_emotion_frontal_subj_indep_scaler.pkl"),
    }
    models["Antro 22 Frontal OVA 60:20:20"] = {
        "model": joblib.load("models/antro22_multi_emotion_frontal_ova_60_20_20_svm.pkl"),
        "scaler": joblib.load("models/antro22_multi_emotion_frontal_ova_60_20_20_scaler.pkl"),
    }

    return models

# =====================================================
# CROP MULTIPLE FACES (SUPPORT LEBIH DARI 1 ORANG)
# =====================================================
def crop_all_faces(image):
    """Mendeteksi seluruh wajah dan mengembalikan list potongan wajah persegi beserta bbox koordinat."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)

    if not result.detections:
        return []

    cropped_faces = []
    h, w = image.shape[:2]

    for detection in result.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        margin_x = int(bw * 0.3)
        margin_y = int(bh * 0.35)

        crop_w = bw + 2 * margin_x
        crop_h = bh + 2 * margin_y
        size = max(crop_w, crop_h)
        half = size // 2

        cx = x + bw // 2
        cy = y + bh // 2

        if w >= size:
            cx = max(half, min(cx, w - half))
        if h >= size:
            cy = max(half, min(cy, h - half))

        half_size = min(half, cx, cy, w - cx, h - cy)

        x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
        x2, y2 = min(w, cx + half_size), min(h, cy + half_size)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size > 0:
            cropped_faces.append((face_crop, (x, y, bw, bh)))

    return cropped_faces

# =====================================================
# REMOVE BACKGROUND
# =====================================================
def remove_background(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)

    if result.segmentation_mask is None:
        return image

    mask = (result.segmentation_mask > 0.35).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    black = np.zeros_like(image)
    background = cv2.bitwise_and(black, black, mask=1 - mask)

    return cv2.add(foreground, background)

# =====================================================
# RESIZE FACE
# =====================================================
def resize_face(image):
    return cv2.resize(image, FACE_SIZE, interpolation=cv2.INTER_AREA)

# =====================================================
# FEATURE EXTRACTION (PER WAJAH)
# =====================================================
def extract_features(image):
    features = {}

    lm468 = detect_landmarks(image, mode="468")
    lm478 = detect_landmarks(image, mode="478")

    if len(lm468) == 0 or len(lm478) == 0:
        return None

    # Mengambil landmark wajah hasil crop
    pt468 = lm468[0]
    pt478 = lm478[0]

    antro10 = extract_antropometri(pt468)
    antro13 = extract_antropometri13(pt478)
    antro22 = extract_antropometri22(pt478)

    lbp = extract_lbp(image)
    lbp_ycbcr = extract_lbp_ycbcr(image)
    gabor = extract_gabor(image)

    feat_lbp_ycbcr_antro = np.concatenate([lbp_ycbcr, antro10])
    feat_lbp_ycbcr_gabor = np.concatenate([lbp_ycbcr, gabor])

    # Model Lama
    features["LBP"] = lbp
    features["Antropometri 10 Multi Angle"] = antro10
    features["Antro Frontal Angry"] = antro10
    features["Antropometri 13 Angry Frontal"] = antro13
    features["Antropometri 22 Angry Frontal"] = antro22
    features["LBP YCbCr + Antropometri"] = feat_lbp_ycbcr_antro
    features["LBP YCbCr + Gabor"] = feat_lbp_ycbcr_gabor
    features["OVA Antro + LBP YCbCr"] = feat_lbp_ycbcr_antro
    features["OVA Gabor + LBP YCbCr"] = feat_lbp_ycbcr_gabor

    # Model Kaggle Sebelumnya
    features["Antro 10 Multi-Emotion Frontal"] = antro10
    features["OVA Antro 10 Multi-Emotion Frontal"] = antro10
    features["Antro 13 Multi-Emotion Frontal"] = antro13
    features["OVA Antro 13 Multi-Emotion Frontal"] = antro13
    features["Antro 22 Multi-Emotion Frontal"] = antro22
    features["OVA Antro 22 Multi-Emotion Frontal"] = antro22

    # Model Baru Tambahan dari Gambar
    features["Fusi LBP-YCbCr-Gabor Subj-Indep SVM"] = feat_lbp_ycbcr_gabor
    features["Fusi LBP-YCbCr-Gabor Subj-Indep OVA"] = feat_lbp_ycbcr_gabor
    features["Antro 22 Frontal Subj-Indep SVM"] = antro22
    features["Antro 22 Frontal OVA 60:20:20"] = antro22

    return features

# =====================================================
# PREDICT
# =====================================================
def predict(model_data, feature):
    model = model_data["model"]
    scaler = model_data["scaler"]

    feature = feature.reshape(1, -1)
    feature_scaled = scaler.transform(feature)
    return model.predict(feature_scaled)[0]

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Ethnicity Classification (Multi-Face)", layout="wide")

st.title("Facial Feature Based Ethnicity Classification")
st.write("SVM Classification Pipeline - 19 Model Grid Comparison (Support Multi-Face Detection)")

option = st.radio("Metode Input", ["Upload Foto", "Ambil Foto"])
image_bgr = None

if option == "Upload Foto":
    file = st.file_uploader("Upload citra wajah", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file)
        img = np.array(image)
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
else:
    camera = st.camera_input("Ambil foto")
    if camera:
        image = Image.open(camera)
        img = np.array(image)
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.flip(image_bgr, 1)

# =====================================================
# PROSES DETEKSI & PREDIKSI
# =====================================================
if image_bgr is not None:
    st.subheader("Citra Asli")
    faces_detected = crop_all_faces(image_bgr)

    # Beri kotak bounding box visual di citra asli jika ada deteksi
    vis_img = image_bgr.copy()
    for idx, (_, (bx, by, bw, bh)) in enumerate(faces_detected, 1):
        cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        cv2.putText(vis_img, f"Wajah {idx}", (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), width=450)

    num_faces = len(faces_detected)
    if num_faces == 0:
        st.error("Wajah tidak terdeteksi pada citra.")
        st.stop()

    st.success(f"Terdeteksi {num_faces} wajah.")
    models = load_models()
    names = list(models.keys())

    # Jika multi-face, buatkan tab terpisah per individu
    face_tabs = st.tabs([f"Wajah {i+1}" for i in range(num_faces)])

    for idx, (face_crop, _) in enumerate(faces_detected):
        with face_tabs[idx]:
            col_preview, col_proc = st.columns(2)
            with col_preview:
                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption=f"Crop Wajah {idx+1}", width=220)

            # Preprocessing Wajah
            clean_face = remove_background(face_crop)
            clean_face = resize_face(clean_face)

            with col_proc:
                st.image(cv2.cvtColor(clean_face, cv2.COLOR_BGR2RGB), caption=f"Preprocessing (510x510 No BG)", width=220)

            with st.spinner(f"Ekstraksi fitur dan inferensi 19 model untuk Wajah {idx+1}..."):
                features = extract_features(clean_face)

            if features is None:
                st.error(f"Gagal mendeteksi landmark pada Wajah {idx+1}.")
            else:
                st.write("---")
                st.subheader(f"Hasil Prediksi 19 Model (Wajah {idx+1})")

                # Tampilkan metrik prediksi dalam format 3 kolom
                for i in range(0, len(names), 3):
                    cols = st.columns(3)
                    for c, name in zip(cols, names[i : i + 3]):
                        with c:
                            res = predict(models[name], features[name])
                            st.metric(label=name, value=res)