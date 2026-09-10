import gc
import time
import av
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from feature_extraction import (
    FACE_SIZE,
    extract_antropometri,
    extract_gabor,
    extract_hog,
    extract_lbp_ycbcr,
    get_all_faces_landmarks,
)

# STUN servers publik gratis Google agar WebRTC lancar di browser HP / cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Demo Klasifikasi Etnis", layout="wide")
st.title("Demo Klasifikasi Etnis")
st.markdown(
    "Upload foto wajah, ambil tangkapan kamera, atau gunakan **Live Webcam** langsung. "
    "Sistem mendeteksi wajah menggunakan **MediaPipe** dan memprediksi etnis."
)


def get_class_probabilities(model, scaled_feat):
    """Menghitung persentase keyakinan seluruh etnis via softmax."""
    decision_scores = model.decision_function(scaled_feat)[0]
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    probabilities = exp_scores / exp_scores.sum()
    pairs = list(zip(model.classes_, probabilities * 100))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def show_prediction_column(title, model, scaled_feat):
    """Menampilkan kolom hasil dan expander seluruh etnis."""
    st.markdown(f"#### {title}")
    probs = get_class_probabilities(model, scaled_feat)
    top_label, top_conf = probs[0]
    st.success(f"**{top_label}**")
    st.caption(f"Keyakinan: **{top_conf:.2f}%**")
    with st.expander("Lihat semua etnis"):
        df_probs = pd.DataFrame(probs, columns=["Etnis", "Keyakinan (%)"])
        df_probs["Keyakinan (%)"] = df_probs["Keyakinan (%)"].round(2)
        st.dataframe(df_probs, hide_index=True, use_container_width=True)


@st.cache_resource
def load_artifacts():
    """Memuat model SVM dan Scaler dengan mmap untuk efisiensi memori."""
    return {
        "antro_standalone": {
            "model": joblib.load("models/antro_svm.pkl"),
            "scaler": joblib.load("models/antro_scaler.pkl"),
        },
        "lbp_ycbcr_antro": {
            "model": joblib.load("models/lbp_ycbcr_antro_svm.pkl", mmap_mode="r"),
            "scaler": joblib.load("models/lbp_ycbcr_antro_scaler.pkl"),
        },
        "gabor_lbp_ycbcr": {
            "model": joblib.load("models/gabor_lbp_ycbcr_svm.pkl", mmap_mode="r"),
            "scaler": joblib.load("models/gabor_lbp_ycbcr_scaler.pkl"),
        },
        "all_fusion": {
            "model": joblib.load("models/all_fusion_svm.pkl", mmap_mode="r"),
            "scaler": joblib.load("models/all_fusion_scaler.pkl"),
        },
    }


def crop_single_face(image_bgr, landmarks, margin=0.25):
    """Crop bounding box per wajah individu dengan margin padding 25%."""
    h, w = image_bgr.shape[:2]
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    x1, x2 = max(0, int(x1 - mw)), min(w, int(x2 + mw))
    y1, y2 = max(0, int(y1 - mh)), min(h, int(y2 + mh))

    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, (x1, y1, x2, y2)
    return cv2.resize(face_crop, FACE_SIZE), (x1, y1, x2, y2)


artifacts = load_artifacts()

# =========================================================
# VIDEO PROCESSOR UNTUK LIVE WEBCAM
# =========================================================
class LiveEthnicClassifier(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_results = []  # Menyimpan hasil prediksi antar-frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Memproses klasifikasi setiap 5 frame sekali agar video tetap mulus (tidak lag)
        if self.frame_count % 5 == 0:
            self.last_results = []
            all_lms = get_all_faces_landmarks(img_bgr)

            if all_lms:
                for lms in all_lms:
                    face_crop, bbox = crop_single_face(img_bgr, lms)
                    if face_crop is None:
                        continue

                    local_lms = get_all_faces_landmarks(face_crop)
                    if not local_lms:
                        continue

                    # Ekstraksi fitur ringan (LBP + Antropometri) untuk efisiensi real-time
                    try:
                        antro = extract_antropometri(local_lms[0])
                        lbp_ycbcr = extract_lbp_ycbcr(face_crop)
                        feat_a = np.concatenate([lbp_ycbcr, antro]).reshape(1, -1)

                        scaler = artifacts["lbp_ycbcr_antro"]["scaler"]
                        model = artifacts["lbp_ycbcr_antro"]["model"]
                        scaled_a = scaler.transform(feat_a)

                        probs = get_class_probabilities(model, scaled_a)
                        label, conf = probs[0]
                        self.last_results.append((bbox, f"{label} ({conf:.1f}%)"))
                    except Exception:
                        pass

            gc.collect()

        # Gambar bounding box dan label prediksi di setiap frame
        for (x1, y1, x2, y2), text in self.last_results:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# =========================================================
# KONTROL INPUT
# =========================================================
input_mode = st.radio(
    "Pilih Metode Input:",
    ["Ambil Foto Kamera", "Upload Foto", "Live Webcam"],
    horizontal=True,
)

uploaded = None

if input_mode == "Live Webcam":
    st.info("Klik tombol **START** di bawah ini untuk memulai video streaming langsung.")
    webrtc_streamer(
        key="ethnic-live",
        video_processor_factory=LiveEthnicClassifier,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

elif input_mode == "Ambil Foto Kamera":
    uploaded = st.camera_input("Arahkan wajah lurus ke kamera dan ambil foto")

else:
    uploaded = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])


# =========================================================
# PIPELINE UNTUK FOTO DIAM (UPLOAD / AMBIL KAMERA)
# =========================================================
if uploaded is not None and input_mode != "Live Webcam":
    image = Image.open(uploaded).convert("RGB")
    image_bgr_full = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Foto yang Diproses", width=300)

    with st.spinner("Mendeteksi wajah dengan MediaPipe..."):
        all_faces_lms = get_all_faces_landmarks(image_bgr_full)

    if not all_faces_lms:
        st.error(
            "Wajah tidak terdeteksi. Pastikan pencahayaan cukup terang, wajah tidak tertutup masker/aksesoris, "
            "dan kepala tegak menghadap ke depan."
        )
    else:
        num_faces = len(all_faces_lms)
        st.info(f"Terdeteksi **{num_faces} wajah** pada foto.")

        for idx, lms in enumerate(all_faces_lms, start=1):
            st.divider()
            st.markdown(f"### 👤 Hasil Wajah #{idx}")

            face_crop, _ = crop_single_face(image_bgr_full, lms)
            if face_crop is None:
                st.warning(f"Gagal memotong area wajah #{idx}.")
                continue

            face_display = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            st.image(face_display, caption=f"Crop Wajah #{idx}", width=150)

            local_lms = get_all_faces_landmarks(face_crop)
            if not local_lms:
                st.warning(f"Landmark gagal dideteksi ulang pada crop wajah #{idx}.")
                continue
            lms_target = local_lms[0]

            gc.collect()

            with st.spinner(f"Mengekstrak fitur Wajah #{idx}..."):
                antro = extract_antropometri(lms_target)
                lbp_ycbcr = extract_lbp_ycbcr(face_crop)
                gabor = extract_gabor(face_crop)
                hog_feat = extract_hog(face_crop)

            feat_antro = antro.reshape(1, -1)
            feat_a = np.concatenate([lbp_ycbcr, antro]).reshape(1, -1)
            feat_b = np.concatenate([lbp_ycbcr, gabor]).reshape(1, -1)
            feat_all = np.concatenate([hog_feat, gabor, lbp_ycbcr, antro]).reshape(1, -1)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                scaled_antro = artifacts["antro_standalone"]["scaler"].transform(feat_antro)
                show_prediction_column("Antropometri", artifacts["antro_standalone"]["model"], scaled_antro)

            with col2:
                scaled_a = artifacts["lbp_ycbcr_antro"]["scaler"].transform(feat_a)
                show_prediction_column("LBP + Antro", artifacts["lbp_ycbcr_antro"]["model"], scaled_a)

            with col3:
                scaled_b = artifacts["gabor_lbp_ycbcr"]["scaler"].transform(feat_b)
                show_prediction_column("Gabor + LBP", artifacts["gabor_lbp_ycbcr"]["model"], scaled_b)

            with col4:
                scaled_all = artifacts["all_fusion"]["scaler"].transform(feat_all)
                show_prediction_column("Fusi Lengkap", artifacts["all_fusion"]["model"], scaled_all)

            gc.collect()
