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
    extract_lbp,
    extract_lbp_ycbcr,
    extract_gabor
)


# =====================================================
# SIZE SESUAI TRAINING
# =====================================================

FACE_SIZE = (510,510)



# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    models = {}


    models["LBP"] = {
        "model": joblib.load(
            "models/lbp_svm.pkl"
        ),
        "scaler": joblib.load(
            "models/lbp_scaler.pkl"
        )
    }



    models["Antropometri 10 Multi Angle"] = {
        "model": joblib.load(
            "models/antro_svm.pkl"
        ),
        "scaler": joblib.load(
            "models/antro_scaler.pkl"
        )
    }



    models["Antro Frontal Angry"] = {
        "model": joblib.load(
            "models/antropometri_angry_frontal_svm.pkl"
        ),
        "scaler": joblib.load(
            "models/antropometri_angry_frontal_scaler.pkl"
        )
    }



    models["Antropometri 13 Angry Frontal"] = {
        "model": joblib.load(
            "models/antropometri13_angry_frontal_svm.pkl"
        ),
        "scaler": joblib.load(
            "models/antropometri13_angry_frontal_scaler.pkl"
        )
    }



    models["LBP YCbCr + Antropometri"] = {
        "model": joblib.load(
            "models/lbp_ycbcr_antro_svm.pkl"
        ),
        "scaler": joblib.load(
            "models/lbp_ycbcr_antro_scaler.pkl"
        )
    }



    models["LBP YCbCr + Gabor"] = {
        "model": joblib.load(
            "models/gabor_lbp_ycbcr_svm.pkl"
        ),
        "scaler": joblib.load(
            "models/gabor_lbp_ycbcr_scaler.pkl"
        )
    }


    return models



# =====================================================
# FACE DETECTION
# =====================================================

mp_detection = mp.solutions.face_detection


face_detector = mp_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)



def crop_face(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    result = face_detector.process(
        rgb
    )


    if not result.detections:
        return None



    detection = result.detections[0]


    bbox = detection.location_data.relative_bounding_box


    h,w = image.shape[:2]


    x = int(bbox.xmin*w)
    y = int(bbox.ymin*h)

    bw = int(bbox.width*w)
    bh = int(bbox.height*h)



    # =============================
    # MARGIN MIRIP DATASET
    # =============================

    margin_x = int(bw*0.25)

    margin_y = int(bh*0.35)



    x1 = max(
        0,
        x-margin_x
    )

    y1 = max(
        0,
        y-margin_y
    )


    x2 = min(
        w,
        x+bw+margin_x
    )


    y2 = min(
        h,
        y+bh+margin_y
    )



    crop = image[
        y1:y2,
        x1:x2
    ]


    return crop



# =====================================================
# RESIZE SESUAI TRAINING
# =====================================================

def resize_face(image):

    image = cv2.resize(
        image,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return image



# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(image):

    features = {}



    # LANDMARK 468

    lm468 = detect_landmarks(
        image,
        mode="468"
    )



    # LANDMARK 478

    lm478 = detect_landmarks(
        image,
        mode="478"
    )



    if len(lm468)==0 or len(lm478)==0:
        return None



    lm468 = lm468[0]

    lm478 = lm478[0]



    # ANTRO

    antro10 = extract_antropometri(
        lm468
    )


    antro13 = extract_antropometri13(
        lm478
    )



    # TEXTURE

    lbp = extract_lbp(
        image
    )


    lbp_ycbcr = extract_lbp_ycbcr(
        image
    )


    gabor = extract_gabor(
        image
    )



    features["LBP"] = lbp


    features["Antropometri 10 Multi Angle"] = antro10


    features["Antro Frontal Angry"] = antro10


    features["Antropometri 13 Angry Frontal"] = antro13



    features["LBP YCbCr + Antropometri"] = np.concatenate(
        [
            lbp_ycbcr,
            antro10
        ]
    )



    features["LBP YCbCr + Gabor"] = np.concatenate(
        [
            lbp_ycbcr,
            gabor
        ]
    )



    return features

# =====================================================
# PREDICT SVM
# =====================================================

def predict(
    model_data,
    feature
):

    model = model_data["model"]

    scaler = model_data["scaler"]


    feature = feature.reshape(
        1,-1
    )


    feature_scaled = scaler.transform(
        feature
    )


    prediction = model.predict(
        feature_scaled
    )[0]


    return prediction



# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="Ethnicity Classification",
    layout="wide"
)



st.title(
    "Facial Feature Based Ethnicity Classification"
)


st.write(
    "SVM RBF - Landmark + Texture Feature Pipeline"
)



# =====================================================
# INPUT MODE
# =====================================================

option = st.radio(
    "Pilih Input",
    [
        "Upload Foto",
        "Ambil Foto"
    ]
)



image_bgr = None



# =====================================================
# UPLOAD FOTO
# =====================================================

if option == "Upload Foto":


    file = st.file_uploader(
        "Upload gambar wajah",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if file:


        image = Image.open(
            file
        )


        image_np = np.array(
            image
        )


        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )



# =====================================================
# CAMERA INPUT
# =====================================================

elif option == "Ambil Foto":


    camera = st.camera_input(
        "Ambil foto wajah"
    )


    if camera:


        image = Image.open(
            camera
        )


        image_np = np.array(
            image
        )


        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )



# =====================================================
# PROCESS IMAGE
# =====================================================

if image_bgr is not None:


    st.subheader(
        "Gambar Input"
    )


    st.image(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        ),
        width=300
    )



    # =========================
    # CROP + MARGIN
    # =========================

    crop = crop_face(
        image_bgr
    )



    if crop is None:


        st.error(
            "Wajah tidak terdeteksi"
        )


    else:


        # =========================
        # RESIZE 510x510
        # =========================

        crop = resize_face(
            crop
        )



        st.subheader(
            "Crop Setelah Preprocessing"
        )


        st.image(
            cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            ),
            width=300
        )



        # =========================
        # FEATURE EXTRACTION
        # =========================

        with st.spinner(
            "Ekstraksi fitur..."
        ):


            features = extract_features(
                crop
            )



        if features is None:


            st.error(
                "Landmark wajah gagal dideteksi"
            )


        else:


            st.success(
                "Feature extraction selesai"
            )



            models = load_models()



            st.divider()


            st.subheader(
                "Hasil Prediksi Semua Model"
            )



            # =====================================================
            # GRID 3 x 2
            # =====================================================

            model_names = list(
                models.keys()
            )



            for i in range(
                0,
                len(model_names),
                3
            ):


                cols = st.columns(
                    3
                )


                for col,name in zip(
                    cols,
                    model_names[i:i+3]
                ):


                    with col:


                        result = predict(
                            models[name],
                            features[name]
                        )



                        st.markdown(
                            f"""
                            <div style="
                            border:1px solid #cccccc;
                            border-radius:15px;
                            padding:15px;
                            text-align:center;
                            margin-bottom:15px;
                            ">

                            <h4>
                            {name}
                            </h4>

                            <h2 style="
                            color:#008000;
                            ">
                            {result}
                            </h2>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )