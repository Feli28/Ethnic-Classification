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
    extract_gabor,
    extract_antropometri22
)


FACE_SIZE=(510,510)


# =====================================================
# MEDIAPIPE
# =====================================================

mp_detection=mp.solutions.face_detection
mp_selfie=mp.solutions.selfie_segmentation


face_detector=mp_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


segmenter=mp_selfie.SelfieSegmentation(
    model_selection=1
)



# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    models={}

    models["LBP"]={
        "model":joblib.load("models/lbp_svm.pkl"),
        "scaler":joblib.load("models/lbp_scaler.pkl")
    }


    models["Antropometri 10 Multi Angle"]={
        "model":joblib.load("models/antro_svm.pkl"),
        "scaler":joblib.load("models/antro_scaler.pkl")
    }


    models["Antro Frontal Angry"]={
        "model":joblib.load(
            "models/antropometri_angry_frontal_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/antropometri_angry_frontal_scaler.pkl"
        )
    }


    models["Antropometri 13 Angry Frontal"]={
        "model":joblib.load(
            "models/antropometri13_angry_frontal_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/antropometri13_angry_frontal_scaler.pkl"
        )
    }


    models["LBP YCbCr + Antropometri"]={
        "model":joblib.load(
            "models/lbp_ycbcr_antro_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/lbp_ycbcr_antro_scaler.pkl"
        )
    }


    models["LBP YCbCr + Gabor"]={
        "model":joblib.load(
            "models/gabor_lbp_ycbcr_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/gabor_lbp_ycbcr_scaler.pkl"
        )
    }

    models["Antropometri 22 Angry Frontal"]={

    "model":joblib.load(
        "models/antro22_angry_frontal_svm.pkl"
    ),

    "scaler":joblib.load(
        "models/antro22_angry_frontal_scaler.pkl"
    )

}


    return models



# =====================================================
# CROP FACE + MARGIN
# =====================================================

def crop_face(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    result = face_detector.process(rgb)


    if not result.detections:
        return None



    detection = result.detections[0]


    bbox = detection.location_data.relative_bounding_box


    h,w = image.shape[:2]


    x = int(bbox.xmin*w)
    y = int(bbox.ymin*h)

    bw = int(bbox.width*w)
    bh = int(bbox.height*h)



    # tambah margin
    margin = int(max(bw,bh)*0.35)



    cx = x + bw//2
    cy = y + bh//2



    # ambil sisi terbesar supaya kotak
    size = max(bw,bh) + margin*2



    x1 = cx - size//2
    y1 = cy - size//2

    x2 = cx + size//2
    y2 = cy + size//2



    # batas gambar
    x1=max(0,x1)
    y1=max(0,y1)

    x2=min(w,x2)
    y2=min(h,y2)



    crop=image[
        y1:y2,
        x1:x2
    ]


    return crop

# =====================================================
# RESIZE
# =====================================================

def resize_face(image):

    return cv2.resize(
        image,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA
    )


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(image):

    features={}


    lm468=detect_landmarks(
        image,
        mode="468"
    )


    lm478=detect_landmarks(
        image,
        mode="478"
    )


    if len(lm468)==0 or len(lm478)==0:

        return None


    lm468=lm468[0]

    lm478=lm478[0]



    # =========================
    # ANTROPOMETRI
    # =========================

    antro10=extract_antropometri(
        lm468
    )


    antro13=extract_antropometri13(
        lm478
    )


    antro22=extract_antropometri22(
        lm478
    )



    # =========================
    # TEXTURE
    # =========================

    lbp=extract_lbp(
        image
    )


    lbp_ycbcr=extract_lbp_ycbcr(
        image
    )


    gabor=extract_gabor(
        image
    )



    # =========================
    # SIMPAN FEATURE
    # =========================

    features["LBP"]=lbp


    features["Antropometri 10 Multi Angle"]=antro10


    features["Antro Frontal Angry"]=antro10


    features["Antropometri 13 Angry Frontal"]=antro13


    features["Antropometri 22 Angry Frontal"]=antro22



    features["LBP YCbCr + Antropometri"]=np.concatenate(
        [
            lbp_ycbcr,
            antro10
        ]
    )


    features["LBP YCbCr + Gabor"]=np.concatenate(
        [
            lbp_ycbcr,
            gabor
        ]
    )


    return features

# =====================================================
# PREDICT
# =====================================================

def predict(model_data,feature):

    model=model_data["model"]
    scaler=model_data["scaler"]


    feature=feature.reshape(
        1,-1
    )


    feature_scaled=scaler.transform(
        feature
    )


    return model.predict(
        feature_scaled
    )[0]



# =====================================================
# STREAMLIT UI
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



option=st.radio(
    "Input",
    [
        "Upload Foto",
        "Ambil Foto"
    ]
)


# =====================================================
# INPUT IMAGE
# =====================================================

image_bgr=None


# =============================
# UPLOAD FOTO
# =============================

if option=="Upload Foto":

    file=st.file_uploader(
        "Upload wajah",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if file:

        image=Image.open(file)

        img=np.array(image)


        image_bgr=cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )



# =============================
# AMBIL FOTO (KAMERA DEPAN)
# =============================

else:

    camera=st.camera_input(
        "Ambil foto"
    )


    if camera:

        image=Image.open(camera)

        img=np.array(image)


        image_bgr=cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )


        # =========================
        # FLIP HORIZONTAL
        # kamera depan mirror
        # samakan dengan dataset
        # =========================

        image_bgr=cv2.flip(
            image_bgr,
            1
        )

# =====================================================
# PROCESS
# =====================================================

if image_bgr is not None:


    st.subheader(
        "Input"
    )


    st.image(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        ),
        width=300
    )



    crop=crop_face(
        image_bgr
    )



    if crop is None:

        st.error(
            "Wajah tidak terdeteksi"
        )


    else:


        crop=remove_background(
            crop
        )


        crop=resize_face(
            crop
        )


        st.subheader(
            "Preprocessing 510x510"
        )


        st.image(
            cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            ),
            width=300
        )



        with st.spinner(
            "Ekstraksi fitur..."
        ):


            features=extract_features(
                crop
            )



        if features is None:

            st.error(
                "Landmark gagal"
            )


        else:

            models=load_models()


            st.success(
                "Prediksi selesai"
            )


            st.subheader(
                "Hasil Semua Model"
            )



            names=list(
                models.keys()
            )


            for i in range(0,len(names),3):

                cols=st.columns(3)


                for col,name in zip(
                    cols,
                    names[i:i+3]
                ):

                    with col:

                        result=predict(
                            models[name],
                            features[name]
                        )


                        st.markdown(
                            f"""
                            <div style="
                            border:1px solid #ddd;
                            border-radius:15px;
                            padding:15px;
                            text-align:center;
                            ">

                            <h4>{name}</h4>

                            <h2 style="color:green">
                            {result}
                            </h2>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )