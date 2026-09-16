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
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    models={}


    models["LBP"]={
        "model":joblib.load(
            "models/lbp_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/lbp_scaler.pkl"
        )
    }


    models["Antropometri"]={
        "model":joblib.load(
            "models/antropometri_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/antropometri_scaler.pkl"
        )
    }


    models["Antro Frontal"]={
        "model":joblib.load(
            "models/antro_frontal_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/antro_frontal_scaler.pkl"
        )
    }


    models["Antropometri 13"]={
        "model":joblib.load(
            "models/antropometri13_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/antropometri13_scaler.pkl"
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
            "models/lbp_ycbcr_gabor_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/lbp_ycbcr_gabor_scaler.pkl"
        )
    }


    return models



# =====================================================
# FACE DETECTION + CROP
# =====================================================

mp_face_detection = mp.solutions.face_detection


face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)



def crop_face(image):

    rgb=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    result=face_detector.process(
        rgb
    )


    if not result.detections:
        return None



    detection=result.detections[0]


    bbox=detection.location_data.relative_bounding_box


    h,w=image.shape[:2]


    x=int(bbox.xmin*w)
    y=int(bbox.ymin*h)

    bw=int(bbox.width*w)
    bh=int(bbox.height*h)



    x=max(0,x)
    y=max(0,y)


    crop=image[
        y:y+bh,
        x:x+bw
    ]


    return crop



# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(image):


    features={}


    # =========================
    # LANDMARK 468
    # =========================

    landmark468=detect_landmarks(
        image,
        mode="468"
    )


    if len(landmark468)==0:
        return None



    landmark468=landmark468[0]



    # =========================
    # LANDMARK 478
    # =========================

    landmark478=detect_landmarks(
        image,
        mode="478"
    )


    if len(landmark478)==0:
        return None



    landmark478=landmark478[0]



    # =========================
    # ANTRO
    # =========================

    antro10=extract_antropometri(
        landmark468
    )


    antro13=extract_antropometri13(
        landmark478
    )



    features["Antropometri"]=antro10


    features["Antro Frontal"]=antro10


    features["Antropometri 13"]=antro13



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



    features["LBP"]=lbp



    # Fusion

    features[
        "LBP YCbCr + Antropometri"
    ]=np.concatenate(
        [
            lbp_ycbcr,
            antro10
        ]
    )



    features[
        "LBP YCbCr + Gabor"
    ]=np.concatenate(
        [
            lbp_ycbcr,
            gabor
        ]
    )



    return features



# =====================================================
# PREDICT SVM
# =====================================================

def predict_svm(
    model_data,
    feature
):


    model=model_data["model"]

    scaler=model_data["scaler"]



    feature=feature.reshape(
        1,-1
    )



    feature_scaled=scaler.transform(
        feature
    )



    prediction=model.predict(
        feature_scaled
    )[0]



    score=model.decision_function(
        feature_scaled
    )



    if len(score.shape)>1:
        score=score[0]



    ranking=list(
        zip(
            model.classes_,
            score
        )
    )


    ranking.sort(
        key=lambda x:x[1],
        reverse=True
    )


    return prediction,ranking



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



uploaded_file=st.file_uploader(
    "Upload gambar wajah",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



if uploaded_file:


    image=Image.open(
        uploaded_file
    )


    image_np=np.array(
        image
    )


    image_bgr=cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )



    st.image(
        image,
        width=300
    )



    # Crop wajah

    face_crop=crop_face(
        image_bgr
    )



    if face_crop is None:


        st.error(
            "Wajah tidak terdeteksi"
        )


    else:


        st.success(
            "Wajah berhasil dideteksi"
        )


        st.image(
            cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            ),
            width=300
        )



        models=load_models()



        with st.spinner(
            "Ekstraksi fitur..."
        ):


            features=extract_features(
                face_crop
            )



        if features is None:


            st.error(
                "Landmark wajah gagal dideteksi"
            )


        else:


            st.success(
                "Feature extraction selesai"
            )



            st.divider()



            for name in models:


                prediction,ranking=predict_svm(
                    models[name],
                    features[name]
                )


                st.subheader(
                    name
                )


                st.write(
                    "Prediksi:",
                    prediction
                )


                st.write(
                    "Top Decision Score:"
                )


                for cls,score in ranking[:3]:

                    st.write(
                        f"{cls}: {score:.4f}"
                    )


                st.divider()