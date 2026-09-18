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

    extract_gabor

)





# =====================================================
# CONFIG
# =====================================================

FACE_SIZE = (510,510)





# =====================================================
# MEDIAPIPE
# =====================================================

mp_detection = mp.solutions.face_detection

mp_selfie = mp.solutions.selfie_segmentation



face_detector = mp_detection.FaceDetection(

    model_selection=1,

    min_detection_confidence=0.5

)



segmenter = mp_selfie.SelfieSegmentation(

    model_selection=1

)







# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource

def load_models():


    models = {}



    # =========================
    # LBP
    # =========================

    models["LBP"] = {

        "model": joblib.load(
            "models/lbp_svm.pkl"
        ),

        "scaler": joblib.load(
            "models/lbp_scaler.pkl"
        )

    }





    # =========================
    # ANTROPOMETRI 10
    # =========================

    models["Antropometri 10 Multi Angle"] = {


        "model": joblib.load(
            "models/antro_svm.pkl"
        ),


        "scaler": joblib.load(
            "models/antro_scaler.pkl"
        )


    }





    # =========================
    # ANTRO FRONTAL ANGRY
    # =========================

    models["Antro Frontal Angry"] = {


        "model": joblib.load(
            "models/antropometri_angry_frontal_svm.pkl"
        ),


        "scaler": joblib.load(
            "models/antropometri_angry_frontal_scaler.pkl"
        )


    }







    # =========================
    # ANTRO 13
    # =========================

    models["Antropometri 13 Angry Frontal"] = {


        "model": joblib.load(
            "models/antropometri13_angry_frontal_svm.pkl"
        ),


        "scaler": joblib.load(
            "models/antropometri13_angry_frontal_scaler.pkl"
        )


    }







    # =========================
    # ANTRO 22 (BARU)
    # =========================

    models["Antropometri 22 Angry Frontal"] = {


        "model": joblib.load(
            "models/antro22_angry_frontal_svm.pkl"
        ),


        "scaler": joblib.load(
            "models/antro22_angry_frontal_scaler.pkl"
        )


    }







    # =========================
    # LBP YCBCR + ANTRO
    # =========================

    models["LBP YCbCr + Antropometri"] = {


        "model": joblib.load(
            "models/lbp_ycbcr_antro_svm.pkl"
        ),


        "scaler": joblib.load(
            "models/lbp_ycbcr_antro_scaler.pkl"
        )


    }







    # =========================
    # LBP YCBCR + GABOR
    # =========================

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
# CROP FACE
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



    x = int(
        bbox.xmin*w
    )


    y = int(
        bbox.ymin*h
    )


    bw = int(
        bbox.width*w
    )


    bh = int(
        bbox.height*h
    )





    # margin sama dataset

    margin_x = int(
        bw*0.25
    )


    margin_y = int(
        bh*0.35
    )





    x1=max(
        0,
        x-margin_x
    )


    y1=max(
        0,
        y-margin_y
    )



    x2=min(
        w,
        x+bw+margin_x
    )


    y2=min(
        h,
        y+bh+margin_y
    )



    crop = image[
        y1:y2,
        x1:x2
    ]



    return crop







# =====================================================
# REMOVE BACKGROUND
# =====================================================

def remove_background(image):


    rgb=cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )



    result = segmenter.process(rgb)



    if result.segmentation_mask is None:

        return image





    mask = result.segmentation_mask



    mask = (

        mask > 0.35

    ).astype(np.uint8)





    kernel=np.ones(

        (5,5),

        np.uint8

    )



    mask=cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel

    )





    foreground=cv2.bitwise_and(

        image,

        image,

        mask=mask

    )



    black=np.zeros_like(image)



    background=cv2.bitwise_and(

        black,

        black,

        mask=1-mask

    )



    output=cv2.add(

        foreground,

        background

    )



    return output







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


    features = {}



    # =========================
    # LANDMARK
    # =========================

    lm468 = detect_landmarks(

        image,

        mode="468"

    )


    lm478 = detect_landmarks(

        image,

        mode="478"

    )




    if len(lm468)==0 or len(lm478)==0:

        return None





    lm468 = lm468[0]

    lm478 = lm478[0]





    # =========================
    # ANTROPOMETRI
    # =========================

    antro10 = extract_antropometri(

        lm468

    )


    antro13 = extract_antropometri13(

        lm478

    )


    antro22 = extract_antropometri22(

        lm478

    )





    # =========================
    # TEXTURE
    # =========================

    lbp = extract_lbp(

        image

    )


    lbp_ycbcr = extract_lbp_ycbcr(

        image

    )


    gabor = extract_gabor(

        image

    )





    # =========================
    # SIMPAN FITUR
    # =========================


    features["LBP"] = lbp



    features["Antropometri 10 Multi Angle"] = antro10



    features["Antro Frontal Angry"] = antro10



    features["Antropometri 13 Angry Frontal"] = antro13



    features["Antropometri 22 Angry Frontal"] = antro22





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
# PREDICTION
# =====================================================

def predict(model_data, feature):


    model = model_data["model"]

    scaler = model_data["scaler"]



    feature = feature.reshape(

        1,-1

    )



    feature_scaled = scaler.transform(

        feature

    )



    result = model.predict(

        feature_scaled

    )[0]



    return result







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
# INPUT
# =====================================================

option = st.radio(

    "Input",

    [

        "Upload Foto",

        "Ambil Foto"

    ]

)



image_bgr = None







# =====================================================
# UPLOAD
# =====================================================

if option=="Upload Foto":


    file = st.file_uploader(

        "Upload wajah",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )


    if file:


        image = Image.open(file)


        img = np.array(image)



        image_bgr = cv2.cvtColor(

            img,

            cv2.COLOR_RGB2BGR

        )







# =====================================================
# CAMERA
# =====================================================

else:


    camera = st.camera_input(

        "Ambil foto"

    )



    if camera:


        image = Image.open(camera)


        img = np.array(image)



        image_bgr = cv2.cvtColor(

            img,

            cv2.COLOR_RGB2BGR

        )



        # =============================
        # FLIP CAMERA MIRROR
        # =============================

        image_bgr = cv2.flip(

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





    crop = crop_face(

        image_bgr

    )




    if crop is None:


        st.error(

            "Wajah tidak terdeteksi"

        )



    else:



        crop = remove_background(

            crop

        )



        crop = resize_face(

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



            features = extract_features(

                crop

            )







        if features is None:


            st.error(

                "Landmark gagal"

            )



        else:



            models = load_models()



            st.success(

                "Prediksi selesai"

            )





            st.subheader(

                "Hasil Semua Model"

            )




            names = list(

                models.keys()

            )





            for i in range(

                0,

                len(names),

                3

            ):



                cols = st.columns(3)





                for col,name in zip(

                    cols,

                    names[i:i+3]

                ):



                    with col:



                        result = predict(

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