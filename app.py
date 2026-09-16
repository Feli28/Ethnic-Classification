import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import mediapipe as mp

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)


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



    models["Antropometri 10 Multi Angle"]={
        "model":joblib.load(
            "models/antro_svm.pkl"
        ),
        "scaler":joblib.load(
            "models/antro_scaler.pkl"
        )
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


    return models



# =====================================================
# FACE DETECTION CROP
# =====================================================

mp_detection = mp.solutions.face_detection


face_detector = mp_detection.FaceDetection(
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
        return None,None



    detection=result.detections[0]


    bbox=detection.location_data.relative_bounding_box


    h,w=image.shape[:2]


    x=int(bbox.xmin*w)
    y=int(bbox.ymin*h)

    bw=int(bbox.width*w)
    bh=int(bbox.height*h)



    # =========================
    # TAMBAH MARGIN
    # =========================

    margin_x=int(bw*0.25)
    margin_y=int(bh*0.35)



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


    crop=image[
        y1:y2,
        x1:x2
    ]



    box=(
        x1,
        y1,
        x2,
        y2
    )


    return crop,box



# =====================================================
# DRAW BOX
# =====================================================

def draw_box(image,box):

    if box is None:
        return image


    x1,y1,x2,y2=box


    img=image.copy()


    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )


    return img



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



    antro10=extract_antropometri(
        lm468
    )


    antro13=extract_antropometri13(
        lm478
    )



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


    features["Antropometri 10 Multi Angle"]=antro10


    features["Antro Frontal Angry"]=antro10


    features["Antropometri 13 Angry Frontal"]=antro13



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


    prediction=model.predict(
        feature_scaled
    )[0]


    return prediction



# =====================================================
# WEBCAM PROCESSOR
# =====================================================


class VideoProcessor(VideoProcessorBase):

    def recv(self,frame):

        img=frame.to_ndarray(
            format="bgr24"
        )


        crop,box=crop_face(
            img
        )


        img=draw_box(
            img,
            box
        )


        return frame.from_ndarray(
            img,
            format="bgr24"
        )



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
    "SVM RBF - Landmark + Texture Feature"
)



option=st.radio(
    "Input gambar",
    [
        "Upload Foto",
        "Webcam",
        "Ambil Foto"
    ]
)



image_bgr=None



# =============================
# UPLOAD
# =============================

if option=="Upload Foto":


    file=st.file_uploader(
        "Upload wajah",
        type=[
            "jpg",
            "png",
            "jpeg"
        ]
    )


    if file:

        img=Image.open(file)

        img_np=np.array(img)


        image_bgr=cv2.cvtColor(
            img_np,
            cv2.COLOR_RGB2BGR
        )



# =============================
# CAMERA
# =============================

elif option=="Ambil Foto":


    camera=st.camera_input(
        "Ambil foto"
    )


    if camera:

        img=Image.open(camera)

        img_np=np.array(img)

        image_bgr=cv2.cvtColor(
            img_np,
            cv2.COLOR_RGB2BGR
        )



# =============================
# WEBCAM
# =============================

else:


    webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor
    )



# =====================================================
# PREDIKSI
# =====================================================


if image_bgr is not None:


    crop,box=crop_face(
        image_bgr
    )



    if crop is None:

        st.error(
            "Wajah tidak ditemukan"
        )


    else:


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
                "Landmark gagal dideteksi"
            )


        else:


            models=load_models()



            st.success(
                "Prediksi selesai"
            )


            for name in models:


                result=predict(
                    models[name],
                    features[name]
                )


                st.subheader(
                    name
                )


                st.success(
                    result
                )