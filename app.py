import streamlit as st
import cv2
import numpy as np
import joblib
import mediapipe as mp

from PIL import Image

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
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

    models = {

        "LBP": {
            "model": joblib.load(
                "models/lbp_svm.pkl"
            ),
            "scaler": joblib.load(
                "models/lbp_scaler.pkl"
            )
        },


        "Antropometri Multi-Angle": {
            "model": joblib.load(
                "models/antropometri_svm.pkl"
            ),
            "scaler": joblib.load(
                "models/antropometri_scaler.pkl"
            )
        },


        "Antro Frontal Angry": {
            "model": joblib.load(
                "models/antro_frontal_svm.pkl"
            ),
            "scaler": joblib.load(
                "models/antro_frontal_scaler.pkl"
            )
        },


        "Antropometri 13 Angry": {
            "model": joblib.load(
                "models/antropometri13_svm.pkl"
            ),
            "scaler": joblib.load(
                "models/antropometri13_scaler.pkl"
            )
        },


        "LBP YCbCr + Antropometri": {
            "model": joblib.load(
                "models/lbp_ycbcr_antro_svm.pkl"
            ),
            "scaler": joblib.load(
                "models/lbp_ycbcr_antro_scaler.pkl"
            )
        },


        "LBP YCbCr + Gabor": {
            "model": joblib.load(
                "models/lbp_ycbcr_gabor_svm.pkl"
            ),
            "scaler": joblib.load(
                "models/lbp_ycbcr_gabor_scaler.pkl"
            )
        }

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

    landmark468 = detect_landmarks(
        image,
        mode="468"
    )


    if len(landmark468)==0:
        return None


    landmark468 = landmark468[0]



    # =========================
    # LANDMARK 478
    # =========================

    landmark478 = detect_landmarks(
        image,
        mode="478"
    )


    if len(landmark478)==0:
        return None


    landmark478 = landmark478[0]



    # =========================
    # ANTROPOMETRI
    # =========================

    antro10 = extract_antropometri(
        landmark468
    )


    antro13 = extract_antropometri13(
        landmark478
    )


    # 10 fitur yang sama
    # dipakai oleh dua model berbeda

    features["Antropometri Multi-Angle"] = antro10


    features["Antro Frontal Angry"] = antro10


    features["Antropometri 13 Angry"] = antro13



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


    features["LBP"] = lbp



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
# SVM PREDICTION
# =====================================================

def predict_svm(
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


    score = model.decision_function(
        feature_scaled
    )


    if len(score.shape)>1:
        score = score[0]


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


    return prediction, ranking



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
    "SVM RBF - Landmark + Texture Feature Classification"
)



input_mode = st.radio(
    "Pilih Input:",
    [
        "Upload",
        "Take Photo",
        "Webcam"
    ]
)



image=None



# =====================================================
# UPLOAD
# =====================================================

if input_mode=="Upload":

    file=st.file_uploader(
        "Upload gambar wajah",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if file:

        image=Image.open(
            file
        )



# =====================================================
# CAMERA
# =====================================================

elif input_mode=="Take Photo":


    photo=st.camera_input(
        "Ambil foto"
    )


    if photo:

        image=Image.open(
            photo
        )



# =====================================================
# WEBCAM
# =====================================================


elif input_mode=="Webcam":


    class VideoProcessor(
        VideoProcessorBase
    ):

        def __init__(self):

            self.frame=None


        def recv(self,frame):

            img=frame.to_ndarray(
                format="bgr24"
            )

            self.frame=img

            return frame



    ctx=webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor
    )


    if ctx.video_processor:

        image=ctx.video_processor.frame



# =====================================================
# PROCESS
# =====================================================


if image is not None:


    if not isinstance(image,np.ndarray):

        image_np=np.array(
            image
        )


        image_bgr=cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )


    else:

        image_bgr=image



    st.image(
        cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        ),
        width=300
    )



    face=crop_face(
        image_bgr
    )



    if face is None:

        st.error(
            "Wajah tidak terdeteksi"
        )


    else:


        st.success(
            "Wajah berhasil dideteksi"
        )


        st.image(
            cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            ),
            width=300
        )



        with st.spinner(
            "Ekstraksi fitur..."
        ):


            features=extract_features(
                face
            )



        if features is None:

            st.error(
                "Landmark gagal dideteksi"
            )


        else:


            models=load_models()


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
                    "Decision Score:"
                )


                for cls,score in ranking[:3]:

                    st.write(
                        f"{cls}: {score:.4f}"
                    )


                st.divider()