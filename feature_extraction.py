import cv2
import numpy as np
import mediapipe as mp
from skimage.feature import local_binary_pattern


# =====================================================
# MEDIAPIPE FACE MESH
# =====================================================

mp_face_mesh = mp.solutions.face_mesh


# 468 landmark
face_mesh_468 = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)


# 478 landmark (iris)
face_mesh_478 = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)



# =====================================================
# LANDMARK EXTRACTION
# =====================================================

def detect_landmarks(image_bgr, mode="468"):

    h, w = image_bgr.shape[:2]


    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    if mode == "478":
        result = face_mesh_478.process(rgb)

    else:
        result = face_mesh_468.process(rgb)



    if not result.multi_face_landmarks:
        return []



    faces=[]


    for face in result.multi_face_landmarks:

        points=np.array(
            [
                [
                    lm.x*w,
                    lm.y*h
                ]

                for lm in face.landmark
            ],
            dtype=np.float32
        )


        faces.append(points)



    return faces



# =====================================================
# ANTROPOMETRI 10 FITUR
# 468 LANDMARK
# =====================================================

def extract_antropometri(landmarks):


    def dist(a,b):

        return np.linalg.norm(
            landmarks[a]-landmarks[b]
        )


    return np.array(

        [

            # Intercanthal Width
            dist(133,362),

            # Biocular Width
            dist(33,263),

            # Nasal Width
            dist(98,327),

            # Nasal Height
            dist(168,2),

            # Nasal Length
            dist(4,2),

            # Mouth Width
            dist(61,291),

            # Philtrum Length
            dist(2,13),

            # Face Width
            dist(234,454),

            # Face Height
            dist(10,152),

            # Nasal Parenthesis Width
            dist(64,294)

        ],

        dtype=np.float32

    )



# =====================================================
# ANTROPOMETRI 13 FITUR
# 478 LANDMARK
# =====================================================

def extract_antropometri13(landmarks):


    def dist(a,b):

        return np.linalg.norm(
            landmarks[a]-landmarks[b]
        )



    return np.array(

        [

            # Mandible Width
            dist(172,397),

            # Upper Vermilion Height
            dist(0,14),

            # Lower Vermilion Height
            dist(14,17),

            # Interpupillary Distance
            dist(468,473),

            # Intercanthal Width
            dist(133,362),

            # Biocular Width
            dist(33,263),

            # Nasal Width
            dist(98,327),

            # Nasal Height
            dist(168,2),

            # Nasal Length
            dist(168,4),

            # Mouth Width
            dist(61,291),

            # Face Width
            dist(234,454),

            # Face Height
            dist(10,152),

            # Nasal Parenthesis Width
            dist(64,294)

        ],

        dtype=np.float32

    )



# =====================================================
# PURE LBP 2304 FEATURES
# =====================================================

def extract_lbp(image):


    gray=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    lbp=local_binary_pattern(
        gray,
        8,
        1,
        method="default"
    )


    h,w=gray.shape


    features=[]


    for r in range(3):

        for c in range(3):


            cell=lbp[
                r*h//3:(r+1)*h//3,
                c*w//3:(c+1)*w//3
            ]



            hist,_=np.histogram(

                cell.ravel(),

                bins=256,

                range=(0,256)

            )


            hist=hist.astype(
                np.float32
            )


            hist/=(
                hist.sum()+1e-6
            )


            features.extend(hist)



    return np.array(
        features,
        dtype=np.float32
    )



# =====================================================
# LBP YCBCR 2368 FEATURES
# =====================================================

def extract_lbp_ycbcr(image):


    ycbcr=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb
    )


    Y,Cb,Cr=cv2.split(
        ycbcr
    )



    lbp=local_binary_pattern(

        Y,

        8,

        1,

        method="default"

    )



    h,w=Y.shape


    features=[]



    for r in range(3):

        for c in range(3):


            cell=lbp[

                r*h//3:(r+1)*h//3,

                c*w//3:(c+1)*w//3

            ]



            hist,_=np.histogram(

                cell.ravel(),

                bins=256,

                range=(0,256)

            )


            hist=hist.astype(
                np.float32
            )


            hist/=(
                hist.sum()+1e-6
            )


            features.extend(hist)



    # Cb histogram

    hist_cb,_=np.histogram(

        Cb.ravel(),

        bins=32,

        range=(0,256)

    )



    # Cr histogram

    hist_cr,_=np.histogram(

        Cr.ravel(),

        bins=32,

        range=(0,256)

    )



    hist_cb=hist_cb.astype(
        np.float32
    )

    hist_cr=hist_cr.astype(
        np.float32
    )



    hist_cb/=(
        hist_cb.sum()+1e-6
    )

    hist_cr/=(
        hist_cr.sum()+1e-6
    )



    features.extend(hist_cb)

    features.extend(hist_cr)



    return np.array(

        features,

        dtype=np.float32

    )



# =====================================================
# GABOR 32 FEATURES
# =====================================================

def extract_gabor(image):


    gray=cv2.cvtColor(

        image,

        cv2.COLOR_BGR2GRAY

    )


    features=[]



    for sigma in [2,4]:

        for theta in [

            0,

            np.pi/4,

            np.pi/2,

            3*np.pi/4

        ]:

            for lambd in [4,12]:


                kernel=cv2.getGaborKernel(

                    (31,31),

                    sigma,

                    theta,

                    lambd,

                    0.5,

                    0

                )


                filtered=cv2.filter2D(

                    gray,

                    cv2.CV_32F,

                    kernel

                )


                features.append(

                    np.mean(filtered)

                )


                features.append(

                    np.std(filtered)

                )



    return np.array(

        features,

        dtype=np.float32

    )