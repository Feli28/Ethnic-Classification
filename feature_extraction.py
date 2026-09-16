import cv2
import numpy as np
import mediapipe as mp
from skimage.feature import local_binary_pattern


mp_face_mesh = mp.solutions.face_mesh


face_mesh_468 = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)


face_mesh_478 = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# =====================================================
# LANDMARK
# =====================================================

def detect_landmarks(image_bgr, mode="468"):

    h,w=image_bgr.shape[:2]

    rgb=cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    if mode=="478":
        result=face_mesh_478.process(rgb)

    else:
        result=face_mesh_468.process(rgb)



    if not result.multi_face_landmarks:
        return []



    faces=[]


    for face in result.multi_face_landmarks:

        pts=np.array(
            [
                [
                    lm.x*w,
                    lm.y*h
                ]

                for lm in face.landmark
            ]
        )


        faces.append(pts)


    return faces



# =====================================================
# ANTROPOMETRI 10
# =====================================================

def extract_antropometri(landmarks):

    def dist(a,b):

        return np.linalg.norm(
            landmarks[a]-landmarks[b]
        )


    return np.array(
        [
            dist(133,362),
            dist(33,263),
            dist(98,327),
            dist(168,2),
            dist(4,2),
            dist(61,291),
            dist(2,13),
            dist(234,454),
            dist(10,152),
            dist(64,294)
        ],
        dtype=np.float32
    )



# =====================================================
# ANTROPOMETRI 13
# =====================================================

def extract_antropometri13(landmarks):

    def dist(a,b):

        return np.linalg.norm(
            landmarks[a]-landmarks[b]
        )


    return np.array(
        [
            dist(172,397),
            dist(0,14),
            dist(14,17),
            dist(468,473),
            dist(133,362),
            dist(33,263),
            dist(98,327),
            dist(168,2),
            dist(168,4),
            dist(61,291),
            dist(234,454),
            dist(10,152),
            dist(64,294)
        ],
        dtype=np.float32
    )



# =====================================================
# PURE LBP 2304
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
                cell,
                bins=256,
                range=(0,256)
            )


            hist=hist.astype(np.float32)

            hist/=hist.sum()+1e-6


            features.extend(hist)


    return np.array(
        features,
        dtype=np.float32
    )



# =====================================================
# LBP YCBCR 2368
# =====================================================

def extract_lbp_ycbcr(image):


    ycbcr=cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb
    )


    Y,Cb,Cr=cv2.split(ycbcr)


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
                cell,
                bins=256,
                range=(0,256)
            )


            hist=hist.astype(np.float32)

            hist/=hist.sum()+1e-6


            features.extend(hist)



    hist_cb,_=np.histogram(
        Cb,
        bins=32,
        range=(0,256)
    )


    hist_cr,_=np.histogram(
        Cr,
        bins=32,
        range=(0,256)
    )



    hist_cb=hist_cb.astype(np.float32)
    hist_cr=hist_cr.astype(np.float32)


    hist_cb/=hist_cb.sum()+1e-6
    hist_cr/=hist_cr.sum()+1e-6



    features.extend(hist_cb)
    features.extend(hist_cr)



    return np.array(
        features,
        dtype=np.float32
    )



# =====================================================
# GABOR 32
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