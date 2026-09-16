"""
Modul ekstraksi fitur - disesuaikan PERSIS dengan pipeline training Kaggle:
- Pure LBP Grid 3x3 (Grayscale - 2304 fitur)
- LBP-YCbCr (2368 fitur)
- Antropometri Standar (10 fitur)
- Antropometri 13 Fitur (478 landmark dengan refine_landmarks=True)
- Gabor (32 fitur)
"""

import cv2
import numpy as np
import mediapipe as mp
from skimage.feature import local_binary_pattern

# Ukuran standar citra sesuai training pipeline
FACE_SIZE = (510, 510)

mp_face_mesh = mp.solutions.face_mesh

# refine_landmarks=True wajib aktif agar menghasilkan 478 landmark (termasuk iris pupil)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def get_all_faces_landmarks(image_bgr):
    """
    Mendeteksi landmark semua wajah.
    Return: list of numpy arrays, masing-masing berukuran (478, 2).
    Jika tidak ada wajah, return list kosong [].
    """
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return []

    all_landmarks = []
    for face_lms in result.multi_face_landmarks:
        pts = np.array([[p.x * w, p.y * h] for p in face_lms.landmark])
        all_landmarks.append(pts)

    return all_landmarks


def extract_antropometri(landmarks):
    """
    10 Fitur Antropometri Standar (FITUR_ANTROPOMETRI_10FITUR.csv).
    """
    def euclidean(p1, p2):
        x1, y1 = landmarks[p1]
        x2, y2 = landmarks[p2]
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    return np.array([
        euclidean(133, 362),  # Intercanthal_Width
        euclidean(33, 263),   # Biocular_Width
        euclidean(98, 327),   # Nasal_Width
        euclidean(168, 2),    # Nasal_Height
        euclidean(4, 2),      # Nasal_Length
        euclidean(61, 291),   # Mouth_Width
        euclidean(2, 13),     # Philtrum_Length
        euclidean(234, 454),  # Face_Width
        euclidean(10, 152),   # Face_Height
        euclidean(64, 294),   # Nasal_Parenthesis_Width
    ])


def extract_antropometri13(landmarks):
    """
    13 Fitur Antropometri (ANTHROPOMETRY_13_FEATURES.csv).
    Menggunakan 478 landmark (termasuk iris pupil 468 & 473).
    """
    def euclidean(p1, p2):
        x1, y1 = landmarks[p1]
        x2, y2 = landmarks[p2]
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    return np.array([
        euclidean(172, 397),  # Mandible_Width
        euclidean(0, 14),     # Upper_Vermilion_Height
        euclidean(14, 17),    # Lower_Vermilion_Height
        euclidean(468, 473),  # Interpupillary_Distance (Iris)
        euclidean(133, 362),  # Intercanthal_Width
        euclidean(33, 263),   # Biocular_Width
        euclidean(98, 327),   # Nasal_Width
        euclidean(168, 2),    # Nasal_Height
        euclidean(168, 4),    # Nasal_Length
        euclidean(61, 291),   # Mouth_Width
        euclidean(234, 454),  # Face_Width
        euclidean(10, 152),   # Face_Height
        euclidean(64, 294),   # Nasal_Parenthesis_Width
    ])


def extract_pure_lbp_grid(image_bgr, grid_rows=3, grid_cols=3, radius=1, n_points=8):
    """
    Pure LBP Grid 3x3 Grayscale (PURE_LBP_GRID3x3_2304_FEATURES.csv).
    Total: 9 sel x 256 bin histogram = 2304 fitur.
    """
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp_matrix = local_binary_pattern(image_gray, n_points, radius, method='default')

    h, w = image_gray.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    lbp_feats = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            crop_cell = lbp_matrix[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            hist, _ = np.histogram(crop_cell.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-6)
            lbp_feats.append(hist)

    return np.concatenate(lbp_feats)


def extract_lbp_ycbcr(image_bgr, grid_rows=3, grid_cols=3, color_bins=32, radius=1, n_points=8):
    """
    LBP Grid 3x3 di kanal Y (2304 fitur) + histogram Cb & Cr (64 fitur) -> total 2368 fitur.
    """
    image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y_channel, Cb_channel, Cr_channel = cv2.split(image_ycbcr)

    lbp_matrix = local_binary_pattern(Y_channel, n_points, radius, method='default')

    h, w = Y_channel.shape
    cell_h, cell_w = h // grid_rows, w // grid_cols

    lbp_feats = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            crop_cell = lbp_matrix[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            hist_lbp, _ = np.histogram(crop_cell.ravel(), bins=256, range=(0, 256))
            hist_lbp = hist_lbp.astype(np.float32)
            hist_lbp /= (hist_lbp.sum() + 1e-6)
            lbp_feats.append(hist_lbp)
    lbp_feats = np.concatenate(lbp_feats)

    hist_cb, _ = np.histogram(Cb_channel.ravel(), bins=color_bins, range=(0, 256))
    hist_cb = hist_cb.astype(np.float32)
    hist_cb /= (hist_cb.sum() + 1e-6)

    hist_cr, _ = np.histogram(Cr_channel.ravel(), bins=color_bins, range=(0, 256))
    hist_cr = hist_cr.astype(np.float32)
    hist_cr /= (hist_cr.sum() + 1e-6)

    return np.concatenate([lbp_feats, hist_cb, hist_cr])


def extract_gabor(image_bgr):
    """
    32 fitur Gabor (mean, std) x 16 kombinasi parameter filter.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    lambdas = [4, 12]
    sigmas = [2, 4]
    gamma, psi = 0.5, 0

    features = []
    for sigma in sigmas:
        for theta in thetas:
            for lambd in lambdas:
                kernel = cv2.getGaborKernel((31, 31), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
    return np.array(features)