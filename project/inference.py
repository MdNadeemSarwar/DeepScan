import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as PILImage
import os

model = tf.keras.models.load_model('model/xception_deepfake_image_5o.h5')

IMAGE_SIZE = (224, 224)
MAX_SEQ_LENGTH = 20
BATCH_SIZE = 32
FRAME_SAMPLE_RATE = 10
feature_extractor = tf.keras.applications.Xception(weights="imagenet", include_top=False, pooling="avg")

detector = MTCNN()

def check_ai_filename(filename):
    filename = os.path.basename(filename).lower()
    ai_patterns = [
        'gemini_generated_image', 'dall-e', 'dalle',
        'generated_image', 'ai_generated', 'chatgpt',
        'midjourney', 'stable_diffusion', 'firefly',
        'bing_image_creator', 'image_fx', 'ideogram', 'imagefx',
    ]
    for pattern in ai_patterns:
        if pattern in filename:
            return True, f"AI filename: {pattern}"
    return False, "Normal filename"

def check_ai_metadata(image_path):
    try:
        img = PILImage.open(image_path)
        ai_keywords = [
            'gemini', 'dall-e', 'openai', 'midjourney',
            'stable diffusion', 'firefly', 'adobe ai',
            'generated', 'ai generated', 'synthetic',
            'artificially', 'neural', 'diffusion',
            'imagen', 'kandinsky', 'playground', 'bing image'
        ]
        info_str = str(img.info).lower()
        for keyword in ai_keywords:
            if keyword in info_str:
                return True, f"AI metadata: {keyword}"
        try:
            exif_data = img._getexif()
            if exif_data:
                exif_str = str(exif_data).lower()
                for keyword in ai_keywords:
                    if keyword in exif_str:
                        return True, f"AI EXIF: {keyword}"
                software = exif_data.get(305, '').lower()
                for keyword in ai_keywords:
                    if keyword in software:
                        return True, f"AI software: {software}"
        except:
            pass
        return False, "No AI metadata"
    except Exception as e:
        return False, str(e)

# def check_gemini_watermark(image_path):
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             return False
#         h, w = image.shape[:2]
#         corner = image[int(h * 0.85):h, int(w * 0.85):w]
#         gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
#         white_ratio = np.sum(thresh == 255) / thresh.size
#         mean_brightness = np.mean(gray)
#         if mean_brightness < 120 and white_ratio > 0.01 and white_ratio < 0.25:
#             return True
#         return False
#     except:
#         return False

def check_gemini_watermark(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False

        # Template logo load karo
        logo_path = 'model/gimini_logo.png'
        if not os.path.exists(logo_path):
            return False
            
        template = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        th, tw = template.shape

        # Multiple scales pe try karo
        scales = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        
        for scale in scales:
            new_w = int(tw * scale)
            new_h = int(th * scale)
            
            if new_w < 5 or new_h < 5:
                continue
            if new_w > w or new_h > h:
                continue
                
            resized = cv2.resize(template, (new_w, new_h))
            
            # Template matching
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # 0.55 threshold — logo milne ka confidence
            if max_val > 0.55:
                print(f"Gemini logo detected! Confidence: {max_val:.2f} at scale {scale}")
                return True

        return False
    except Exception as e:
        print(f"Watermark check error: {e}")
        return False

def fft_ai_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    region = magnitude[cy-30:cy+30, cx-30:cx+30]
    outer = magnitude.copy()
    outer[cy-50:cy+50, cx-50:cx+50] = 0
    high_freq_energy = np.mean(outer)
    low_freq_energy = np.mean(region)
    if low_freq_energy == 0:
        return 0.5
    ratio = high_freq_energy / low_freq_energy
    return float(max(0, min(1, 1 - (ratio / 0.15))))

def fft_audio_score(audio_path):
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, duration=5)
        fft = np.fft.fft(y)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        mag = magnitude[freqs > 0]
        if len(mag) == 0:
            return 0.5
        mean_mag = np.mean(mag)
        max_mag = np.max(mag)
        std_mag = np.std(mag)
        peak_ratio = max_mag / (mean_mag + 1e-10)
        uniformity = std_mag / (mean_mag + 1e-10)
        return float(min(1, peak_ratio / 100) * 0.5 + max(0, 1 - uniformity / 10) * 0.5)
    except:
        return 0.5

def extract_frames_from_video(video_path, sample_rate=FRAME_SAMPLE_RATE):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % sample_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames

def detect_and_crop_faces(frame):
    faces = []
    detections = detector.detect_faces(frame)
    for detection in detections:
        x, y, width, height = detection['box']
        face = frame[y:y+height, x:x+width]
        face = cv2.resize(face, IMAGE_SIZE)
        faces.append(face)
    return faces

def preprocess_faces(faces):
    face_features = np.zeros((len(faces), *IMAGE_SIZE, 3))
    for i, face in enumerate(faces):
        face_features[i] = tf.keras.applications.xception.preprocess_input(face)
    return face_features

def detect_faces_parallel(frames):
    with ThreadPoolExecutor() as executor:
        results = executor.map(detect_and_crop_faces, frames)
    all_faces = []
    for faces in results:
        all_faces.extend(faces)
    return all_faces

def predict_from_image(image_path, original_filename=None):
    image = cv2.imread(image_path)

    # Step 1 — Original filename check
    check_name = original_filename if original_filename else image_path
    is_ai_file, file_reason = check_ai_filename(check_name)
    if is_ai_file:
        print(f"AI detected via filename: {file_reason}")
        return 'FAKE', 0.99

    # Step 2 — Metadata check
    is_ai_meta, meta_reason = check_ai_metadata(image_path)
    if is_ai_meta:
        print(f"AI detected via metadata: {meta_reason}")
        return 'FAKE', 0.99

    # Step 3 — Gemini watermark check
    if check_gemini_watermark(image_path):
        print("Gemini watermark detected!")
        return 'FAKE', 0.97

    # Step 4 — FFT analysis
    fft_score = fft_ai_score(image)

    # Step 5 — Face detection + XceptionNet
    faces = detect_and_crop_faces(image)

    if not faces:
        label = 'FAKE' if fft_score >= 0.5 else 'REAL'
        confidence = fft_score if fft_score >= 0.5 else (1 - fft_score)
        return label, round(confidence, 4)

    preprocessed_faces = preprocess_faces(faces)
    predictions = model.predict(np.array(preprocessed_faces))
    xception_score = float(np.mean(predictions))
    combined_score = (xception_score * 0.7) + (fft_score * 0.3)
    label = 'FAKE' if combined_score >= 0.5 else 'REAL'
    return label, round(combined_score, 4)

def predict_fake_real(video_path):
    frames = extract_frames_from_video(video_path)
    all_faces = detect_faces_parallel(frames)
    fft_scores = [fft_ai_score(f) for f in frames[:5]]
    avg_fft = float(np.mean(fft_scores)) if fft_scores else 0.5

    if not all_faces:
        label = 'FAKE' if avg_fft >= 0.5 else 'REAL'
        confidence = avg_fft if avg_fft >= 0.5 else (1 - avg_fft)
        return label, round(confidence, 4)

    all_faces = all_faces[:MAX_SEQ_LENGTH]
    preprocessed_faces = preprocess_faces(all_faces)
    predictions = []
    for i in range(0, len(preprocessed_faces), BATCH_SIZE):
        batch = preprocessed_faces[i:i+BATCH_SIZE]
        predictions.extend(model.predict(np.array(batch)))
    xception_score = float(np.mean(predictions))
    combined_score = (xception_score * 0.7) + (avg_fft * 0.3)
    label = 'FAKE' if combined_score >= 0.5 else 'REAL'
    return label, round(combined_score, 4)

if __name__ == "__main__":
    video_path = 'test.mp4'
    result = predict_fake_real(video_path)
    print(f'The video is {result}')
    image_path = 'test.jpg'
    result = predict_from_image(image_path)
    print(f'The image is {result}')