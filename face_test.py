import cv2
import mediapipe as mp
import math
import time
import winsound
import threading

# -----------------------------
# 0. making the beep sound
# -----------------------------
def play_beep(freq=1000, duration=100):
    def _beep():
        try:
            winsound.Beep(freq, duration)
        except Exception as e:
            print("خطأ في الصوت:", e)
            
    threading.Thread(target=_beep, daemon=True).start()

# -----------------------------
# 1.Morse arabic translator 
# -----------------------------
MORSE_CODE_DICT = {
    '.-': 'أ', '-...': 'ب', '-': 'ت', '-.-.': 'ث', '.---': 'ج',
    '....': 'ح', '---': 'خ', '-..': 'د', '--..': 'ذ', '.-.': 'ر',
    '---.': 'ز', '...': 'س', '----': 'ش', '---.': 'ص', '...-': 'ض',
    '..-': 'ط', '-.--': 'ظ', '.-.-': 'ع', '--.': 'غ', '..-.': 'ف',
    '--.-': 'ق', '-.-': 'ك', '.-..': 'ل', '--': 'م', '-.': 'ن',
    '....': 'هـ', '---': 'و', '..': 'ي',
    
    '..': 'أحتاج ماء',
    '....': 'أشعر بألم',
    '---': 'مساعدة'
}

def decode_morse(morse_str):
    if not morse_str:
        return ""
    words = morse_str.strip().split(' / ')
    decoded_message = []
    for word in words:
        letters = word.split(' ')
        decoded_word = "".join([MORSE_CODE_DICT.get(letter, '') for letter in letters])
        decoded_message.append(decoded_word)
    return " ".join(decoded_message)

# -----------------------------
# 2. MediaPipe set up
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO
)
landmarker = FaceLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)
frame_timestamp_ms = 0
EYE_THRESHOLD = 0.18

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def calculate_eye_ratio(face_landmarks, outer, inner, upper, lower):
    horizontal = calculate_distance(face_landmarks[outer], face_landmarks[inner])
    vertical = calculate_distance(face_landmarks[upper], face_landmarks[lower])
    if horizontal == 0:
        return 0
    return vertical / horizontal

# -----------------------------
# 3. Time and Control
# -----------------------------
current_morse = ""
eye_closed_start = None
last_action_eye = None

print("=== تم تشغيل البرنامج بنجاح ===")
print("العين اليسرى: نقطة (.))")
print("العين اليمنى: شرطة (-)")
print("العينان معاً قصيرة: مسافة بين الأحرف")
print("العينان معاً ثانيتين : مسافة بين الكلمات /")
print("العينان معاً اكثر من 3 ثواني : مسح النص بالكامل")
print("زر (C) في لوحة المفاتيح: لمسح الشاشة يدوياً")
print("زر (Q) في لوحة المفاتيح: لإغلاق البرنامج")

while True:
    ret, frame = camera.read()
    if not ret:
        print("تعذر الاتصال بالكاميرا")
        break

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms += 33

    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    left_closed = False
    right_closed = False

    if result is not None and result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        left_ratio = calculate_eye_ratio(face_landmarks, 33, 133, 159, 145)
        right_ratio = calculate_eye_ratio(face_landmarks, 362, 263, 386, 374)

        left_closed = left_ratio < EYE_THRESHOLD
        right_closed = right_ratio < EYE_THRESHOLD

    # -----------------------------
    #Logic 
    # -----------------------------
    current_time = time.time()

    if left_closed or right_closed:
        if eye_closed_start is None:
            eye_closed_start = current_time

        if left_closed and right_closed:
            last_action_eye = 'both'
        elif left_closed and not right_closed:
            last_action_eye = 'left'
        elif right_closed and not left_closed:
            last_action_eye = 'right'

    else:
        if eye_closed_start is not None:
            duration = current_time - eye_closed_start

            if duration > 0.5:
                if last_action_eye == 'left':
                    current_morse += "."
                    play_beep(1200, 80) 
                elif last_action_eye == 'right':
                    current_morse += "-"
                    play_beep(600, 200) 
                elif last_action_eye == 'both':
                    if duration < 1.0:
                        current_morse += " "
                        play_beep(1500, 100)
                    elif 1.0 <= duration < 2.5:
                        current_morse += " / "
                        play_beep(1800, 200)
                    elif duration >= 3:
                        current_morse = ""
                        play_beep(400, 400) 
                        print("تم مسح النص")

            eye_closed_start = None
            last_action_eye = None

    translated_text = decode_morse(current_morse)

    # -----------------------------
    # 5. UI
    # -----------------------------
    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)

    cv2.putText(frame, f"Morse: {current_morse}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(frame, f"Text: {translated_text}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, "Left=- | Right=. | Both Short=Space | Both Med=Word Space / | Both Long=Clear", 
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.imshow("BlinkCode - Eye Morse Communicator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        current_morse = ""

camera.release()
cv2.destroyAllWindows()
landmarker.close()
