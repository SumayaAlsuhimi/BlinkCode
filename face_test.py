import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe setup
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = FaceLandmarker.create_from_options(options)

# -----------------------------
# Open webcam
# -----------------------------
camera = cv2.VideoCapture(0)

frame_timestamp_ms = 0

# Threshold to decide OPEN / CLOSED
EYE_THRESHOLD = 0.18


# -----------------------------
# Helper functions
# -----------------------------
def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def eye_ratio(face_landmarks, outer, inner, upper, lower):
    horizontal = distance(
        face_landmarks[outer],
        face_landmarks[inner]
    )

    vertical = distance(
        face_landmarks[upper],
        face_landmarks[lower]
    )

    if horizontal == 0:
        return 0

    return vertical / horizontal


# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read from camera")
        break

    # Convert OpenCV BGR image to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp_ms += 33

    # Detect face landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    left_status = "NO FACE"
    right_status = "NO FACE"

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]

        h, w, _ = frame.shape

        # -----------------------------
        # Calculate left eye ratio
        # -----------------------------
        left_ratio = eye_ratio(
            face_landmarks,
            33,
            133,
            159,
            145
        )

        # -----------------------------
        # Calculate right eye ratio
        # -----------------------------
        right_ratio = eye_ratio(
            face_landmarks,
            362,
            263,
            386,
            374
        )

        # Decide each eye status separately
        if left_ratio < EYE_THRESHOLD:
            left_status = "CLOSED"
        else:
            left_status = "OPEN"

        if right_ratio < EYE_THRESHOLD:
            right_status = "CLOSED"
        else:
            right_status = "OPEN"

        # -----------------------------
        # Draw eye landmarks
        # -----------------------------
        eye_points = [
            33, 133, 159, 145,
            362, 263, 386, 374
        ]

        for idx in eye_points:
            landmark = face_landmarks[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

        # -----------------------------
        # Show eye ratios
        # -----------------------------
        cv2.putText(
            frame,
            f"Left Ratio: {left_ratio:.3f}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Right Ratio: {right_ratio:.3f}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # Show eye statuses
    # -----------------------------
    cv2.putText(
        frame,
        f"Left Eye: {left_status}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Right Eye: {right_status}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show camera
    cv2.imshow(
        "BlinkCode - Eye Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
camera.release()
cv2.destroyAllWindows()
landmarker.close()