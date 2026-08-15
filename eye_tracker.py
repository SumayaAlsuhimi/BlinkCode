import math


EYE_THRESHOLD = 0.18


def calculate_distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_eye_ratio(
    face_landmarks,
    outer,
    inner,
    upper,
    lower
):
    horizontal = calculate_distance(
        face_landmarks[outer],
        face_landmarks[inner]
    )

    vertical = calculate_distance(
        face_landmarks[upper],
        face_landmarks[lower]
    )

    if horizontal == 0:
        return 0

    return vertical / horizontal


def detect_eye_state(face_landmarks):
    left_ratio = calculate_eye_ratio(
        face_landmarks,
        33,
        133,
        159,
        145
    )

    right_ratio = calculate_eye_ratio(
        face_landmarks,
        362,
        263,
        386,
        374
    )

    left_closed = left_ratio < EYE_THRESHOLD
    right_closed = right_ratio < EYE_THRESHOLD

    return {
        "left_closed": left_closed,
        "right_closed": right_closed,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio
    }