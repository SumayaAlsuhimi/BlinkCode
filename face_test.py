import time
import threading

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from eye_tracker import detect_eye_state
from morse_decoder import decode_morse
from text_to_speech import text_to_speech


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EyeMorse",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class EyeMorseProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.last_timestamp_ms = 0
        self.frame_counter = 0
        self.latest_landmarks = None

        self.face_detected = False

        self.left_closed = False
        self.right_closed = False

        self.left_ratio = 0.0
        self.right_ratio = 0.0

        self.eye_closed_start = None
        self.last_action_eye = None

        self.blink_duration_ms = 0

        self.current_morse = ""
        self.current_letter = ""

        self.message = ""

        self.last_action = "Waiting"

        def result_callback(
            result,
            output_image,
            timestamp_ms
        ):

            if result.face_landmarks:

                face_landmarks = result.face_landmarks[0]

                eye_state = detect_eye_state(
                    face_landmarks
                )

                left_closed = eye_state["left_closed"]
                right_closed = eye_state["right_closed"]

                with self.lock:

                    self.latest_landmarks = face_landmarks

                    self.face_detected = True

                    self.left_closed = left_closed
                    self.right_closed = right_closed

                    self.left_ratio = eye_state["left_ratio"]
                    self.right_ratio = eye_state["right_ratio"]

                    self.process_eye_action(
                        left_closed,
                        right_closed
                    )

            else:

                with self.lock:

                    self.latest_landmarks = None

                    self.face_detected = False

                    self.left_closed = False
                    self.right_closed = False

                    self.left_ratio = 0.0
                    self.right_ratio = 0.0

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="face_landmarker.task"
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=result_callback
        )

        self.landmarker = FaceLandmarker.create_from_options(
            options
        )


    # =====================================================
    # STATUS
    # =====================================================

    def get_status(self):

        with self.lock:

            return {
                "face_detected": self.face_detected,
                "left_closed": self.left_closed,
                "right_closed": self.right_closed,
                "left_ratio": self.left_ratio,
                "right_ratio": self.right_ratio,
                "blink_duration_ms": self.blink_duration_ms,
                "current_morse": self.current_morse,
                "current_letter": self.current_letter,
                "translated_text": self.message,
                "last_action": self.last_action
            }


    # =====================================================
    # CLEAR
    # =====================================================

    def clear_message(self):

        with self.lock:

            self.current_morse = ""
            self.current_letter = ""

            self.message = ""

            self.eye_closed_start = None
            self.last_action_eye = None

            self.blink_duration_ms = 0

            self.last_action = "Cleared"


    # =====================================================
    # CURRENT LETTER
    # =====================================================

    def update_current_letter(self):

        if not self.current_morse:

            self.current_letter = ""

            return

        self.current_letter = decode_morse(
            self.current_morse
        )


    # =====================================================
    # CONFIRM LETTER
    # =====================================================

    def confirm_letter(self):

        if not self.current_morse:

            self.last_action = "NO MORSE"

            return

        letter = decode_morse(
            self.current_morse
        )

        if letter:

            self.message += letter

            self.last_action = f"LETTER: {letter}"

        else:

            self.last_action = "INVALID MORSE"

        self.current_morse = ""
        self.current_letter = ""


    # =====================================================
    # WORD SPACE
    # =====================================================

    def add_word_space(self):

        if self.current_morse:

            self.confirm_letter()

        if (
            self.message
            and not self.message.endswith(" ")
        ):

            self.message += " "

        self.last_action = "WORD SPACE"


    # =====================================================
    # EYE → MORSE
    # =====================================================

    def process_eye_action(
        self,
        left_closed,
        right_closed
    ):

        current_time = time.monotonic()

        if left_closed or right_closed:

            if self.eye_closed_start is None:

                self.eye_closed_start = current_time

            duration = (
                current_time
                - self.eye_closed_start
            )

            self.blink_duration_ms = int(
                duration * 1000
            )

            if left_closed and right_closed:

                self.last_action_eye = "both"

            elif left_closed and not right_closed:

                self.last_action_eye = "left"

            elif right_closed and not left_closed:

                self.last_action_eye = "right"

        else:

            if self.eye_closed_start is None:

                self.blink_duration_ms = 0

                return

            duration = (
                current_time
                - self.eye_closed_start
            )

            self.blink_duration_ms = int(
                duration * 1000
            )

            if duration >= 0.42:

                # LEFT = DOT
                if self.last_action_eye == "left":

                    self.current_morse += "."

                    self.last_action = "DOT •"

                    self.update_current_letter()

                # RIGHT = DASH
                elif self.last_action_eye == "right":

                    self.current_morse += "-"

                    self.last_action = "DASH —"

                    self.update_current_letter()

                # BOTH
                elif self.last_action_eye == "both":

                    if duration < 1.0:

                        self.confirm_letter()

                    elif 1.0 <= duration < 2.5:

                        self.add_word_space()

                    elif duration >= 3.0:

                        self.current_morse = ""
                        self.current_letter = ""
                        self.message = ""

                        self.last_action = "CLEAR"

            self.eye_closed_start = None
            self.last_action_eye = None


    # =====================================================
    # VIDEO
    # =====================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        self.frame_counter += 1

        should_detect = (
            self.frame_counter % 2 == 0
        )

        if should_detect:

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            timestamp_ms = int(
                time.monotonic() * 1000
            )

            if timestamp_ms <= self.last_timestamp_ms:

                timestamp_ms = (
                    self.last_timestamp_ms + 1
                )

            self.last_timestamp_ms = timestamp_ms

            try:

                self.landmarker.detect_async(
                    mp_image,
                    timestamp_ms
                )

            except Exception:

                pass

        with self.lock:

            face_landmarks = self.latest_landmarks

        if face_landmarks:

            h, w, _ = image.shape

            left_eye_indices = [
                33,
                160,
                158,
                133,
                153,
                144
            ]

            right_eye_indices = [
                362,
                385,
                387,
                263,
                373,
                380
            ]

            left_points = []

            for index in left_eye_indices:

                landmark = face_landmarks[index]

                left_points.append(
                    [
                        int(landmark.x * w),
                        int(landmark.y * h)
                    ]
                )

            right_points = []

            for index in right_eye_indices:

                landmark = face_landmarks[index]

                right_points.append(
                    [
                        int(landmark.x * w),
                        int(landmark.y * h)
                    ]
                )

            left_points = np.array(
                left_points,
                dtype=np.int32
            )

            right_points = np.array(
                right_points,
                dtype=np.int32
            )

            cv2.polylines(
                image,
                [left_points],
                True,
                (0, 210, 255),
                2,
                cv2.LINE_AA
            )

            cv2.polylines(
                image,
                [right_points],
                True,
                (0, 210, 255),
                2,
                cv2.LINE_AA
            )

            for point in left_points:

                cv2.circle(
                    image,
                    tuple(point),
                    2,
                    (0, 210, 255),
                    -1
                )

            for point in right_points:

                cv2.circle(
                    image,
                    tuple(point),
                    2,
                    (0, 210, 255),
                    -1
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# =========================================================
# CSS
# =========================================================

st.html("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap'
);

.stApp {

    background:
        radial-gradient(
            circle at 5% 5%,
            rgba(255, 220, 80, 0.11),
            transparent 26%
        ),
        radial-gradient(
            circle at 95% 80%,
            rgba(255, 211, 70, 0.10),
            transparent 28%
        ),
        #FFFDF8;

    color: #24211A;

    font-family:
        'Manrope',
        sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {

    max-width: 1450px;

    padding-top: .8rem;

    padding-bottom: 2rem;
}


/* =========================================================
   NAVBAR
========================================================= */

.navbar {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 11px 18px;

    background:
        rgba(
            255,
            255,
            255,
            .92
        );

    border:
        1px solid #F2E6C4;

    border-radius: 22px;

    box-shadow:
        0 10px 32px
        rgba(91, 65, 10, .055);

    margin-bottom: 13px;
}


.brand {

    display: flex;

    align-items: center;

    gap: 10px;

    font-family:
        'Sora',
        sans-serif;

    font-size: 22px;

    font-weight: 750;

    color: #211F1A;
}


.brand-icon {

    width: 42px;

    height: 42px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 12px;

    background:
        linear-gradient(
            145deg,
            #FFE066,
            #FFC000
        );

    box-shadow:
        0 8px 20px
        rgba(218, 164, 0, .18);
}


.nav-links {

    display: flex;

    align-items: center;

    gap: 5px;

    padding: 4px;

    background: #FFFEFA;

    border:
        1px solid #F1E2B8;

    border-radius: 999px;
}


.nav-links span {

    padding:
        8px
        22px;

    color: #474238;

    border-radius: 999px;

    font-size: 13px;

    font-weight: 650;
}


.nav-links .active {

    background:
        linear-gradient(
            135deg,
            #FFF1A6,
            #FFE06D
        );

    color: #332A14;
}


.live {

    display: flex;

    align-items: center;

    gap: 8px;

    padding:
        8px
        14px;

    border:
        1px solid #EFDA83;

    border-radius: 999px;

    background: #FFFDF5;

    color: #BE8500;

    font-size: 11px;

    font-weight: 800;
}


.live-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #F4B700;

    box-shadow:
        0 0 0 5px
        rgba(244, 183, 0, .10);
}


/* =========================================================
   HERO
========================================================= */

.hero {

    text-align: center;

    margin-bottom: 13px;
}


.hero h1 {

    margin: 0;

    font-family:
        'Sora',
        sans-serif;

    font-size:
        clamp(
            40px,
            4.5vw,
            60px
        );

    font-weight: 800;

    letter-spacing: -3px;

    line-height: 1.02;

    color: #201E19;
}


.hero-yellow {

    color: #E7A700;
}


.hero p {

    margin:
        5px 0
        0;

    color: #8C857A;

    font-size: 13px;

    font-weight: 500;
}


/* =========================================================
   CAMERA HEADER
========================================================= */

.camera-head {

    display: flex;

    align-items: center;

    justify-content: space-between;

    margin-bottom: 8px;
}


.camera-title {

    display: flex;

    align-items: center;

    gap: 7px;

    color: #353127;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1.1px;
}


.camera-title-dot {

    width: 8px;

    height: 8px;

    background: #E3A800;

    border-radius: 50%;
}


.camera-badge {

    display: inline-flex;

    align-items: center;

    gap: 6px;

    padding:
        6px
        10px;

    color: #9F7200;

    background: #FFF4BF;

    border:
        1px solid #EED36A;

    border-radius: 999px;

    font-size: 9px;

    font-weight: 800;
}


/* =========================================================
   CAMERA
========================================================= */

[data-testid="stCustomComponentV1"] {

    width: 100% !important;

    margin: 0 !important;

    padding: 6px;

    background: #FFFFFF;

    border:
        1px solid #EDCB5C;

    border-radius: 22px;

    box-shadow:
        0 12px 34px
        rgba(93, 66, 7, .07);

    overflow: hidden;
}


[data-testid="stCustomComponentV1"] iframe {

    width: 100% !important;

    border-radius: 16px !important;
}


/* =========================================================
   STATUS CARDS
========================================================= */

.status-grid {

    height: 100%;

    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    grid-template-rows:
        repeat(2, minmax(0, 1fr));

    gap: 14px;
}


.status-card {

    position: relative;

    overflow: hidden;

    min-height: 145px;

    padding:
        20px
        19px;

    display: flex;

    align-items: center;

    gap: 15px;

    border:
        1px solid rgba(225, 169, 0, .30);

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            #FFF7C8 0%,
            #FFE883 52%,
            #FFD65B 100%
        );

    box-shadow:
        0 12px 28px
        rgba(202, 146, 0, .11);
}


.status-card::before {

    content: "";

    position: absolute;

    width: 140px;

    height: 140px;

    right: -50px;

    bottom: -65px;

    border-radius: 50%;

    background:
        rgba(
            255,
            255,
            255,
            .16
        );
}


.status-card::after {

    content: "✦";

    position: absolute;

    right: 18px;

    bottom: 10px;

    color:
        rgba(
            215,
            151,
            0,
            .23
        );

    font-size: 30px;
}


.status-card-icon {

    width: 56px;

    height: 56px;

    flex:
        0 0
        56px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 50%;

    color: #FFFFFF;

    background:
        linear-gradient(
            145deg,
            #FFD025,
            #F2A600
        );

    border:
        2px solid
        rgba(
            255,
            255,
            255,
            .90
        );

    box-shadow:
        0 8px 18px
        rgba(
            190,
            130,
            0,
            .20
        );

    font-size: 23px;
}


.status-card-content {

    position: relative;

    z-index: 2;

    min-width: 0;
}


.status-label {

    margin-bottom: 5px;

    color: #554A2D;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1.05px;
}


.status-value {

    color: #2E2412;

    font-family:
        'Sora',
        sans-serif;

    font-size: 24px;

    font-weight: 750;
}


.status-action {

    color: #775300;

    font-size: 19px;
}


/* =========================================================
   GUIDE
========================================================= */

.guide {

    display: flex;

    align-items: center;

    justify-content: center;

    flex-wrap: wrap;

    gap: 17px;

    margin:
        10px 0
        12px;

    padding:
        9px
        15px;

    border:
        1px solid #F0DEAA;

    border-radius: 999px;

    background:
        rgba(
            255,
            255,
            255,
            .65
        );

    color: #514B40;

    font-size: 11px;

    font-weight: 600;
}


.guide-item {

    display: flex;

    align-items: center;

    gap: 7px;
}


.guide-icon {

    color: #E4A300;

    font-size: 16px;

    font-weight: 900;
}


.guide-sep {

    width: 1px;

    height: 18px;

    background: #E4D2A0;
}


/* =========================================================
   MORSE + LETTER
========================================================= */

.morse-letter-row {

    display: grid;

    grid-template-columns:
        1fr
        1fr;

    gap: 14px;

    margin-bottom: 13px;
}


.result-card {

    position: relative;

    overflow: hidden;

    min-height: 105px;

    padding:
        17px
        24px;

    border:
        1px solid #EDCE6D;

    border-radius: 20px;

    background:
        linear-gradient(
            160deg,
            #FFFDF3 0%,
            #FFF3B5 55%,
            #FFE078 100%
        );

    box-shadow:
        0 9px 24px
        rgba(
            128,
            91,
            0,
            .065
        );

    text-align: center;
}


.result-card::after {

    content: "✦";

    position: absolute;

    right: 23px;

    top: 18px;

    color:
        rgba(
            222,
            158,
            0,
            .22
        );

    font-size: 34px;
}


.result-label {

    color: #4A4233;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1.3px;
}


.result-value {

    margin-top: 9px;

    color: #B97400;

    font-family:
        'Sora',
        sans-serif;

    font-size: 40px;

    font-weight: 800;

    line-height: 1;
}


.morse-value {

    color: #38280C;

    letter-spacing: 10px;
}


/* =========================================================
   MESSAGE
========================================================= */

.message-card {

    position: relative;

    overflow: hidden;

    min-height: 110px;

    padding:
        18px
        30px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    border:
        1px solid #EFCF67;

    border-radius: 21px;

    background:
        radial-gradient(
            circle at 50% 100%,
            rgba(255, 209, 49, .39),
            transparent 50%
        ),
        linear-gradient(
            135deg,
            #FFFDF4,
            #FFF0A9
        );

    box-shadow:
        0 9px 25px
        rgba(
            124,
            89,
            0,
            .06
        );

    margin-bottom: 13px;
}


.message-card::before {

    content: "";

    position: absolute;

    width: 180px;

    height: 70px;

    left: -35px;

    bottom: -25px;

    background-image:
        radial-gradient(
            #EBAF00 1.4px,
            transparent 1.4px
        );

    background-size:
        9px
        9px;

    opacity: .34;
}


.message-card::after {

    content: "✦";

    position: absolute;

    right: 35px;

    top: 25px;

    color:
        rgba(
            220,
            157,
            0,
            .31
        );

    font-size: 40px;
}


.message-label {

    position: relative;

    z-index: 2;

    color: #4E4637;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1.4px;
}


.message-value {

    position: relative;

    z-index: 2;

    margin-top: 4px;

    color: #BA7300;

    font-family:
        'Sora',
        sans-serif;

    font-size:
        clamp(
            35px,
            4vw,
            54px
        );

    font-weight: 800;

    line-height: 1.15;

    direction: rtl;
}


.message-empty {

    color: #9A907C;

    font-size: 18px;

    font-weight: 600;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {

    min-height: 50px;

    border:
        1px solid #F1BC15 !important;

    border-radius: 16px !important;

    background:
        linear-gradient(
            135deg,
            #FFD63C,
            #F5B600
        ) !important;

    color: #FFFFFF !important;

    font-family:
        'Manrope',
        sans-serif !important;

    font-size: 15px !important;

    font-weight: 800 !important;

    box-shadow:
        0 9px 22px
        rgba(
            202,
            140,
            0,
            .17
        );

    transition: .2s ease;
}


.stButton > button:hover {

    transform: translateY(-1px);

    border-color: #E4A600 !important;

    color: #FFFFFF !important;
}


/* =========================================================
   RESPONSIVE
========================================================= */

@media(max-width:950px) {

    .nav-links {
        display: none;
    }


    .status-grid {

        margin-top: 14px;

        grid-template-columns:
            repeat(
                2,
                1fr
            );
    }
}


@media(max-width:650px) {

    .hero h1 {
        font-size: 39px;
    }


    .live {
        display: none;
    }


    .status-card {

        min-height: 118px;
    }


    .status-card-icon {

        width: 42px;

        height: 42px;

        flex-basis: 42px;

        font-size: 18px;
    }


    .status-value {
        font-size: 17px;
    }


    .morse-letter-row {
        grid-template-columns: 1fr;
    }


    .guide-sep {
        display: none;
    }
}

</style>
""")


# =========================================================
# NAVBAR
# =========================================================

st.html("""
<nav class="navbar">

    <div class="brand">

        <div class="brand-icon">
            ◉
        </div>

        EyeMorse

    </div>


    <div class="nav-links">

        <span class="active">
            Home
        </span>

        <span>
            Camera
        </span>

        <span>
            Morse
        </span>

        <span>
            Message
        </span>

    </div>


    <div class="live">

        <div class="live-dot"></div>

        LIVE

    </div>

</nav>
""")


# =========================================================
# HERO
# =========================================================

st.html("""
<section class="hero">

    <h1>

        Your Eyes.

        <span class="hero-yellow">

            Your Voice.

        </span>

    </h1>


    <p>

        Blink. Decode. Communicate.

    </p>

</section>
""")


# =========================================================
# CAMERA + STATUS
# =========================================================

camera_col, status_col = st.columns(
    [1.1, 1],
    gap="medium"
)


# =========================================================
# CAMERA
# =========================================================

with camera_col:

    st.html("""
    <div class="camera-head">

        <div class="camera-title">

            <span class="camera-title-dot"></span>

            LIVE EYE TRACKING

        </div>


        <div class="camera-badge">

            <span class="live-dot"></span>

            TRACKING ACTIVE

        </div>

    </div>
    """)


    webrtc_ctx = webrtc_streamer(

        key="eyemorse-camera",

        video_processor_factory=EyeMorseProcessor,

        media_stream_constraints={

            "video": {

                "width": {
                    "ideal": 480
                },

                "height": {
                    "ideal": 360
                },

                "frameRate": {
                    "ideal": 24,
                    "max": 30
                },

                "facingMode": "user"
            },

            "audio": False
        },

        desired_playing_state=True,

        async_processing=True
    )


# =========================================================
# STATUS PLACEHOLDER
# =========================================================

with status_col:

    status_placeholder = st.empty()


# =========================================================
# GUIDE
# =========================================================

st.html("""
<div class="guide">

    <div class="guide-item">
        <span class="guide-icon">●</span>
        Left Eye • = Dot
    </div>

    <div class="guide-sep"></div>

    <div class="guide-item">
        <span class="guide-icon">▬</span>
        Right Eye — = Dash
    </div>

    <div class="guide-sep"></div>

    <div class="guide-item">
        <span class="guide-icon">✓</span>
        Both Short = Confirm
    </div>

    <div class="guide-sep"></div>

    <div class="guide-item">
        <span class="guide-icon">◫</span>
        Both Medium = Space
    </div>

    <div class="guide-sep"></div>

    <div class="guide-item">
        <span class="guide-icon">⌫</span>
        Both Long = Clear
    </div>

</div>
""")


# =========================================================
# RESULT PLACEHOLDER
# =========================================================

result_placeholder = st.empty()


# =========================================================
# LIVE INTERFACE
# =========================================================

@st.fragment(run_every=0.30)
def live_interface():

    processor = (
        webrtc_ctx.video_processor
        if webrtc_ctx
        else None
    )


    left_text = "WAITING"
    right_text = "WAITING"

    left_class = ""
    right_class = ""

    blink_duration = 0

    last_action = "Waiting"

    morse = ""
    current_letter = ""

    message = ""


    if processor is not None:

        status = processor.get_status()


        if status["face_detected"]:

            if status["left_closed"]:

                left_text = "CLOSED"

                left_class = "status-closed"

            else:

                left_text = "OPEN"

                left_class = "status-open"


            if status["right_closed"]:

                right_text = "CLOSED"

                right_class = "status-closed"

            else:

                right_text = "OPEN"

                right_class = "status-open"

        else:

            left_text = "NO FACE"
            right_text = "NO FACE"


        blink_duration = (
            status["blink_duration_ms"]
        )


        last_action = (
            status["last_action"]
            or "Waiting"
        )


        morse = (
            status["current_morse"]
        )


        current_letter = (
            status["current_letter"]
        )


        message = (
            status["translated_text"]
        )


    # =====================================================
    # STATUS CARDS
    # =====================================================

    status_placeholder.html(
        f"""
        <div class="status-grid">

            <div class="status-card">

                <div class="status-card-icon">
                    ◉
                </div>

                <div class="status-card-content">

                    <div class="status-label">
                        LEFT EYE
                    </div>

                    <div class="status-value {left_class}">
                        {left_text}
                    </div>

                </div>

            </div>


            <div class="status-card">

                <div class="status-card-icon">
                    ◉
                </div>

                <div class="status-card-content">

                    <div class="status-label">
                        RIGHT EYE
                    </div>

                    <div class="status-value {right_class}">
                        {right_text}
                    </div>

                </div>

            </div>


            <div class="status-card">

                <div class="status-card-icon">
                    ◷
                </div>

                <div class="status-card-content">

                    <div class="status-label">
                        BLINK DURATION
                    </div>

                    <div class="status-value">
                        {blink_duration} ms
                    </div>

                </div>

            </div>


            <div class="status-card">

                <div class="status-card-icon">
                    ⚡
                </div>

                <div class="status-card-content">

                    <div class="status-label">
                        LAST ACTION
                    </div>

                    <div class="status-value status-action">
                        {last_action}
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # =====================================================
    # MORSE
    # =====================================================

    if morse:

        display_morse = (
            morse
            .replace(".", "•")
            .replace("-", "—")
        )

    else:

        display_morse = "—"


    display_letter = (
        current_letter
        if current_letter
        else "—"
    )


    if message:

        display_message = message

        message_class = "message-value"

    else:

        display_message = (
            "Your message will appear here"
        )

        message_class = (
            "message-value message-empty"
        )


    # =====================================================
    # RESULTS
    # =====================================================

    result_placeholder.html(
        f"""
        <div class="morse-letter-row">

            <div class="result-card">

                <div class="result-label">
                    CURRENT MORSE
                </div>

                <div class="result-value morse-value">
                    {display_morse}
                </div>

            </div>


            <div class="result-card">

                <div class="result-label">
                    CURRENT LETTER
                </div>

                <div class="result-value">
                    {display_letter}
                </div>

            </div>

        </div>


        <div class="message-card">

            <div class="message-label">
                YOUR MESSAGE
            </div>

            <div class="{message_class}">
                {display_message}
            </div>

        </div>
        """
    )


live_interface()


# =========================================================
# CENTER BUTTONS
# =========================================================

space_left, clear_col, speak_col, space_right = st.columns(
    [1.35, 1.55, 1.55, 1.35],
    gap="small"
)


with clear_col:

    clear_clicked = st.button(
        "🗑  Clear Message",
        use_container_width=True
    )


with speak_col:

    speak_clicked = st.button(
        "🔊  Speak",
        use_container_width=True
    )


# =========================================================
# CLEAR ACTION
# =========================================================

if clear_clicked:

    processor = (
        webrtc_ctx.video_processor
        if webrtc_ctx
        else None
    )

    if processor:

        processor.clear_message()

        st.rerun()


# =========================================================
# SPEAK ACTION
# =========================================================

if speak_clicked:

    processor = (
        webrtc_ctx.video_processor
        if webrtc_ctx
        else None
    )

    if processor is None:

        st.warning(
            "Camera is not ready yet."
        )

    else:

        status = processor.get_status()

        message = status[
            "translated_text"
        ]

        if not message:

            st.warning(
                "No decoded message yet."
            )

        else:

            try:

                audio = text_to_speech(
                    message,
                    lang="ar"
                )

                if audio is not None:

                    st.audio(
                        audio,
                        format="audio/mp3",
                        autoplay=True
                    )

            except Exception as error:

                st.error(
                    f"Voice error: {error}"
                )