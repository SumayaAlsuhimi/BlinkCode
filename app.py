import time
import threading
from pathlib import Path

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
# PAGE CONFIG + CSS
# =========================================================

st.set_page_config(
    page_title="EyeMorse",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STYLE_FILE = Path(__file__).with_name("style.css")
if STYLE_FILE.exists():
    st.html(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>")
else:
    st.error("style.css was not found.")


# =========================================================
# SESSION STATE
# =========================================================

if "current_message" not in st.session_state:
    st.session_state.current_message = ""


if "morse_inputs" not in st.session_state:
    st.session_state.morse_inputs = 0


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
        self.total_morse_inputs = 0

        def result_callback(result, output_image, timestamp_ms):
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                eye_state = detect_eye_state(face_landmarks)

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
            result_callback=result_callback,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

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
                "last_action": self.last_action,
                "total_morse_inputs": self.total_morse_inputs,
            }

    def clear_message(self):
        with self.lock:
            self.current_morse = ""
            self.current_letter = ""
            self.message = ""

            self.eye_closed_start = None
            self.last_action_eye = None
            self.blink_duration_ms = 0

            self.total_morse_inputs = 0
            self.last_action = "Cleared"

    def update_current_letter(self):
        if not self.current_morse:
            self.current_letter = ""
            return

        self.current_letter = decode_morse(
            self.current_morse
        )

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

    def add_word_space(self):
        if self.current_morse:
            self.confirm_letter()

        if self.message and not self.message.endswith(" "):
            self.message += " "

        self.last_action = "WORD SPACE"

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

            # Ignore natural fast blinks
            if duration >= 0.42:

                # LEFT = DOT
                if self.last_action_eye == "left":
                    self.current_morse += "."
                    self.total_morse_inputs += 1

                    self.last_action = "DOT •"
                    self.update_current_letter()

                # RIGHT = DASH
                elif self.last_action_eye == "right":
                    self.current_morse += "-"
                    self.total_morse_inputs += 1

                    self.last_action = "DASH —"
                    self.update_current_letter()

                # BOTH
                elif self.last_action_eye == "both":

                    # Short = confirm current letter
                    if duration < 1.0:
                        self.confirm_letter()

                    # Medium = word space
                    elif 1.0 <= duration < 2.5:
                        self.add_word_space()

                    # Long = clear
                    elif duration >= 3.0:
                        self.current_morse = ""
                        self.current_letter = ""
                        self.message = ""
                        self.total_morse_inputs = 0
                        self.last_action = "CLEAR"

            self.eye_closed_start = None
            self.last_action_eye = None

    def recv(self, frame):
        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(image, 1)

        # Detect every second frame for lower latency
        self.frame_counter += 1

        if self.frame_counter % 2 == 0:
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
                timestamp_ms = self.last_timestamp_ms + 1

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
                33, 160, 158,
                133, 153, 144
            ]

            right_eye_indices = [
                362, 385, 387,
                263, 373, 380
            ]

            left_points = np.array(
                [
                    [
                        int(face_landmarks[i].x * w),
                        int(face_landmarks[i].y * h),
                    ]
                    for i in left_eye_indices
                ],
                dtype=np.int32
            )

            right_points = np.array(
                [
                    [
                        int(face_landmarks[i].x * w),
                        int(face_landmarks[i].y * h),
                    ]
                    for i in right_eye_indices
                ],
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
# HELPERS
# =========================================================


def speak_message(message):
    message = (message or "").strip()

    if not message:
        st.warning("No decoded message yet.")
        return

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


# =========================================================
# NAVIGATION
# =========================================================

allowed_pages = [
    "Home",
    "Morse",
    "Impact"
]

page = st.query_params.get(
    "page",
    "Home"
)

if page not in allowed_pages:
    page = "Home"


def nav_link(label):
    active = page == label

    background = (
        "linear-gradient(135deg,#fff1b1,#ffdc68)"
        if active
        else "transparent"
    )

    color = (
        "#3c3017"
        if active
        else "#5b5448"
    )

    return f"""
    <a
        href="?page={label}"
        class="nav-page-link"
        style="
            background:{background};
            color:{color};
        "
    >
        {label}
    </a>
    """


st.html(
    f"""
    <nav class="navbar">

        <div class="brand">

            <div class="brand-icon">
                ◉
            </div>

            EyeMorse

        </div>

        <div class="nav-links">

            {nav_link("Home")}
            {nav_link("Morse")}
            {nav_link("Impact")}

        </div>

        <div class="live">

            <div class="live-dot"></div>

            LIVE

        </div>

    </nav>
    """
)


# =========================================================
# HOME PAGE
# =========================================================

def render_home():

    st.html("""
    <section class="hero-compact">

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

    camera_col, status_col = st.columns(
        [1.35, 1],
        gap="medium"
    )

    with camera_col:

        st.html("""
        <div class="camera-header">

            <div class="camera-title">
                ◉ LIVE EYE TRACKING
            </div>

            <div class="camera-status">

                <div class="live-dot"></div>

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

    with status_col:
        st.html(
            '<div class="status-top-spacer"></div>'
        )

        status_placeholder = st.empty()

    st.html("""
    <div class="guide-strip">

        <strong>
            Left Eye • = Dot
        </strong>

        <span class="guide-separator">
            |
        </span>

        <strong>
            Right Eye — = Dash
        </strong>

        <span class="guide-separator">
            |
        </span>

        Both Short = Confirm

        <span class="guide-separator">
            |
        </span>

        Both Medium = Space

        <span class="guide-separator">
            |
        </span>

        Both Long = Clear

    </div>
    """)

    result_placeholder = st.empty()

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

        message = st.session_state.current_message
        total_morse_inputs = st.session_state.morse_inputs

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

            morse = status[
                "current_morse"
            ]

            current_letter = status[
                "current_letter"
            ]

            message = status[
                "translated_text"
            ]

            total_morse_inputs = status.get(
                "total_morse_inputs",
                0
            )

            st.session_state.current_message = message
            st.session_state.morse_inputs = total_morse_inputs

        status_placeholder.html(
            f"""
            <div class="status-side">

                <div class="status-card">

                    <div class="status-icon">
                        ◉
                    </div>

                    <div class="status-content">

                        <div class="status-label">
                            LEFT EYE
                        </div>

                        <div class="status-value {left_class}">
                            {left_text}
                        </div>

                    </div>

                </div>

                <div class="status-card">

                    <div class="status-icon">
                        ◉
                    </div>

                    <div class="status-content">

                        <div class="status-label">
                            RIGHT EYE
                        </div>

                        <div class="status-value {right_class}">
                            {right_text}
                        </div>

                    </div>

                </div>

                <div class="status-card">

                    <div class="status-icon">
                        ◷
                    </div>

                    <div class="status-content">

                        <div class="status-label">
                            BLINK
                        </div>

                        <div class="status-value">
                            {blink_duration} ms
                        </div>

                    </div>

                </div>

                <div class="status-card">

                    <div class="status-icon">
                        ✦
                    </div>

                    <div class="status-content">

                        <div class="status-label">
                            ACTION
                        </div>

                        <div class="status-value action-yellow">
                            {last_action}
                        </div>

                    </div>

                </div>

            </div>
            """
        )

        display_morse = (
            morse
            .replace(".", "•")
            .replace("-", "—")
            if morse
            else "—"
        )

        display_letter = (
            current_letter
            if current_letter
            else "—"
        )

        if message:
            display_message = message
            message_class = "message"
        else:
            display_message = (
                "Your message will appear here"
            )
            message_class = (
                "message message-empty"
            )

        result_placeholder.html(
            f"""
            <section class="morse-panel">

                <div class="morse-main">

                    <div class="section-title">
                        CURRENT MORSE
                    </div>

                    <div class="morse-code">
                        {display_morse}
                    </div>

                </div>

                <div class="current-letter">

                    <span>
                        CURRENT LETTER
                    </span>

                    <strong>
                        {display_letter}
                    </strong>

                </div>

            </section>

            <section class="message-card">

                <div class="section-title">
                    YOUR MESSAGE
                </div>

                <div class="{message_class}">
                    {display_message}
                </div>

            </section>
            """
        )

    live_interface()

    st.write("")

    left_space, clear_col, speak_col, right_space = st.columns(
        [1.4, 1.25, 1.25, 1.4],
        gap="small"
    )

    with clear_col:
        clear_clicked = st.button(
            "🗑 Clear Message",
            use_container_width=True,
            key="home_clear"
        )

    with speak_col:
        speak_clicked = st.button(
            "🔊 Speak",
            use_container_width=True,
            key="home_speak"
        )

    if clear_clicked:

        processor = (
            webrtc_ctx.video_processor
            if webrtc_ctx
            else None
        )

        if processor:
            processor.clear_message()

        st.session_state.current_message = ""
        st.session_state.morse_inputs = 0

        st.rerun()

    if speak_clicked:

        processor = (
            webrtc_ctx.video_processor
            if webrtc_ctx
            else None
        )

        if processor:
            status = processor.get_status()
            message = status["translated_text"]
            st.session_state.current_message = message
        else:
            message = st.session_state.current_message

        speak_message(message)


# =========================================================
# MORSE PAGE
# =========================================================

def render_morse():

    st.html("""
    <section class="hero-compact">

        <h1>
            Learn
            <span class="hero-yellow">
                Eye Morse.
            </span>
        </h1>

        <p>
            Type an Arabic letter and learn how to create it using your eyes.
        </p>

    </section>
    """)

    # Eye controls
    st.html("""
    <div class="morse-controls-grid">

        <div class="status-card morse-control-card">

            <div class="status-icon morse-control-icon">
                •
            </div>

            <div class="status-label">
                LEFT EYE
            </div>

            <div class="status-value morse-control-value">
                DOT
            </div>

        </div>

        <div class="status-card morse-control-card">

            <div class="status-icon morse-control-icon">
                —
            </div>

            <div class="status-label">
                RIGHT EYE
            </div>

            <div class="status-value morse-control-value">
                DASH
            </div>

        </div>

        <div class="status-card morse-control-card">

            <div class="status-icon morse-control-icon">
                ✓
            </div>

            <div class="status-label">
                BOTH SHORT
            </div>

            <div class="status-value morse-control-value small">
                CONFIRM
            </div>

        </div>

        <div class="status-card morse-control-card">

            <div class="status-icon morse-control-icon">
                ↔
            </div>

            <div class="status-label">
                BOTH MEDIUM
            </div>

            <div class="status-value morse-control-value small">
                SPACE
            </div>

        </div>

        <div class="status-card morse-control-card">

            <div class="status-icon morse-control-icon">
                ⌫
            </div>

            <div class="status-label">
                BOTH LONG
            </div>

            <div class="status-value morse-control-value small">
                CLEAR
            </div>

        </div>

    </div>
    """)

    # Explanation
    st.html("""
    <section class="message-card morse-info-card">

        <div class="section-title">
            HOW IT WORKS
        </div>

        <div class="morse-info-text">

            Enter an Arabic letter below.
            EyeMorse will show its Morse code
            and the exact eye movements needed to create it.

        </div>

    </section>
    """)

    # Arabic Morse dictionary
    arabic_morse = {
        "أ": ".-",
        "ا": ".-",
        "ب": "-...",
        "ت": "-",
        "ث": "-.-.",
        "ج": ".---",
        "ح": "....",
        "خ": "---",
        "د": "-..",
        "ذ": "--..",
        "ر": ".-.",
        "ز": "---.",
        "س": "...",
        "ش": "----",
        "ص": "-..-",
        "ض": "...-",
        "ط": "..-",
        "ظ": "-.--",
        "ع": ".-.-",
        "غ": "--.",
        "ف": "..-.",
        "ق": "--.-",
        "ك": "-.-",
        "ل": ".-..",
        "م": "--",
        "ن": "-.",
        "ه": "..-..",
        "هـ": "..-..",
        "و": ".--",
        "ي": "..",
    }

    st.html("""
    <div class="explorer-heading">

        <div class="section-title">
            LETTER EXPLORER
        </div>

        <div class="explorer-subtitle">
            Enter an Arabic letter to see its Morse code.
        </div>

    </div>
    """)

    with st.form(
        "letter_explorer_form",
        clear_on_submit=False
    ):

        letter = st.text_input(
            "Arabic Letter",
            placeholder="مثال: ن",
            max_chars=1,
            key="letter_explorer_input"
        )

        submitted = st.form_submit_button(
            "Show Morse Code",
            use_container_width=True
        )

    if submitted:

        letter = letter.strip()

        normalize_letters = {
            "إ": "ا",
            "آ": "ا",
            "ة": "ه",
            "ى": "ي",
        }

        letter = normalize_letters.get(
            letter,
            letter
        )

        if not letter:

            st.warning(
                "اكتبي حرف عربي أولاً."
            )

        elif letter in arabic_morse:

            code = arabic_morse[letter]

            display_code = (
                code
                .replace(".", "•")
                .replace("-", "—")
            )

            movements = []

            for symbol in code:

                if symbol == ".":
                    movements.append(
                        "Left Eye"
                    )

                elif symbol == "-":
                    movements.append(
                        "Right Eye"
                    )

            movement_text = (
                " → ".join(movements)
            )

            st.html(
                f"""
                <section class="letter-result-card">

                    <div class="letter-result-left">

                        <div class="result-small-label">
                            LETTER
                        </div>

                        <div class="result-letter">
                            {letter}
                        </div>

                    </div>

                    <div class="letter-result-center">

                        <div class="result-small-label">
                            MORSE CODE
                        </div>

                        <div class="result-morse">
                            {display_code}
                        </div>

                    </div>

                    <div class="letter-result-right">

                        <div class="result-small-label">
                            EYE MOVEMENT
                        </div>

                        <div class="result-movement">
                            {movement_text}
                        </div>

                        <div class="result-confirm">
                            Both Eyes → Confirm
                        </div>

                    </div>

                </section>
                """
            )

        else:

            st.warning(
                "هذا الحرف غير موجود في قاموس EyeMorse الحالي."
            )




# =========================================================
# IMPACT PAGE
# =========================================================

def render_impact():

# =====================================================
    # HERO
    # =====================================================

    st.html("""
    <section class="hero-compact impact-hero">

        <h1>
            Communication
            <span class="hero-yellow">
                Beyond Hands.
            </span>
        </h1>

        <p>
            EyeMorse transforms intentional eye movements
            into Morse code, Arabic text, and voice.
        </p>

    </section>
    """)


    # =====================================================
    # INTRO
    # =====================================================

    st.html("""
    <section class="message-card impact-intro-card">

        <div class="section-title">
            WHY EYEMORSE?
        </div>

        <div class="impact-intro-text">
            Communication should not depend on hands or keyboards.
            EyeMorse offers a simple way to communicate
            using intentional eye movements only.
        </div>

    </section>
    """)


    # =====================================================
    # HOW EYEMORSE WORKS
    # =====================================================

    st.html("""
    <div class="impact-section-heading">

        <div class="section-title">
            HOW EYEMORSE WORKS
        </div>

        <div class="impact-section-subtitle">
            From eye movement to spoken communication.
        </div>

    </div>
    """)


    st.html("""
    <div class="impact-flow">

        <div class="impact-flow-card">

            <div class="impact-flow-icon">
                ◉
            </div>

            <div class="impact-flow-number">
                01
            </div>

            <div class="impact-flow-title">
                EYE MOVEMENT
            </div>

            <div class="impact-flow-text">
                Left, right, and both-eye actions
                are detected in real time.
            </div>

        </div>


        <div class="impact-arrow">
            →
        </div>


        <div class="impact-flow-card">

            <div class="impact-flow-icon">
                • —
            </div>

            <div class="impact-flow-number">
                02
            </div>

            <div class="impact-flow-title">
                MORSE CODE
            </div>

            <div class="impact-flow-text">
                Eye actions become dots,
                dashes, and control commands.
            </div>

        </div>


        <div class="impact-arrow">
            →
        </div>


        <div class="impact-flow-card">

            <div class="impact-flow-icon">
                أ
            </div>

            <div class="impact-flow-number">
                03
            </div>

            <div class="impact-flow-title">
                ARABIC TEXT
            </div>

            <div class="impact-flow-text">
                Morse sequences are decoded
                into Arabic letters and words.
            </div>

        </div>


        <div class="impact-arrow">
            →
        </div>


        <div class="impact-flow-card">

            <div class="impact-flow-icon">
                🔊
            </div>

            <div class="impact-flow-number">
                04
            </div>

            <div class="impact-flow-title">
                VOICE
            </div>

            <div class="impact-flow-text">
                The final message can be
                spoken aloud instantly.
            </div>

        </div>

    </div>
    """)


    # =====================================================
    # VALUE
    # =====================================================

    st.html("""
    <div class="impact-section-heading impact-value-heading">

        <div class="section-title">
            WHAT MAKES EYEMORSE USEFUL?
        </div>

    </div>
    """)


    st.html("""
    <div class="impact-value-grid">

        <div class="impact-value-card">

            <div class="impact-value-icon">
                ◉
            </div>

            <div class="impact-value-title">
                HANDS-FREE
            </div>

            <div class="impact-value-text">
                Communication can be controlled
                entirely through intentional eye movements.
            </div>

        </div>


        <div class="impact-value-card">

            <div class="impact-value-icon">
                ع
            </div>

            <div class="impact-value-title">
                ARABIC SUPPORT
            </div>

            <div class="impact-value-text">
                Morse sequences are decoded directly
                into Arabic letters and words.
            </div>

        </div>


        <div class="impact-value-card">

            <div class="impact-value-icon">
                ⚡
            </div>

            <div class="impact-value-title">
                REAL-TIME
            </div>

            <div class="impact-value-text">
                Eye movements, Morse code,
                text, and actions update live.
            </div>

        </div>

    </div>
    """)


    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    st.html("""
    <section class="impact-final">

        <div class="impact-final-small">
            EYEMORSE
        </div>

        <div class="impact-final-title">
            Your Eyes Are Enough
            <span>
                to Be Heard.
            </span>
        </div>

        <div class="impact-final-text">
            A simple interaction can become a meaningful voice.
        </div>

    </section>
    """)


# =========================================================
# RENDER
# =========================================================

if page == "Home":
    render_home()

elif page == "Morse":
    render_morse()

elif page == "Impact":
    render_impact()