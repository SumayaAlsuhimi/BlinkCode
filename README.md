# EyeMorse 👁️

### Your Eyes. Your Voice.

EyeMorse is an assistive communication system that transforms intentional eye movements into Morse code, Arabic text, and voice.

The project enables users to communicate through eye gestures using real-time computer vision and facial landmark tracking.

---

## Overview

EyeMorse detects eye movements through a live camera feed and converts them into Morse code symbols.

Each eye action represents a specific command:

| Eye Action | Command |
|---|---|
| Left Eye | Dot `.` |
| Right Eye | Dash `-` |
| Both Eyes - Short Blink | Confirm Letter |
| Both Eyes - Medium Blink | Add Space |
| Both Eyes - Long Blink | Clear Message |

The generated Morse sequence is decoded into Arabic letters, allowing users to construct words and messages using only eye movements.

The final message can also be converted into speech.

---

## How It Works

```text
Eye Movement
      ↓
Morse Code
      ↓
Arabic Letter
      ↓
Arabic Message
      ↓
Voice Output
```

EyeMorse continuously tracks facial landmarks and detects the state of each eye.

Based on the detected eye action, the system builds a Morse sequence and converts it into Arabic text.

---

## Features

### Real-Time Eye Tracking
Tracks the user's eyes through the webcam using facial landmark detection.

### Eye-Controlled Morse Code
Users generate dots and dashes using left and right eye movements.

### Arabic Morse Decoder
Morse sequences are converted directly into Arabic letters.

### Message Construction
Confirmed letters are automatically added to the current message.

### Text-to-Speech
The completed Arabic message can be converted into voice.

### Morse Explorer
Users can enter an Arabic letter and view:

- Its Morse code
- The required eye movements
- The confirmation action

### Impact
Explains the complete EyeMorse communication pipeline and the potential impact of hands-free communication.

---

## Application Pages

### Home

The main real-time communication interface.

It includes:

- Live camera
- Eye tracking status
- Left and right eye status
- Blink duration
- Current action
- Current Morse sequence
- Current Arabic letter
- Generated message
- Clear Message
- Speak

---

### Morse Explorer

An interactive page for exploring the eye movements required to generate Arabic letters.

Users enter an Arabic letter and EyeMorse displays:

```text
Letter
Morse Code
Eye Movement Sequence
```

Example:

```text
Letter: ن

Morse:
- .

Eye Movement:
Right Eye → Left Eye → Both Eyes → Confirm
```

---

### Impact

Explains the EyeMorse communication workflow:

```text
Eye Movement
      ↓
Morse Code
      ↓
Arabic Text
      ↓
Voice
```

The page highlights the project's focus on:

- Hands-Free Communication
- Arabic Support
- Real-Time Interaction

---

## Technologies

EyeMorse was developed using:

- Python
- Streamlit
- Streamlit WebRTC
- MediaPipe
- OpenCV
- NumPy
- gTTS
- HTML / CSS

---

## Project Structure

```text
EyeMorse/
│
├── app.py
├── eye_tracker.py
├── morse_decoder.py
├── text_to_speech.py
├── face_landmarker.task
├── style.css
├── requirements.txt
├── packages.txt
│
├── .streamlit/
│   └── config.toml
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SumayaAlsuhimi/BlinkCode.git
```

Move into the project directory:

```bash
cd BlinkCode
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Allow camera access when prompted.

---

## Eye Interaction Logic

EyeMorse uses intentional eye movements to generate Morse code.

```text
Left Eye
   ↓
Dot .

Right Eye
   ↓
Dash -

Both Eyes
   ↓
Confirm / Space / Clear
```

Different both-eye blink durations are used to perform different actions.

---

## Example

To generate the Arabic letter:

```text
ن
```

The Morse sequence is:

```text
-.
```

The user performs:

```text
Right Eye
    ↓
Left Eye
    ↓
Both Eyes - Short Blink
    ↓
Confirm
```

The system confirms:

```text
ن
```

The process continues until a complete word or message is created.

---

## Goal

EyeMorse explores how computer vision can provide an alternative communication method using only eye movements.

The project focuses on making communication:

- Hands-free
- Accessible
- Real-time
- Arabic-friendly

---

## Future Improvements

Possible future developments include:

- Personalized eye calibration
- Improved blink sensitivity
- Word prediction
- Smart phrase suggestions
- Faster communication shortcuts
- Support for additional languages
- Improved cloud WebRTC connectivity

---

## Live Demo

[https://eyemorse.streamlit.app](https://eyemorse.streamlit.app/)

For the best real-time camera performance, the application can also be run locally.

---

## EyeMorse

**Your Eyes. Your Voice.**

Blink. Decode. Communicate.
