# Empathetic Interviewer: Adaptive Robotic Interaction

A real-time, multimodal robotic interviewing system that adapts its behavior based on the user's emotional state. Built for the **Furhat Robot**, this system combines foundational computer vision models (DINOv2) with Large Language Models to create an interviewer that doesn't just ask questions—it *understands* how you feel.

## 🚀 Key Features

*   **Multimodal Perception**: Uses a custom-trained **DINOv2** vision model with a Spatial Attention Head to detect 7 micro-expressions (Happiness, Sadness, Fear, Anger, Disgust, Surprise, Neutral) in real-time.
*   **Empathetic Dialogue**: The LLM "Interviewer" dynamically adjusts its questioning strategy and tone.
    *   *User is Anxious?* → The robot becomes supportive, slows down, and validates feelings.
    *   *User is Confident?* → The robot challenges the candidate with deeper questions.
*   **Live Backchanneling**: The robot performs non-verbal gestures (nodding, smiling, frowning) *while listening* to the user, creating a natural, active-listening experience.
*   **Latency Masking**: Performs "thinking" gestures during LLM processing to maintain engagement.
*   **Structured Interview Flow**: Automatically manages interview phases (Introduction → Behavioral Questions → Closing).

## 📂 Project Structure

```
Empathetic-Interviewer/
├── emotion.py                   # VISION MODULE: DINOv2 model definition & real-time inference logic
├── furhat_conversation.py       # MAIN CONTROLLER: Manages Dialogue, Robot Control, and Threading
├── empathetic_interviewer_perception.pth  # Trained model weights for emotion detection
├── requirements.txt             # Python dependencies
├── test_camera.py               # Utility to verify webcam access
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Furhat SDK (running locally or on a physical robot)
*   A working Webcam
*   An API Key for OpenRouter (or OpenAI/Anthropic directly)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Empathetic-Interviewer.git
    cd Empathetic-Interviewer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory:
    ```bash
    OPENROUTER_API_KEY=your_api_key_here
    ```

## 🏃‍♂️ Usage

1.  **Start Furhat**: Ensure your Furhat robot (or the Furhat Virtual Skill) is running and accessible at `127.0.0.1` (localhost).

2.  **Run the Interviewer:**
    ```bash
    python furhat_conversation.py
    ```

3.  **Interaction Flow:**
    *   The robot will greet you warmly and introduce the session.
    *   It will ask you to introduce yourself.
    *   It will proceed through a behavioral interview, reacting to your facial expressions in real-time.
    *   Press `Ctrl+C` in the terminal to stop the session.

## 🧠 How It Works

### 1. Vision Pipeline (`emotion.py`)
*   Captures frames from the webcam.
*   Feeds them into a **Frozen DINOv2 Backbone**.
*   Passes features through a **Trainable Spatial Attention Head** to focus on facial landmarks.
*   Outputs an emotion class and confidence score.

### 2. The Conversation Loop (`furhat_conversation.py`)
*   **Thread 1 (Perception):** Runs the vision model in the background. It updates a shared `latest_emotion` variable and sends immediate "listening gestures" to the robot.
*   **Thread 2 (Dialogue):**
    *   Listens to user speech.
    *   Constructs a prompt injecting the **User's Current Emotion** and **Interview Phase**.
    *   Queries the LLM (GPT-4) for a structured JSON response containing both *speech* and *timed gestures*.
    *   Commands the robot to speak and act.

## ⚠️ Troubleshooting

*   **"Could not open webcam"**: Ensure no other app (Zoom, Teams) is using the camera. Run `python test_camera.py` to verify.
*   **"Connection Refused"**: Check that the Furhat server is running and the IP address in `furhat_conversation.py` matches (default is `127.0.0.1`).
*   **Thread Safety Errors on macOS**: If the video window crashes, ensure `cv2.imshow` is disabled.

## 👥 Credits
Developed by [Project Group 18 Team: Hao Zhe Lee, Siyu Lu, Peixin Li, Minyao Xu] for the Intelligent Interactive Systems Project (1MD032) HT2025, Uppsala University.
Based on research into Affective Computing and Human-Robot Interaction.
