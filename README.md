# Empathetic Interviewer

Real-time Furhat interviewing system that adapts to the user’s emotional state using a DINOv2 vision model and an LLM.

## 🚀 Key Features
*   **Emotion Perception**: DINOv2 + spatial attention for 7 micro-expressions.
*   **Adaptive Dialogue**: LLM adjusts tone and difficulty.
*   **Live Gestures**: Backchanneling while listening.

## 📂 Project Structure

```
Empathetic-Interviewer/
├── emotion.py                   # VISION MODULE: DINOv2 model definition & real-time inference logic
├── furhat_conversation.py       # MAIN CONTROLLER: Manages Dialogue, Robot Control, and Threading
├── modelTraining.ipynb          # MODEL TRAINING: DINOv2 + spatial attention training notebook (RAF-DB)
├── empathetic_interviewer_perception.pth  # Trained model weights for emotion detection
├── requirements.txt             # Python dependencies
├── test_camera.py               # Utility to verify webcam access
└── README.md                    # This file
```

## 🛠️ Setup
**Requirements:** Python 3.10+, Furhat SDK (or Virtual Skill), webcam, and an LLM API key.

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

## 🏃‍♂️ Run
1. Start Furhat at `127.0.0.1`.
2. Run:
```bash
python furhat_conversation.py
```

## 🧪 Model Training Notebook (`modelTraining.ipynb`)
Trains the emotion model used in `emotion.py` and saves `empathetic_interviewer_perception.pth`. Uses RAF-DB with a DINOv2 backbone and spatial attention.

## 👥 Credits
Developed by [Project Group 18 Team: Hao Zhe Lee, Siyu Lu, Peixin Li, Minyao Xu] for the Intelligent Interactive Systems Project (1MD032) HT2025, Uppsala University.
Based on research into Affective Computing and Human-Robot Interaction.
