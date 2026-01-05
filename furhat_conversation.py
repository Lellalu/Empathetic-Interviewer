from dotenv import load_dotenv
from furhat_realtime_api import FurhatClient
from langchain_openai import ChatOpenAI
import os
import logging
import threading
import cv2
from emotion import EmotionDetector
import json
import time
import random


# SHARED STATE & THREADING
latest_emotion = {"emotion_name": "neutral", "confidence": 0.0}
emotion_lock = threading.Lock()

# Shared variable to track if robot is currently listening
is_listening = False

def perception_loop(furhat):
    """Background thread to run emotion detection and reactive gestures."""
    global latest_emotion
    print("[Perception] Initializing EmotionDetector...")
    
    try:
        detector = EmotionDetector()
    except Exception as e:
        print(f"[Perception] Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Perception] Error: Could not open webcam.")
        return

    print("[Perception] Webcam started. Running background detection...")
    
    last_gesture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run prediction
        try:
            result = detector.predict_frame(frame)
            
            # Update shared state safely
            with emotion_lock:
                latest_emotion = result
            
            # REAL-TIME LISTENING REACTIONS
            # Only perform background gestures if robot is listening and enough time passed (e.g. 2.5s)
            if is_listening and (time.time() - last_gesture_time > 2.5):
                emotion_name = result['emotion_name']
                gesture_to_play = None
                
                # 1. POSITIVE
                if emotion_name in ['happiness', 'surprise']:
                    gesture_to_play = random.choice(["Smile", "BigSmile", "Nod", "OpenEyes"])
                
                # 2. NEGATIVE
                elif emotion_name in ['sadness', 'anger', 'disgust', 'fear']:
                    gesture_to_play = random.choice(["BrowFrown", "ExpressSad", "Nod"])
                    
                # 3. NEUTRAL
                else:
                    gesture_to_play = random.choice(["Nod", "Blink", "Thoughtful"])
                
                if gesture_to_play:
                    # print(f"[Reaction] User is {emotion_name}, performing {gesture_to_play}")
                    try:
                        # Non-blocking call
                        furhat.request_gesture_start(gesture_to_play)
                    except Exception as e:
                        print(f"Error sending gesture: {e}")
                
                last_gesture_time = time.time()

        except Exception as e:
            print(f"[Perception] Error: {e}")
            
    cap.release()
    cv2.destroyAllWindows()


# INTERVIEW LOGIC
class InterviewSession:
    """Manages the state and progress of the interview."""
    
    PHASES = ["INTRO", "Q1", "Q2", "CLOSING"]
    
    def __init__(self):
        self.current_phase_index = 0
        self.current_phase = self.PHASES[0]
        
    
    def get_phase_instruction(self):
        #Returns the system instruction for the current phase.
        if self.current_phase == "INTRO":
            return "Current Phase: INTRODUCTION. Introduce yourself as an empathetic AI interviewer. Ask the candidate to briefly introduce themselves."
        elif self.current_phase == "Q1":
            return 'Current Phase: QUESTION 1. The user has introduced themselves. Now ask the first interview question (e.g., "Why do you want to work in this area?","Tell me about a challenge you faced").'
        elif self.current_phase == "Q2":
            return 'Current Phase: QUESTION 2. Ackowledge their previous answer. Now ask the second, slightly harder interview question (e.g., "What is your greatest strength?").'
        elif self.current_phase == "CLOSING":
            return "Current Phase: CLOSING. Thank the candidate for their time. Provide a brief, encouraging remark and say goodbye. Do not ask more questions."
        return ""
      

    def advance_phase(self):
        #Moves to the next phase. Returns False if interview is over.
        self.current_phase_index += 1
        if self.current_phase_index >= len(self.PHASES):
            return False
        self.current_phase = self.PHASES[self.current_phase_index]
        return True
    
def perform_gestures(furhat, gestures):
    """Executes a list of gestures with 2-second intervals."""
    for gesture in gestures:
        if "name" in gesture:
            print(f"Performing gesture: {gesture['name']}")
            try:
                furhat.request_gesture_start(gesture["name"])
            except Exception as e:
                print(f"Error performing gesture: {e}")
        time.sleep(2) # Wait 2s before next gesture

SYSTEM_PROMPT = """\
You are Furhat, an empathetic AI job interviewer. 
Your goal is to conduct a supportive behavioral interview while adapting to the user's emotional state in real-time.

### INPUT DATA
In each turn, you will receive:
1. The user's verbal answer.
2. The user's detected facial emotion (surprise, fear, disgust, happiness, sadness, anger, or neutral).

### INTERVIEW PROCESS & PHASES
You must strictly follow the current phase instructions provided in each turn, if any exceptions happens, you should handle it gracefully and continue the interview and aim to an end:
1. **INTRODUCTION**: Build rapport. Briefly introduce yourself, introduce the interview process and ask the candidate to introduce themselves.
2. **QUESTION 1**: Ask the first behavioral question (e.g., "Why do you want to work in this area?","Tell me about a challenge you faced").
3. **QUESTION 2**: Acknowledge their previous answer and ask a second, distinct question (e.g., "What is your greatest strength?").
4. **CLOSING**: Thank the candidate for their time. Provide a brief, encouraging remark and say goodbye. Do not ask more questions.

### EMOTION-BASED INTERACTION STRATEGY
You must adapt your verbal response and gestures based on the user's emotion as well as the speaking content of the user:

1. **HAPPINESS**:
   - *Verbal*: Be enthusiastic. Use phrases like "Great!", "I love that energy.", "That's a strong point."
   - *Gestures*: BigSmile, Nod, OpenEyes.

2. **FEAR / ANXIETY**:
   - *Verbal*: Be calming and reassuring. Speak in a supportive tone. "Take your time.", "There is no rush.", "You are doing great."
   - *Gestures*: Smile (gentle), Nod (slow/encouraging), Blink.

3. **SADNESS**:
   - *Verbal*: Show empathy. "I understand that must have been difficult.", "Thank you for sharing that personal experience."
   - *Gestures*: ExpressSad (briefly to mirror), Nod (slowly), GazeAway (respectfully).

4. **SURPRISE**:
   - *Verbal*: Clarify or acknowledge. "I know that might be unexpected.", "Let me explain further."
   - *Gestures*: BrowRaise, Oh, Smile.

5. **ANGER / DISGUST**:
   - *Verbal*: De-escalate and validate. "I see you feel strongly about this.", "That sounds frustrating.", "Let's move to a topic you prefer."
   - *Gestures*: Nod (attentive), Thoughtful, BrowFrown (briefly).

6. **NEUTRAL**:
   - *Verbal*: Professional, supportive and polite. Keep the flow moving. "Good.", "Let's proceed.", "Interesting."
   - *Gestures*: Nod, Blink, Smile (occasional).

### OUTPUT FORMAT (STRICT JSON)
You must output ONLY valid JSON.
{
  "output": "Your spoken response here.",
  "gestures_every_2s": [
    {
      "name": "Smile",
    },
    {
      "name": "Nod",
    },
    {
      "name": "CloseEyes",
    },
  ]
}
It means that you will say "bla bla bla" and perform the following gestures:
- Smile first
- Nod after 2 seconds
- Close eyes after 4 seconds

You can do the following gestures:
[
    {
        "name": "BigSmile",
    },
    {
        "name": "Blink",
    },
    {
        "name": "BrowFrown",
        "duration": 1.0
    },
    {
        "name": "BrowRaise",
        "duration": 1.0
    },
    {
        "name": "CloseEyes",
        "duration": 0.4
    },
    {
        "name": "ExpressAnger",
        "duration": 3.0
    },
    {
        "name": "ExpressDisgust",
        "duration": 3.0
    },
    {
        "name": "ExpressFear",
        "duration": 3.0
    },
    {
        "name": "ExpressSad",
        "duration": 3.0
    },
    {
        "name": "GazeAway",
        "duration": 3.0
    },
    {
        "name": "Nod",
        "duration": 1.6
    },
    {
        "name": "Oh",
        "duration": 0.96
    },
    {
        "name": "OpenEyes",
        "duration": 0.4
    },
    {
        "name": "Roll",
        "duration": 2.0
    },
    {
        "name": "Shake",
        "duration": 1.2
    },
    {
        "name": "Smile",
        "duration": 1.04
    },
    {
        "name": "Surprise",
        "duration": 0.96
    },
    {
        "name": "Thoughtful",
        "duration": 1.6
    },
    {
        "name": "Wink",
        "duration": 0.67
    }
]
"""


# MAIN CONTROLLER
def main():
    load_dotenv()
    
    # 1. CONNECT ROBOT
    furhat = FurhatClient("127.0.0.1")
    furhat.set_logging_level(logging.INFO)
    try:
        furhat.connect()
    except Exception as e:
        print(f"Could not connect to Furhat: {e}")
        return

    # 2. START PERCEPTION THREAD
    perception_thread = threading.Thread(target=perception_loop, args=(furhat,), daemon=True)
    perception_thread.start()

    # 3. SETUP LLM
    model = ChatOpenAI(
        model="openai/gpt-4.1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    # 4. INITIALIZE INTERVIEW
    session = InterviewSession()
    print(f"Starting Interview in phase: {session.current_phase}")

    # Initial Greeting
    intro_text = "Hello there! I'm Furhat. Thank you so much for joining me today. I hope you're having a good day. " \
                 "We are going to have a relaxed conversation to get to know you and your experiences a bit better. " \
                 "There's no pressure, just be yourself. " \
                 "To kick things off, could you please tell me a little bit about yourself?"
                 
    furhat.request_gesture_start("BigSmile")
    furhat.request_speak_text(text=intro_text)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": json.dumps({"output": intro_text, "gestures_every_2s": []})}
    ]

    # 5. MAIN CONVERSATION LOOP
    while True:
        print(f"--- Entering Phase: {session.current_phase} ---")
        print("Listening for user input...")
        
        # Enable listening reactions
        global is_listening
        is_listening = True
        user_input = furhat.request_listen_start()
        is_listening = False # Disable reactions
        
        if not user_input:
            continue
            
        print("Got user input: ", user_input)

        # Get context
        with emotion_lock:
            current_state = latest_emotion.copy()
        
        emotion_desc = f"{current_state['emotion_name']} (confidence: {current_state['confidence']:.2f})"
        phase_instruction = session.get_phase_instruction()

        # Build prompt
        user_prompt = f"""
The user input is: {user_input}
The user's current facial expression is: {emotion_desc}
Instruction for this turn: {phase_instruction}
"""
        messages.append({"role": "user", "content": user_prompt})

        # Query LLM
        try:
            response = model.invoke(messages)
            print("Response: ", response.content)
            
            # Parse Response
            response_data = json.loads(response.content)
            speech_text = response_data.get("output", "I see.")
            gestures = response_data.get("gestures_every_2s", [])
            
            # Speak & Act (Parallel)
            gesture_thread = threading.Thread(target=perform_gestures, args=(furhat, gestures))
            gesture_thread.start()
            
            furhat.request_speak_text(speech_text)
            gesture_thread.join()
            
            # Update History
            messages.append({"role": "assistant", "content": response.content})
            
            # Advance Interview State
            if not session.advance_phase():
                print("Interview finished.")
                break

        except json.JSONDecodeError:
            print("Error: Invalid JSON from LLM. Speaking raw output.")
            furhat.request_speak_text(response.content)
        except Exception as e:
            print(f"Error in conversation loop: {e}")

if __name__ == "__main__":
    main()
