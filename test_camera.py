import cv2
import sys

def test_camera():
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Please check your permissions.")
        return

    print("Camera opened successfully!")
    ret, frame = cap.read()
    
    if ret:
        print("Successfully captured a frame.")
        print(f"Frame shape: {frame.shape}")
    else:
        print("Error: Could not read frame.")
    
    cap.release()
    print("Camera released.")

if __name__ == "__main__":
    test_camera()

