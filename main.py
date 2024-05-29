import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Set up cooldown period
COOLDOWN_TIME = 2  # seconds
last_action_time = 0

def fingers_up(hand_landmarks):
    # List to hold whether each finger is up (True) or down (False)
    fingers = []

    # Thumb
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(True)
    else:
        fingers.append(False)

    # Index, Middle, Ring, and Pinky fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[mp_hands.HandLandmark(id * 4)].y < hand_landmarks.landmark[mp_hands.HandLandmark(id * 4 - 2)].y:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers

def recognize_gesture(fingers, hand_landmarks):
    if fingers.count(True) == 1:
        return "ONE_UP"
    elif fingers.count(True) == 2:
        return "TWO_UP"
    elif fingers.count(True) == 3:
        return "THREE_UP"
    elif fingers.count(True) == 5:
        return "FIVE_UP"
    elif is_c_shape(hand_landmarks):
        return "C_SHAPE"
    else:
        return "UNKNOWN"

def is_c_shape(hand_landmarks):
    # Check for C shape by comparing distances between thumb and other fingers
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Define a threshold for the C shape
    c_threshold = 0.05
    
    # Check if the distance between thumb tip and other finger tips is less than the threshold
    if (abs(thumb_tip.x - index_tip.x) < c_threshold and
        abs(thumb_tip.y - index_tip.y) < c_threshold):
        return True
    return False

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check which fingers are up
            fingers = fingers_up(hand_landmarks)
            gesture = recognize_gesture(fingers, hand_landmarks)

            # Get current time
            current_time = time.time()

            # Check if enough time has passed since the last action
            if current_time - last_action_time > COOLDOWN_TIME:
                print(gesture)
                if gesture == "ONE_UP":
                    cv2.putText(frame, "Increase Volume", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    pyautogui.press('volumeup')
                    # last_action_time = current_time
                elif gesture == "TWO_UP":
                    cv2.putText(frame, "Decrease Volume", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    pyautogui.press('volumedown')
                    # last_action_time = current_time 
                elif gesture == "THREE_UP":
                    cv2.putText(frame, "Previous Video", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    pyautogui.press('prevtrack')
                    last_action_time = current_time
                elif gesture == "FIVE_UP":
                    cv2.putText(frame, "Pause/Play", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    pyautogui.press('space')
                    last_action_time = current_time
                elif gesture == "C_SHAPE":
                    cv2.putText(frame, "Next Video", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    pyautogui.press('nexttrack')
                    last_action_time = current_time

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
