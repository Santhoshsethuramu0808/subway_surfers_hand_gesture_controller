import cv2
import mediapipe as mp
import pyautogui
import time
import webbrowser

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open the game URL
game_url = "https://poki.com/en/g/subway-surfers"
webbrowser.open(game_url)
time.sleep(5)  # Give the game time to load

# Set up video capture
cap = cv2.VideoCapture(0)
tipIds = [4, 8, 12, 16, 20]

gesture_history = {"jump": 0, "left": 0, "right": 0, "click_w": 0, "click_s": 0}

# Main loop for hand detection and gesture recognition
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        frame = cv2.flip(frame, 1)  # Flip frame for a mirror view
        height, width, _ = frame.shape

        # Convert the BGR image to RGB and process it
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw landmarks and process gestures
        hand_cor_list_right = []
        hand_cor_list_left = []
        if results.multi_hand_landmarks:
            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[hand_no].classification[0].label
                hand_cor_list = []
                
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    hand_cor_list.append([id, cx, cy])
                
                # Separate left and right hands
                if hand_type == 'Left':
                    hand_cor_list_left = hand_cor_list
                else:
                    hand_cor_list_right = hand_cor_list

                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Detect gestures
        fingers_right = [0] * 5
        fingers_left = [0] * 5

        if hand_cor_list_right:
            # Detect thumb and fingers for the right hand
            fingers_right[0] = int(hand_cor_list_right[tipIds[0]][1] < hand_cor_list_right[tipIds[0] - 1][1])
            for i in range(1, 5):
                fingers_right[i] = int(hand_cor_list_right[tipIds[i]][2] < hand_cor_list_right[tipIds[i] - 2][2])

        if hand_cor_list_left:
            # Detect thumb and fingers for the left hand
            fingers_left[0] = int(hand_cor_list_left[tipIds[0]][1] > hand_cor_list_left[tipIds[0] - 1][1])
            for i in range(1, 5):
                fingers_left[i] = int(hand_cor_list_left[tipIds[i]][2] < hand_cor_list_left[tipIds[i] - 2][2])

        # Action conditions based on gestures
        # Jump: Both hands fully open
        if sum(fingers_right) == 5 and sum(fingers_left) == 5:
            gesture_history["jump"] += 1
            if gesture_history["jump"] >= 3:  # Confirm gesture with 3 frames
                pyautogui.press('space')
                print("Jump Triggered")
                gesture_history["jump"] = 0  # Reset counter after action
        else:
            gesture_history["jump"] = 0

        # Move Left: Left hand open, Right hand closed (swapped)
        if sum(fingers_left) == 5 and sum(fingers_right) == 0:
            gesture_history["left"] += 1
            if gesture_history["left"] >= 3:
                pyautogui.press('left')
                print("Move Left Triggered")
                gesture_history["left"] = 0
        else:
            gesture_history["left"] = 0

        # Move Right: Right hand open, Left hand closed (swapped)
        if sum(fingers_right) == 5 and sum(fingers_left) == 0:
            gesture_history["right"] += 1
            if gesture_history["right"] >= 3:
                pyautogui.press('right')
                print("Move Right Triggered")
                gesture_history["right"] = 0
        else:
            gesture_history["right"] = 0

        # Click 'W': One finger (index) up on either hand
        if (fingers_right.count(1) == 1 and fingers_right[1] == 1) or (fingers_left.count(1) == 1 and fingers_left[1] == 1):
            gesture_history["click_w"] += 1
            if gesture_history["click_w"] >= 3:  # Confirm gesture with 3 frames
                pyautogui.press('w')
                print("W Key Click Triggered")
                gesture_history["click_w"] = 0
        else:
            gesture_history["click_w"] = 0

        # Click 'S': Two fingers (index and middle) up on either hand
        if (fingers_right.count(1) == 2 and fingers_right[1] == 1 and fingers_right[2] == 1) or \
           (fingers_left.count(1) == 2 and fingers_left[1] == 1 and fingers_left[2] == 1):
            gesture_history["click_s"] += 1
            if gesture_history["click_s"] >= 3:  # Confirm gesture with 3 frames
                pyautogui.press('s')
                print("S Key Click Triggered")
                gesture_history["click_s"] = 0
        else:
            gesture_history["click_s"] = 0

        # Display the video frame
        cv2.imshow('Hand Detection', frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
