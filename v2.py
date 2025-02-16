import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Webcam
cap = cv2.VideoCapture(1)  # Set to 1 as requested
cap.set(cv2.CAP_PROP_FPS, 30)  # Set higher FPS for smoother video

# Initialize Whiteboard Canvas
canvas = np.ones((600, 900, 3), dtype=np.uint8) * 255  # White background

# Create separate windows
cv2.namedWindow("Video Feed")
cv2.namedWindow("Whiteboard")

# Screen size for mouse control
screen_w, screen_h = pyautogui.size()

# Variables to track state
prev_x, prev_y = 0, 0
is_drawing = False
is_erasing = False
is_dragging = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract finger tip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            # Convert coordinates for mouse movement
            mouse_x = np.interp(index_x, (0, w), (0, screen_w))
            mouse_y = np.interp(index_y, (0, h), (0, screen_h))
            
            # Move mouse with index finger
            pyautogui.moveTo(mouse_x, mouse_y, duration=0.005)
            
            # Detect pinch gestures
            index_pinch = np.hypot(index_x - thumb_x, index_y - thumb_y) < 40
            drag_pinch = np.hypot(index_x - thumb_x, index_y - thumb_y) < 40 and \
                         np.hypot(middle_x - thumb_x, middle_y - thumb_y) < 40
            eraser_detected = index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
                             middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            
            # Get whiteboard window position
            whiteboard_pos = pyautogui.getWindowsWithTitle("Whiteboard")
            if whiteboard_pos:
                wb_x, wb_y = whiteboard_pos[0].left, whiteboard_pos[0].top
                wb_w, wb_h = whiteboard_pos[0].width, whiteboard_pos[0].height
                
                # Check if mouse is inside the whiteboard window
                if wb_x < mouse_x < wb_x + wb_w and wb_y < mouse_y < wb_y + wb_h:
                    if index_pinch:
                        is_drawing = True
                        draw_x = int(np.interp(mouse_x, (wb_x, wb_x + wb_w), (0, 900)))
                        draw_y = int(np.interp(mouse_y, (wb_y, wb_y + wb_h), (0, 600)))
                        
                        if prev_x == 0 and prev_y == 0:
                            prev_x, prev_y = draw_x, draw_y
                        cv2.line(canvas, (prev_x, prev_y), (draw_x, draw_y), (0, 0, 0), 5)
                        prev_x, prev_y = draw_x, draw_y
                    else:
                        prev_x, prev_y = 0, 0  # Reset when lifting finger
                    
                    if eraser_detected:
                        erase_x = int(np.interp(mouse_x, (wb_x, wb_x + wb_w), (0, 900)))
                        erase_y = int(np.interp(mouse_y, (wb_y, wb_y + wb_h), (0, 600)))
                        cv2.circle(canvas, (erase_x, erase_y), 20, (255, 255, 255), -1)
                    
                    if drag_pinch:
                        pyautogui.mouseDown()
                        is_dragging = True
                    elif is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False
                else:
                    if index_pinch:
                        pyautogui.click()
    
    # Display the separate windows
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Whiteboard", canvas)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
