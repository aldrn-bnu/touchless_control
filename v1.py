import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up webcam capture
cap = cv2.VideoCapture(0)

# Initialize hands object
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    pinch_active = False  # Flag to keep track of pinch gesture status

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks and detect pinching gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the landmarks of thumb tip and index finger tip 
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calculate the Euclidean distance between the thumb tip and index finger tip
                distance_index_thumb = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2 + (thumb_tip.z - index_finger_tip.z) ** 2) ** 0.5
                distance_middle_thumb = ((thumb_tip.x - middle_finger_tip.x) ** 2 + (thumb_tip.y - middle_finger_tip.y) ** 2 + (thumb_tip.z - middle_finger_tip.z) ** 2) ** 0.5


                # ethra adth ethumbo pinching nn
                pinch_threshold = 0.08
                middle_pinch_threshold=0.08

                # Calculate the midpoint between thumb tip and index finger tip
                mid_x = int((thumb_tip.x + index_finger_tip.x) / 2 * frame.shape[1])
                mid_y = int((thumb_tip.y + index_finger_tip.y) / 2 * frame.shape[0])
                
                # Draw a circle at the midpoint flag pole vekkan
                cv2.circle(frame, (mid_x, mid_y), 10, (255, 0, 0), -1)

                # Move the mouse cursor to the midpoint
                screen_width, screen_height = pyautogui.size()
                cursor_x = int(mid_x / frame.shape[1] * screen_width)
                cursor_y = int(mid_y / frame.shape[0] * screen_height)
                pyautogui.moveTo(cursor_x, cursor_y)

                #saadha_pinch
                if distance_index_thumb < pinch_threshold:
                    if not pinch_active:
                        pyautogui.mouseDown()
                        pinch_active = True
                else:
                    if pinch_active:
                        pyautogui.mouseUp()
                        pinch_active = False

                #middle_finger_pinch
                if distance_middle_thumb<middle_pinch_threshold:
                    pyautogui.click()

        else:
            if pinch_active:
                pyautogui.mouseUp()
                pinch_active = False

        # Display the frame
        cv2.imshow('Hsample saanm', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
