import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

history = []
current = []

def distance(landmark_1, landmark_2):
    dist = (landmark_2.y - landmark_1.y)**2
    dist += (landmark_2.x - landmark_1.x)**2
    dist = (dist)**(1/2)

    return dist

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as hands:
    whiteboard = np.full(shape=(800, 1400, 3), fill_value=255).astype(np.uint8)
    pointer = 0
    pointers = []
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        results = hands.process(img)
        
        # Draw annotations
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                landmarks = [p for p in hand_landmarks.landmark]
                pointer = landmarks[8]
                pointers.append(pointer)
                k_mov_avg = 5
                if len(pointers) > k_mov_avg:
                    pointers = pointers[-k_mov_avg:]
                if len(pointers) == k_mov_avg:
                    pointer.x = sum([p.x for p in pointers]) / len(pointers)
                    pointer.y = sum([p.y for p in pointers]) / len(pointers)
                if (distance(landmarks[5], landmarks[0]) < 2 * distance(landmarks[12], landmarks[9])):
                    if len(current) > 0:
                        history.append(current)
                    current = []
                    pass
                else:
                    current.append(pointer)
        
        w = 1400
        h = 800
        '''
        for trail in history:
            for i in range(len(trail)-1):
                cv2.line(whiteboard,(int(w*trail[i].x), int(h*trail[i].y)), (int(w*trail[i+1].x), int(h*trail[i+1].y)), (255, 0, 0), 5)
        '''
        for i in range(len(current)-1):
            cv2.line(whiteboard,(int(w*current[i].x), int(h*current[i].y)), (int(w*current[i+1].x), int(h*current[i+1].y)), (255, 0, 0), 5)

        whiteboard_cur = whiteboard.copy()
        if results.multi_hand_landmarks:
            cv2.circle(whiteboard_cur, (int(w*pointer.x), int(h*pointer.y)), 5, (0, 255, 0), 3)
            cv2.circle(whiteboard_cur, (int(w*pointer.x), int(h*pointer.y)), 1, (0, 255, 0), -1)

        cv2.imshow('Mediapipe Hands', cv2.flip(img, 1))
        cv2.imshow("Whiteboard", cv2.flip(whiteboard_cur, 1))

        if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()