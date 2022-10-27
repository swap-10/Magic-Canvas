import cv2
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
                if (distance(landmarks[5], landmarks[0]) < 2 * distance(landmarks[12], landmarks[9])):
                    if len(current) > 0:
                        history.append(current)
                    current = []
                    pass
                else:
                    pointer = [p for p in hand_landmarks.landmark][8]
                    current.append(pointer)
        
        for trail in history:
            for i in range(len(trail)-1):
                cv2.line(img,(int(w*trail[i].x), int(h*trail[i].y)), (int(w*trail[i+1].x), int(h*trail[i+1].y)), (255, 0, 0), 5)
        for i in range(len(current)-1):
            cv2.line(img,(int(w*current[i].x), int(h*current[i].y)), (int(w*current[i+1].x), int(h*current[i+1].y)), (255, 0, 0), 5)

        cv2.imshow('Mediapipe Hands', cv2.flip(img, 1))

        if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()