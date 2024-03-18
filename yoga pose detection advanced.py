import cv2
import mediapipe as mp
import numpy as np
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
def detect_landmarks_on_image(image_path):
    image = cv2.imread(image_path)
    mp_pose = mp.solutions.     pose
    pose = mp_pose.Pose()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_image)
    landmarks = results.pose_landmarks

    if landmarks:
        for lm in landmarks.landmark:
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)
    image = cv2.resize(image, (400, 600))

    return image, landmarks
def detect_landmarks_on_webcam(reference_landmarks):
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    screen_width = int(cap.get(3))
    screen_height = int(cap.get(4))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)
        landmarks = results.pose_landmarks
        if landmarks:
            for lm in landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

            accuracy = 0
            for ref_lm, cam_lm in zip(reference_landmarks.landmark, landmarks.landmark):
                ref_point = (int(ref_lm.x * screen_width), int(ref_lm.y * screen_height))
                cam_point = (int(cam_lm.x * screen_width), int(cam_lm.y * screen_height))
                accuracy += (calculate_distance(ref_point, cam_point))

            accuracy /= len(reference_landmarks.landmark)
            accuracy = 1 - accuracy / screen_width  

            cv2.putText(frame, f"Accuracy: {accuracy*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        frame = cv2.resize(frame, (screen_width, screen_height))
        reference_image = cv2.resize(image, (screen_width, screen_height))
        display = np.concatenate((frame, reference_image), axis=1)
        cv2.imshow("Yoga Pose Comparison", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

image_path = "C:/Users/dipto/Downloads/tadasana.jpg"
image, reference_landmarks = detect_landmarks_on_image(image_path)

detect_landmarks_on_webcam(reference_landmarks)
