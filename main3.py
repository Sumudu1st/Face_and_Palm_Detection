import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.3)

# Path to your local video file
video_path = 'Videos/video720.mp4'

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    hands_results = hands.process(frame_rgb)

    # Process the frame with MediaPipe Face Detection
    face_results = face_detection.process(frame)

    # Draw bounding boxes around detected faces
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw bounding boxes around detected hands with forward-facing palms
    if hands_results.multi_hand_landmarks:
        for landmarks in hands_results.multi_hand_landmarks:
            # Extract landmarks for the palm
            palm_landmarks = landmarks.landmark[0:21]

            # Get the x-coordinates of the landmarks
            x_coordinates = [landmark.x for landmark in palm_landmarks]

            # Check if the palm is facing forward (camera side)
            if x_coordinates[0] < x_coordinates[-1]:
                # Draw bounding box around the hand
                bounding_box = cv2.boundingRect(np.array([(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in palm_landmarks], dtype=np.int32))
                x, y, w, h = bounding_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face and Forward-Facing Palm Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
