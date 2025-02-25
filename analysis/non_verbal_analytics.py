from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'])
        print(analysis)  # Log this for GPT processing
    except:
        print("Face not detected")

    cv2.imshow("Emotion Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
