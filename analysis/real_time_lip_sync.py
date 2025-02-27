from Wav2Lip.inference import lip_sync_accuracy
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    lip_sync_score = lip_sync_accuracy(frame)
    print(f"Lip Sync Accuracy: {lip_sync_score}")

    cv2.imshow("Lip Sync Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
