from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('./model/best.pt')  # Update path if needed

# Open webcam (0 = default cam)
{% comment %} cap = cv2.VideoCapture(0) {% endcomment %}
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF


if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run detection
    results = model.predict(frame, stream=True, conf=0.5)

    # Display results
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("License Plate Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()