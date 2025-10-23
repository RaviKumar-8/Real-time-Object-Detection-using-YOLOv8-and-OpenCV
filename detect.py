import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
# 'yolov8s.pt' is the 'small' model. It's a good balance of speed and accuracy.
# We are using 's' (small) instead of 'n' (nano) for better accuracy.
model = YOLO('yolov8s.pt')

# Start the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened. Starting detection...")
print("Press 'q' in the video window to quit.")

# Loop to process each frame from the video stream
while True:
    # Capture one frame from the webcam
    success, frame = cap.read()

    if success:
        # Give the frame to the model to detect objects
        # stream=True is more efficient for video processing
        results = model(frame, stream=True)

        # Loop through the detected results
        for r in results:
            boxes = r.boxes  # Get the bounding boxes

            for box in boxes:
                # 1. Bounding Box Coordinates
                # (x1, y1) -> top-left, (x2, y2) -> bottom-right
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer

                # 2. Confidence (How sure the model is)
                confidence = float(box.conf[0]) * 100  # Show as percentage

                # 3. Class ID (E.g., 0 for 'person', 1 for 'bicycle', etc.)
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]  # Get the class name from the ID

                # Only draw the box if confidence is above a threshold (e.g., 50%)
                if confidence > 50:
                    # Draw a rectangle on the frame using OpenCV
                    # BGR color format: (0, 255, 0) is Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display the class name and confidence
                    label = f'{class_name}: {confidence:.2f}%'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the processed frame in a window
        cv2.imshow("Real-time Object Detection", frame)

        # If the 'q' key is pressed, break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # If a frame could not be captured, break the loop
        print("Failed to capture frame.")
        break

# Release all resources
cap.release()
cv2.destroyAllWindows()
print("Detection stopped. Window closed.")