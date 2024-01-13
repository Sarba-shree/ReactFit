import cv2
from ultralytics import YOLO
import tkinter
from tkinter import messagebox

def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Set up video capture from webcam
    cap = cv2.VideoCapture(0)

    # Array to store detected TVs
    detected_tvs = []

    while True:
        ret, frame = cap.read()

        if ret:
            # Resize the frame to fit the model input size
            resized_frame = cv2.resize(frame, (500, 500))

            # Detect and track objects
            results = model.track(resized_frame, persist=True)

            # Access the results for the first image
            results = results[0]

            # Update the list of detected objects
            current_objects = []
            for obj in results.boxes.numpy():
                if model.names[int(obj.cls)] != 'person':
                    bbox = tuple(obj.xyxy.tolist())
                    current_objects.append(bbox)
                    if bbox not in detected_tvs:
                        # Object is not in the list, alert the user
                        detected_tvs.append(bbox)
                        result = tkinter.messagebox.askyesno(title="Object Detected", message="Do you want to proceed?")
                        if not result:
                            # Code to execute if 'No' is clicked
                            pass
            # Remove objects that are no longer present in the frame
            detected_tvs = [obj for obj in detected_tvs if obj in current_objects]


            # Plot results on the frame
            frame_ = results.plot()

            # Display the frame with tracked objects
            cv2.imshow('YOLOv8 Object Tracking', frame_)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
