import cv2
import numpy as np
import time

def webcam_capture():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Set the frame rate to 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Calculate the delay needed to achieve 30 FPS
    frame_delay = 1 / 30

    while True:
        start_time = time.time()

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize the frame to 640x480
            resized = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)

            # Display the resulting frame
            cv2.imshow('Grayscale Webcam Feed (30 FPS)', resized)

            # Calculate the time taken for processing
            process_time = time.time() - start_time

            # Wait for the remaining time to achieve 30 FPS
            if process_time < frame_delay:
                time.sleep(frame_delay - process_time)

            # Press 'q' to exit the video stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture frame")
            break

    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_capture()