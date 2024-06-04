import cv2
import os
from datetime import datetime, timedelta

from lightglue import LightGlueRunner
from utils import load_image, rgb_to_grayscale
import viz2d

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

extractor_path = "superpoint.onnx"
lightglue_path = "superpoint_lightglue.onnx"
providers = providers

runner = LightGlueRunner(
    extractor_path=extractor_path,
    lightglue_path=lightglue_path,
    providers=providers,
)

# Capture video from camera
cap = cv2.VideoCapture(0)

# Number of frames to skip
n_frames = 10

# Interval for saving comparison images (in seconds)
save_interval = 5

# Get current timestamp
start_time = datetime.now()

# Initialize variable for holding the time of the last saved image
last_save_time = start_time

# Create directory to store comparison images
output_dir = start_time.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(output_dir, exist_ok=True)

# Initialize variable for holding previous frame
prev_frame = None

print("Script started.")

# Loop to capture frames from the camera stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    print("Frame captured.")

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check if it's time to process frames
    if prev_frame is not None:
        # Process frames every n_frames
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % n_frames == 0:
            print("Processing frames.")

            # Run SuperPoint on current and previous frames
            m_kpts0, m_kpts1 = runner.run(prev_frame, gray_frame)

            # Visualize comparison
            comparison_img = viz2d.plot_matches(prev_frame, gray_frame, m_kpts0, m_kpts1, color="lime", lw=0.2)
            
            # Get current time
            current_time = datetime.now()

            # Check if it's time to save the comparison image
            if (current_time - last_save_time).total_seconds() >= save_interval:
                print("Saving comparison image.")
                # Save comparison image
                filename = f"comparison_{current_time.strftime('%H-%M-%S')}.png"
                cv2.imwrite(os.path.join(output_dir, filename), comparison_img)
                print(f"Comparison image saved: {filename} at {current_time.strftime('%H:%M:%S')}")
                last_save_time = current_time  # Update the last saved time

    # Store current frame for next iteration
    prev_frame = gray_frame.copy()

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

print("Script ended.")
