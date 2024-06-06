import argparse
from typing import List
import cv2
import os
from lightglue import LightGlueRunner
from utils import load_image, rgb_to_grayscale
import viz2d
import time
import matplotlib.pyplot as plt

# Initialize paths and providers
extractor_path = "/home/user/Light/LightGlue-ONNX/weights/superpoint.onnx"
lightglue_path = "/home/user/Light/LightGlue-ONNX/weights/superpoint_lightglue.onnx"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

runner = LightGlueRunner(
    extractor_path=extractor_path,
    lightglue_path=lightglue_path,
    providers=providers,
)

video_path = "video.mp4"
output_dir = "frames"
img_size = 512

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Extract frames from video
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_index = 0

save_nth_frame = 10
print("CREATING DATASET FROM THE VIDEO -- -- -- -- --")
print("Saving every", save_nth_frame, "th frame")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_index % save_nth_frame == 0:
        frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
    frame_index += 1

cap.release()


def process_frames(frame1_path, frame2_path):
    if isinstance(img_size, List):
        if len(img_size) == 1:
            size0 = size1 = img_size[0]
        elif len(img_size) == 2:
            size0 = size1 = img_size
        elif len(img_size) == 4:
            size0, size1 = img_size[:2], img_size[2:]
        else:
            raise ValueError("Invalid img_size. Please provide 1, 2, or 4 integers.")
    else:
        size0 = size1 = img_size

    image0, scales0 = load_image(frame1_path, resize=size0)
    image1, scales1 = load_image(frame2_path, resize=size1)

    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)

    m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)
    return m_kpts0, m_kpts1


print("DATASET GENERATED WITH", count, "images")
print()
print("Starting Inference")

batch_size = 30
counter = 1
batch_start_time = time.time()

# Process each consecutive pair of frames
for i in range(count - 1):
    frame1_path = os.path.join(output_dir, f"frame_{i}.jpg")
    frame2_path = os.path.join(output_dir, f"frame_{i + 1}.jpg")

    m_kpts0, m_kpts1 = process_frames(frame1_path, frame2_path)

    if counter % batch_size == 0:


        batch_end_time = time.time()
        batch_process_time = batch_end_time - batch_start_time

        frames_per_sec = batch_size  / batch_process_time
        print(f"Batch {counter // batch_size} process time: {batch_process_time} seconds  |  Batch Size: ", batch_size, "+",batch_size, "   |   FPS: " , frames_per_sec)

        orig_image0, _ = load_image(frame1_path)
        orig_image1, _ = load_image(frame2_path)
        fig, ax = viz2d.plot_images(
            [orig_image0[0].transpose(1, 2, 0), orig_image1[0].transpose(1, 2, 0)]
        )
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2, axes=ax)
        output_path = os.path.join(output_dir, f"matches_plot_{os.path.basename(frame1_path).split('.')[0]}_{os.path.basename(frame2_path).split('.')[0]}.png")
        viz2d.save_plot(output_path)
        plt.close(fig)


        # Reset the timer for the next batch
        batch_end_time = batch_start_time = time.time()

    counter += 1

print("All frames processed!")
