# LightGlue Image Matching

This repository contains a script for image matching using pre-trained models. The script processes two images, extracts key points, and visualizes the matching results.

## Requirements

- Python 3.6+
- `onnxruntime`
- `utils` (custom utility functions)
- `viz2d` (custom visualization functions)

## Setup

1. Install required Python packages:

   ```sh
   pip install onnxruntime
   ```


Ensure superpoint.onnx, superpoint_lightglue.onnx utils.py and viz2d.py are in the same directory as your script.

## Usage

Update the paths to your images in the script:

```sh
    img0_path = "1.jpeg"
    img1_path = "2.jpeg"
```

Set the image size for processing (default is 512):

```
img_size = 512
```

Run the script:

```
python3 infer_using_CPU.py
```
