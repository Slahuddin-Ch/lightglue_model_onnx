import argparse

from typing import List

from lightglue import LightGlueRunner
from utils import load_image, rgb_to_grayscale
import viz2d


providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


extractor_path = "superpoint.onnx"
lightglue_path = "superpoint_lightglue.onnx"
providers= providers


runner = LightGlueRunner(
        extractor_path=extractor_path,
        lightglue_path=lightglue_path,
        providers=providers,
    )
img0_path = "/home/user/Light/LightGlue/assets/DSC_0411.JPG"
img1_path = "/home/user/Light/LightGlue/assets/DSC_0410.JPG"


img0_path = "1.jpeg"
img1_path = "2.jpeg"


img_size = 512


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


image0, scales0 = load_image(img0_path, resize=size0)
image1, scales1 = load_image(img1_path, resize=size1)

image0 = rgb_to_grayscale(image0)
image1 = rgb_to_grayscale(image1)


m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)

orig_image0, _ = load_image(img0_path)
orig_image1, _ = load_image(img1_path)
viz2d.plot_images(
    [orig_image0[0].transpose(1, 2, 0), orig_image1[0].transpose(1, 2, 0)]
    )
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.plt.show()
print ("Completed!")