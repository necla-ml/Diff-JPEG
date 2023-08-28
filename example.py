import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from diff_jpeg import DiffJPEGCoding
from non_diff_jpeg import jpeg_coding_cv2

JPEG_QUALITY: Tensor = torch.tensor([1.0])


def main() -> None:
    # Load test image and reshape to [B, 3, H, W]
    image: Tensor = torchvision.io.read_image("test_images/test_image.png").float()[None]
    # Init differentiable JPEG module
    diff_jpeg_module: nn.Module = DiffJPEGCoding(ste=True)
    # Perform differentiable JPEG coding
    image_coded_diff: Tensor = diff_jpeg_module(
        image,
        JPEG_QUALITY,
        quantization_table_y=torch.randint(low=1, high=256, size=(8, 8)),
        quantization_table_c=torch.randint(low=1, high=256, size=(8, 8)),
    )
    # Perform non-differentiable JPEG coding
    image_coded_non_diff: Tensor = jpeg_coding_cv2(image, JPEG_QUALITY)
    # Compute mean L1 distance
    print((image_coded_diff - image_coded_non_diff).abs().mean().item())


if __name__ == "__main__":
    main()
