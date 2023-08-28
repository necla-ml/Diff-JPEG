from typing import List

import cv2
import torch
import torch.nn as nn
from torch import Tensor


def jpeg_coding(image_rgb: Tensor, compression_strength: Tensor) -> Tensor:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
        compression_strength (Tensor): Compression strength of the shape [B].

    Returns:
        image_rgb_jpeg (Tensor): JPEG coded image of the shape [B, 3, H, W].
    """
    # Check inputs
    assert isinstance(image_rgb, Tensor), f"image_rgb must be a torch tensor, got {type(image_rgb)}."
    assert isinstance(
        compression_strength, Tensor
    ), f"jpeg_quality must be a torch tensor, got {type(compression_strength)}."
    assert image_rgb.ndim == 4, f"image_rgb must be a 4D tensor, got a {image_rgb.ndim}D tensor."
    assert (
        compression_strength.ndim == 1
    ), f"jpeg_quality must be a 1D tensor, got a {compression_strength.ndim}D tensor."
    assert image_rgb.shape[0] == compression_strength.shape[0], (
        f"Batch size between image_rgb and jpeg_quality are not matching. "
        f"Got {image_rgb.shape[0]} and {compression_strength.shape[0]}."
    )
    assert image_rgb.shape[1] == 3, f"image_rgb must be a RGB image, got {image_rgb.shape}."
    # Get shape
    B, _, _, _ = image_rgb.shape
    # Init list to store compressed images
    image_rgb_jpeg: List[Tensor] = []
    # Code each image
    for index in range(B):
        # Make encoding parameter
        encode_parameters = (int(cv2.IMWRITE_JPEG_QUALITY), int(compression_strength[index].item()))
        # Encode image note CV2 is using [B, G, R]
        _, encoding = cv2.imencode(".jpeg", image_rgb[index].flip(0).permute(1, 2, 0).numpy(), encode_parameters)
        image_rgb_jpeg.append(torch.from_numpy(cv2.imdecode(encoding, 1)).permute(2, 0, 1).flip(0))
    # Stack images
    image_rgb_jpeg: Tensor = torch.stack(image_rgb_jpeg, dim=0)
    return image_rgb_jpeg


class JPEGCoding(nn.Module):
    """This class implements JPEG coding."""

    def __init__(
        self,
    ) -> None:
        """Constructor method."""
        # Call super constructor
        super(JPEGCoding, self).__init__()

    def forward(self, image_rgb: Tensor, compression_strength: Tensor) -> Tensor:
        """Forward pass performs JPEG coding.

        Args:
            image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
            compression_strength (Tensor): Compression strength of the shape [B].

        Returns:
            image_rgb_jpeg (Tensor): JPEG coded image of the shape [B, 3, H, W].
        """
        # Perform coding
        image_rgb_jpeg: Tensor = jpeg_coding(
            image_rgb=image_rgb,
            compression_strength=compression_strength,
        )
        return image_rgb_jpeg
