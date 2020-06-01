"""
Collection of utility methods to convert segmentation masks between supported formats.
"""

import numpy as np

def boolean_to_bitwise_masks(masks):
    """ This function takes up to 8 boolean masks (2D vector of 0s and 1s) and combines them into a single bitwise label mask.

        - masks: array of numpy arrays. Contains up to 8 boolean masks
        Returns one mask were each output bit represents one mask.
    """

    # Once you know that you comply with these asserts you could remove them
    assert len(masks) > 0 and len(masks) <= 8, "A maximum of 8 masks can be combined into a uint8 bitwise mask"
    assert np.array(masks).max() == 1, "Masks must be boolean (only 0 and 1)"

    result = masks[0]
    for i, m in enumerate(masks[1:]):
        result += m << (i + 1)
    return result.astype(np.uint8)
