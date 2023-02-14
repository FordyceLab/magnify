import cv2 as cv
import numpy as np
import numpy.ma as ma

from magnify import utils
from magnify.assay import Assay


def segment_buttons(
    assay: Assay, subimage_length: int = 61, min_button_radius=4, max_button_radius=15
) -> Assay:
    num_rows, num_cols = assay.centers.shape[:2]
    image = assay.images[0]

    # Create the subimages array.
    assay.subimages = np.empty(
        (num_rows, num_cols, subimage_length, subimage_length), dtype=image.dtype
    )
    assay.subimage_offsets = np.empty((num_rows, num_cols, 2), dtype=int)
    for i in range(num_rows):
        for j in range(num_cols):
            top, bottom, left, right = utils.bounding_box(
                round(assay.centers[i, j, 0]),
                round(assay.centers[i, j, 1]),
                subimage_length,
            )
            assay.subimages[i, j] = image[top:bottom, left:right]
            assay.subimage_offsets[i, j] = top, left

    # Compute the foreground and background masks for all buttons.
    assay.fg_values = ma.array(assay.subimages, copy=False)
    assay.bg_values = ma.array(assay.subimages, copy=False)
    for i in range(num_rows):
        for j in range(num_cols):
            subimage = utils.to_uint8(assay.subimages[i, j])
            # Filter the subimage to smooth edges and remove noise.
            filtered = cv.bilateralFilter(
                subimage,
                d=9,
                sigmaColor=75,
                sigmaSpace=75,
                borderType=cv.BORDER_DEFAULT,
            )

            # Find any circles in the subimage.
            circles = cv.HoughCircles(
                filtered,
                method=cv.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=20,
                param2=5,
                minRadius=min_button_radius,
                maxRadius=max_button_radius,
            )

            # Update our estimate of the button position if we found some circles.
            if circles is not None:
                # Change circle locations to use row-column indexing.
                circles = circles[0, :, [1, 0]]
                # Use the circle center closest to our previous estimate of the button.
                closest_idx = np.argmin(
                    np.linalg.norm(circles - assay.centers[i, j], axis=1)
                )
                assay.centers[i, j] = (
                    circles[closest_idx] + assay.subimage_offsets[i, j]
                )

            center = (
                np.round(assay.centers[i, j]).astype(int) - assay.subimage_offsets[i, j]
            )

            # Set the foreground (the button) to be a circle of fixed radius.
            fg_mask = utils.circle(
                subimage_length,
                row=center[0],
                col=center[1],
                radius=max_button_radius,
                value=True,
            )

            # Set the background to be the annulus around our foreground.
            bg_mask = utils.circle(
                subimage_length,
                row=center[0],
                col=center[1],
                radius=2 * max_button_radius,
                value=True,
            )
            bg_mask &= ~fg_mask

            # Refine the foreground & background by finding areas within that are bright and dim.
            _, bright_mask = cv.threshold(
                subimage, thresh=0, maxval=1, type=cv.THRESH_BINARY + cv.THRESH_OTSU
            )
            dim_mask = ~cv.dilate(
                bright_mask, np.ones((max_button_radius, max_button_radius))
            )
            bright_mask = bright_mask.astype(bool)
            dim_mask = dim_mask.astype(bool)

            # If part of the button is bright then set the foreground to that bright area.
            if np.any(fg_mask & bright_mask):
                fg_mask &= bright_mask

            # The background on the other hand should not be bright.
            if np.any(bg_mask & dim_mask):
                bg_mask &= dim_mask

            assay.fg_values[i, j, ~fg_mask] = ma.masked
            assay.bg_values[i, j, ~bg_mask] = ma.masked

    return assay
