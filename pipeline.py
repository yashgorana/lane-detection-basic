import numpy as np
import cv2
from functools import partial

# =====================================================================================
# Helper Functions
# =====================================================================================


def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, draw_function=draw_lines, **kwargs):
    """
    `img` should be the output of a Canny transform.

    draw_function: Which which accepts image & line to render lanes. Default: draw_lines()

    Returns an image with hough lines drawn.
    """
    rho = max(rho, 1)
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_function(line_img, lines, **kwargs)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# =====================================================================================
# Custom Functions
# =====================================================================================


def draw_lanes(
    img,
    lines,
    horizon=0.5,
    lane_color=(255, 0, 0),
    lane_thickness=10,
    hough_color=(0, 255, 0),
    hough_thickness=1,
    draw_lanes=True,
    draw_hough_points=False,
    draw_hough_lines=False,
    **kwargs
):
    """Draw averaged, filtered & extrapolated lane lines from hough line

    Arguments:
        img: Source image
        lines: Result of HoughLinesP

        horizon: At what percent does the virtual horizon lie (Default: 0.5 i.e. 50%)

        draw_hough_points: Should draw hough line end-points (Default: False)
        draw_hough_lines: Should draw hough lines (Default: False)
    """

    left_coords = []
    right_coords = []

    max_x = img.shape[1]
    max_y = img.shape[0]
    min_y = int(img.shape[0] * horizon)

    if lines is None or len(lines) == 0:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            # filter out slopes not within the slope range
            m = (y2 - y1) / (x2 - x1)
            if np.isnan(m):
                continue

            # Use "left_coords" list if co-ord lies in the left half of the image
            # else use "right_coords" list
            coords = left_coords if (x1, x2) < (max_x / 2, max_x / 2) else right_coords

            # Append both co-ords to the selected lane list
            coords.append((x1, y1))
            coords.append((x2, y2))

            # Draw hough lines if required
            if draw_hough_lines:
                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color=hough_color,
                    thickness=hough_thickness,
                )

            # Draw hough points if required
            if draw_hough_points:
                radius = hough_thickness * 2
                cv2.circle(
                    img,
                    (x1, y1),
                    radius=radius,
                    color=hough_color,
                    thickness=radius,
                )
                cv2.circle(
                    img,
                    (x2, y2),
                    radius=radius,
                    color=hough_color,
                    thickness=radius,
                )

    # Return if either of co-ord list is empty
    if len(left_coords) == 0 or len(right_coords) == 0:
        return

    # Convert to co-ords to numpy array. Transpose it so that:
    # coords[0] = all x co-ordinates
    # coords[1] = all y co-ordinates
    left_coords = np.array(left_coords).T
    right_coords = np.array(right_coords).T

    # Fit two lines for left_coords & right_coords using "polyfit".
    # These lines are the "averaged/smoothed" line
    # "poly1d" will return a "line-function" for the polyfit
    left_line = np.poly1d(np.polyfit(left_coords[1], left_coords[0], deg=1))
    right_line = np.poly1d(np.polyfit(right_coords[1], right_coords[0], deg=1))

    # Resolve points for averaged line using the "line-function"
    left_p1 = np.array([left_line(min_y), min_y], dtype=np.uint)
    left_p2 = np.array([left_line(max_y), max_y], dtype=np.uint)
    right_p1 = np.array([right_line(min_y), min_y], dtype=np.uint)
    right_p2 = np.array([right_line(max_y), max_y], dtype=np.uint)

    # Draw the averaged lane lines
    if draw_lanes:
        cv2.line(img, left_p1, left_p2, color=lane_color, thickness=lane_thickness)
        cv2.line(img, right_p1, right_p2, color=lane_color, thickness=lane_thickness)

    # DEBUG: vertical mid line
    # cv2.line(img, (int(max_x/2), 0), (int(max_x/2), max_y), (0, 255, 0), thickness=1)


def lane_mask_quad(img, top_width=100, offset_top=0.5):
    """Creates a quadrilateral mask for lane detection"""

    x_size = img.shape[1]
    y_size = img.shape[0]

    offset_center = top_width / 2
    top = y_size * offset_top

    return np.array(
        [
            [
                (0, y_size),
                ((x_size / 2) - offset_center, top),
                ((x_size / 2) + offset_center, top),
                (x_size, y_size),
            ]
        ],
        dtype=np.int32,
    )


def grayscale_channel(img, channel="red"):
    """Returns a channel (grayscale).

    Arguments:
        img: Input Image in RBG color space.
        channel:
            0 = Red
            1 = Green
            2 = Blue
    """
    channels = {
        "red": 0,
        "green": 1,
        "blue": 2,
    }
    channel = channels.get(channel, 0)
    return img[:, :, channel]


def color_select_hsv(img, lower=(0, 0, 0), upper=(179, 255, 255)):
    """Color selection in HSV color space"""
    _img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(_img, lower, upper)
    masked = cv2.bitwise_and(_img, _img, mask=mask)
    return cv2.cvtColor(masked, cv2.COLOR_HSV2RGB)


# =====================================================================================
# Lane Detection Pipeline
# =====================================================================================

from functools import partial


def lane_detection_pipeline(img, **kwargs):
    """Lane Detection Pipeline

    Arguments:
        img: Input image in RGB color space

    Returns: Processed image in RGB color space
    """

    # Default args
    horizon = kwargs.get("horizon", 0.6)
    mask_top_width = kwargs.get("mask_top_width", 80)
    blur_kernel = kwargs.get("blur_kernel", 7)
    output_intermediate = kwargs.get("output_intermediate", False)

    color_select_args = {
        "lower": (0, 0, 210),
        "upper": (179, 255, 255),
    }

    canny_args = {
        "low_threshold": 50,
        "high_threshold": 150,
    }

    hough_args = {
        "rho": 2,
        "theta": np.pi / 180,
        "threshold": 20,
        "min_line_len": 2,
        "max_line_gap": 10,
    }

    lane_args = {
        "horizon": horizon,
        "draw_hough_points": False,
        "draw_hough_lines": False,
    }

    # Update default with overrides
    color_select_args.update(kwargs.get("color_select_hsv", {}))
    canny_args.update(kwargs.get("canny", {}))
    lane_args.update(kwargs.get("lane", {}))
    hough_args.update(kwargs.get("hough", {}))

    # Custom draw_lanes function+args for the "hough_lines" functions to invoke
    hough_args["draw_function"] = partial(draw_lanes, **lane_args)

    # Create a region mask
    region_mask = lane_mask_quad(img, top_width=mask_top_width, offset_top=horizon)

    # Copy reference of original image
    original_img = img

    # Work on a copy of the image
    img = np.copy(img)

    # Run the pipeline
    img = color_select_hsv(img, **color_select_args)
    img = grayscale_channel(img, channel="red")
    img = gaussian_blur(img, kernel_size=blur_kernel)

    # slight modification to capture intermediate state
    if output_intermediate:
        original_img = np.dstack([img, img, img])

    img = canny(img, **canny_args)
    img = region_of_interest(img, vertices=region_mask)
    img = hough_lines(img, **hough_args)
    img = weighted_img(img, initial_img=original_img)

    return img
