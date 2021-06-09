import cv2
import numpy as np

import pipeline as p


def noop(*args, **kwargs):
    pass


class HsvSlider:
    @staticmethod
    def create():
        cv2.createTrackbar("Hue[L]", "controls", 0, 180, noop)
        cv2.createTrackbar("Hue[U]", "controls", 0, 180, noop)
        cv2.createTrackbar("Sat[L]", "controls", 0, 255, noop)
        cv2.createTrackbar("Sat[U]", "controls", 0, 255, noop)
        cv2.createTrackbar("Val[L]", "controls", 0, 255, noop)
        cv2.createTrackbar("Val[U]", "controls", 0, 255, noop)

        # Set default value for MAX HSV trackbars.
        cv2.setTrackbarPos("Hue[H]", "controls", 0)
        cv2.setTrackbarPos("Sat[L]", "controls", 0)
        cv2.setTrackbarPos("Val[L]", "controls", 0)
        cv2.setTrackbarPos("Hue[U]", "controls", 179)
        cv2.setTrackbarPos("Sat[U]", "controls", 255)
        cv2.setTrackbarPos("Val[U]", "controls", 255)

    @staticmethod
    def get_values():
        # get current positions of all trackbars
        h_min = cv2.getTrackbarPos("Hue[L]", "controls")
        s_min = cv2.getTrackbarPos("Sat[L]", "controls")
        v_min = cv2.getTrackbarPos("Val[L]", "controls")
        h_max = cv2.getTrackbarPos("Hue[U]", "controls")
        s_max = cv2.getTrackbarPos("Sat[U]", "controls")
        v_max = cv2.getTrackbarPos("Val[U]", "controls")
        return np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])


class CannySliders:
    @staticmethod
    def create():
        cv2.createTrackbar("Canny[L]", "controls", 10, 100, noop)
        cv2.createTrackbar("Canny[U]", "controls", 20, 300, noop)

        cv2.setTrackbarPos("Canny[L]", "controls", 50)
        cv2.setTrackbarPos("Canny[U]", "controls", 150)

    @staticmethod
    def get_values():
        return (
            cv2.getTrackbarPos("Canny[L]", "controls"),
            cv2.getTrackbarPos("Canny[U]", "controls"),
        )


class HoughSliders:
    @staticmethod
    def create():
        cv2.createTrackbar("HRho", "controls", 1, 10, noop)
        cv2.createTrackbar("HThresh", "controls", 1, 200, noop)
        cv2.createTrackbar("HMinLen", "controls", 1, 200, noop)
        cv2.createTrackbar("HMaxGap", "controls", 1, 200, noop)

        cv2.setTrackbarPos("HRho", "controls", 2)
        cv2.setTrackbarPos("HThresh", "controls", 20)
        cv2.setTrackbarPos("HMinLen", "controls", 2)
        cv2.setTrackbarPos("HMaxGap", "controls", 10)

    @staticmethod
    def get_values():
        return (
            cv2.getTrackbarPos("HRho", "controls"),
            cv2.getTrackbarPos("HThresh", "controls"),
            cv2.getTrackbarPos("HMinLen", "controls"),
            cv2.getTrackbarPos("HMaxGap", "controls"),
        )


class OtherSliders:
    @staticmethod
    def create():
        cv2.createTrackbar("Blur", "controls", 1, 15, noop)
        cv2.createTrackbar("LaneW", "controls", 1, 15, noop)

        cv2.setTrackbarPos("Blur", "controls", 7)
        cv2.setTrackbarPos("LaneW", "controls", 2)

    @staticmethod
    def get_values():
        return (cv2.getTrackbarPos("Blur", "controls"), cv2.getTrackbarPos("LaneW", "controls"))


def main():
    # Create a window
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)

    # Add UI components
    HsvSlider.create()
    CannySliders.create()
    HoughSliders.create()
    OtherSliders.create()

    target_width = 600

    # Load in images
    images = [
        cv2.imread("test_images/solidYellowCurve2.jpg"),
        cv2.imread("test_images/solidWhiteRight.jpg"),
        cv2.imread("test_images/whiteCarLaneSwitch.jpg"),
        cv2.imread("test_images/challenge/excess-shadow.png"),
        cv2.imread("test_images/challenge/high-brightness.png"),
        cv2.imread("test_images/challenge/yellow-curve.png"),
        cv2.imread("test_images/challenge/yellow-shadow.png"),
    ]

    while True:
        # Get slider values
        lower, upper = HsvSlider.get_values()
        canny_low, canny_high = CannySliders.get_values()
        rho, threshold, min_len, max_gap = HoughSliders.get_values()
        (blur, lane_width) = OtherSliders.get_values()

        # Auto-adjust slider values
        blur = max((blur - 1) if blur % 2 == 0 else blur, 1)
        lane_width = max(lane_width, 1)

        # run lane detection pipeline for all images
        for idx, image in enumerate(images, start=1):

            if image is None:
                continue

            _img = np.copy(image)
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

            _img = p.lane_detection_pipeline(
                _img,
                blur_kernel=blur,
                color_select_hsv={
                    "lower": lower,
                    "upper": upper,
                },
                canny={
                    "low_threshold": canny_low,
                    "high_threshold": canny_high,
                },
                hough={
                    "rho": rho,
                    "theta": np.pi / 180,
                    "threshold": threshold,
                    "min_line_len": min_len,
                    "max_line_gap": max_gap,
                },
                lane={
                    "lane_thickness": lane_width,
                    "draw_deadzone": True,
                    "draw_hough_points": True,
                    "draw_hough_lines": True,
                },
                output_intermediate=True,
            )

            # create a smaller image from the output
            img_width = _img.shape[1]
            multiplier = target_width / img_width
            _img = cv2.resize(_img, (0, 0), fx=multiplier, fy=multiplier)
            # convert to cv2 compatible color space
            _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
            # and render the image
            cv2.imshow(f"rendered{idx}", _img)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
