import cv2
import numpy as np

def get_chessboard(board_size, screenshot_path="/tmp/screenshot.png", template_path="./templ1.png", output_path="extracted_chessboard.png"):
    """
    Extracts the chessboard region from a screenshot image using back projection.

    Parameters:
    - screenshot_path (str): Path to the screenshot image.
    - template_path (str): Path to the template image for histogram back projection.
    - output_path (str): Path where the extracted chessboard image will be saved.

    Returns:
    - chessboard_region (ndarray): Extracted chessboard region as an OpenCV image.
    - coordinates (tuple): Coordinates (x, y, w, h) of the detected chessboard.
    """
    # Load the screenshot and template images
    image = cv2.imread(screenshot_path)
    template_image = cv2.imread(template_path)

    if image is None or template_image is None:
        raise FileNotFoundError("Screenshot or template image not found. Check the file paths.")

    # Convert both images to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2HSV)

    # Calculate the histogram for the template (ROI)
    roi_hist = cv2.calcHist([hsv_template], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Perform back projection on the screenshot image
    back_proj = cv2.calcBackProject([hsv_image], [0, 1], roi_hist, [0, 180, 0, 256], scale=1)

    # Apply a convolution filter to smooth the result
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    back_proj = cv2.filter2D(back_proj, -1, kernel)

    # Threshold the back projection to create a binary mask
    _, thresh = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found. Unable to detect the chessboard.")

    # Find the largest contour, assumed to be the chessboard
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract the chessboard region from the original image
    tmp_img = image[y:y+h, x:x+w]

    chessboard_region = cv2.resize(tmp_img, board_size, interpolation=cv2.INTER_LINEAR)

    # Save the extracted chessboard image
    cv2.imwrite(output_path, chessboard_region)

    # Return the extracted region and its coordinates
    return chessboard_region, (x, y, w, h)
