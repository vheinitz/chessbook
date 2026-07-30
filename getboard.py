import cv2
import numpy as np


def get_chessboard(board_size, screenshot_path="/tmp/screenshot.png",
                   template_path="./templ1.png", output_path="extracted_chessboard.png"):
    """
    Extracts the chessboard region from a screenshot image.

    Uses OpenCV's findChessboardCorners as the primary detection method,
    with an energy-based fallback.

    Parameters:
    - board_size: Tuple (width, height) to resize the extracted board to.
    - screenshot_path: Path to the screenshot image.
    - template_path: Unused (kept for backwards compatibility).
    - output_path: Path where the extracted chessboard image will be saved.

    Returns:
    - chessboard_region: Extracted chessboard region as an OpenCV image.
    - coordinates: (x, y, w, h) of the detected chessboard in the screenshot.
    """
    image = cv2.imread(screenshot_path)
    if image is None:
        raise FileNotFoundError(f"Screenshot image not found at: {screenshot_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strategy 1: OpenCV's built-in chessboard corner detector
    board_rect = _detect_by_corners(gray)
    if board_rect is not None:
        x, y, bw, bh = board_rect
    else:
        # Strategy 2: Energy-based detection (fallback)
        board_rect = _detect_by_energy(gray)
        if board_rect is not None:
            x, y, bw, bh = board_rect
        else:
            raise ValueError("Could not detect a chessboard in the screenshot. "
                             "Make sure a chessboard is visible on screen.")

    # Extract the chessboard region
    chessboard_region = image[y:y+bh, x:x+bw]

    # Resize to the expected board dimensions
    chessboard_region = cv2.resize(chessboard_region, board_size,
                                   interpolation=cv2.INTER_LINEAR)

    # Save the extracted chessboard image
    cv2.imwrite(output_path, chessboard_region)

    return chessboard_region, (x, y, bw, bh)


def _detect_by_corners(gray):
    """
    Detect chessboard using OpenCV's findChessboardCorners.
    Tries multiple pattern sizes to find the inner corners of the board.
    Returns (x, y, w, h) or None.
    """
    for pattern in [(7, 7), (6, 6), (5, 5)]:
        ret, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )
        if ret and corners is not None:
            # Refine corner positions for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            corners = corners.reshape(-1, 2)
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # Expand from inner corners to full board:
            # For (n,n) inner corners, there are (n+1)*(n+1) squares
            # Each gap between corners = 1 square width
            gap_x = (x_max - x_min) / (pattern[0] - 1)
            gap_y = (y_max - y_min) / (pattern[1] - 1)

            # Add padding for the outer squares (half square on each side)
            pad = 5  # extra pixels for the board border
            x1 = max(0, int(x_min - gap_x - pad))
            y1 = max(0, int(y_min - gap_y - pad))
            x2 = min(gray.shape[1], int(x_max + gap_x + pad))
            y2 = min(gray.shape[0], int(y_max + gap_y + pad))

            w = x2 - x1
            h = y2 - y1

            if w > 100 and h > 100:
                return (x1, y1, w, h)

    return None


def _detect_by_energy(gray):
    """
    Fallback: detect chessboard using gradient energy + connected components.
    """
    h, w = gray.shape

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    energy = cv2.GaussianBlur(grad_mag, (31, 31), 0)

    for pct in [90, 88, 92, 85, 95]:
        threshold = np.percentile(energy, pct)
        _, high_energy = cv2.threshold(energy, threshold, 255, cv2.THRESH_BINARY)
        high_energy = high_energy.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            high_energy, connectivity=8)

        best = None
        best_area = 0
        for i in range(1, num_labels):
            x, y_st, bw, bh, area = stats[i]
            if area < 5000:
                continue
            ratio = bw / bh if bh > 0 else 0
            if 0.7 < ratio < 1.5 and area > best_area:
                best_area = area
                best = (x, y_st, bw, bh)

        if best is not None:
            # Try to refine by finding inner board contour
            refined = _refine_contour(gray, best)
            return refined if refined is not None else best

    return None


def _refine_contour(gray, rect):
    """Refine the board region by finding the inner rectangular contour."""
    region_x, region_y, region_w, region_h = rect
    margin = 15
    x1 = max(0, region_x - margin)
    y1 = max(0, region_y - margin)
    x2 = min(gray.shape[1], region_x + region_w + margin)
    y2 = min(gray.shape[0], region_y + region_h + margin)

    sub_img = gray[y1:y2, x1:x2]
    edges = cv2.Canny(sub_img, 30, 100)
    cts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for c in cts:
        area = cv2.contourArea(c)
        if area < 3000:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        ratio = rw / rh if rh > 0 else 0
        fill = area / (rw * rh) if rw * rh > 0 else 0
        if 0.8 < ratio < 1.25 and fill > 0.5 and area > best_area:
            best_area = area
            best = (x1 + rx, y1 + ry, rw, rh)

    return best
