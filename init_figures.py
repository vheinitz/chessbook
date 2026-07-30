import cv2
import numpy as np


# Define piece mappings for FEN notation
piece_map = {
    'p': "black_pawn", 'r': "black_rook", 'n': "black_knight",
    'b': "black_bishop", 'q': "black_queen", 'k': "black_king",
    'P': "white_pawn", 'R': "white_rook", 'N': "white_knight",
    'B': "white_bishop", 'Q': "white_queen", 'K': "white_king",
    ' ': "empty"
}

# Reverse mapping: piece name -> FEN char
fen_map = {v: k for k, v in piece_map.items()}


def match_histogram(source, template):
    """Match the histogram of source image to template image."""
    src = source.copy()
    for i in range(3):
        src_hist, _ = np.histogram(src[:, :, i].ravel(), 256, [0, 256])
        tmpl_hist, _ = np.histogram(template[:, :, i].ravel(), 256, [0, 256])
        src_cdf = src_hist.cumsum().astype(float) / src_hist.sum()
        tmpl_cdf = tmpl_hist.cumsum().astype(float) / tmpl_hist.sum()
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for k in range(256):
            while j < 256 and tmpl_cdf[j] < src_cdf[k]:
                j += 1
            lut[k] = min(j, 255)
        src[:, :, i] = cv2.LUT(src[:, :, i], lut)
    return src


def extract_figure(square_image):
    """
    Extract the piece figure from a square image by isolating foreground
    from the background square.

    Uses adaptive thresholding to separate piece from background.
    """
    # Apply Gaussian blur to reduce noise
    blurred_square = cv2.GaussianBlur(square_image, (5, 5), 0)

    # Apply adaptive thresholding to get a binary image (figure contours)
    binary_square = cv2.adaptiveThreshold(
        blurred_square, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 6
    )

    # Find contours of the figure
    contours, _ = cv2.findContours(binary_square, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask of the same size as the square image
    mask = np.zeros_like(binary_square)

    # Fill the detected contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Erode slightly to remove border artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)

    # Subtract the background using the filled mask
    figure_only = cv2.bitwise_and(square_image, mask)
    return figure_only


def extract_piece_images(image_path, fen):
    """
    Extracts images of chess pieces based on the given FEN string.

    Parameters:
    - image_path: Path to the chessboard image (reference board).
    - fen: FEN string representing the board position.

    Returns:
    - board_size: Tuple (width, height) of the board image.
    - piece_images: Dict mapping piece type names to lists of image patches.
    """
    # Load the chessboard image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the size of each square (assuming an 8x8 board)
    height, width = gray_image.shape[:2]
    square_size = width // 8

    # Initialize a dictionary to store images of each piece type
    piece_images = {ptype: [] for ptype in piece_map.values()}

    # Split the FEN string into ranks
    ranks = fen.split(' ')[0].split('/')

    # Iterate over the ranks (from top to bottom)
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            # Calculate the coordinates of the square
            x_start = file_idx * square_size + 3
            y_start = rank_idx * square_size + 3
            x_end = x_start + square_size - 6
            y_end = y_start + square_size - 6

            # Extract the square image
            square_image = gray_image[y_start:y_end, x_start:x_end]

            figure_only = extract_figure(square_image)

            if char in piece_map:
                piece_type = piece_map[char]
                piece_images[piece_type].append(figure_only)
                file_idx += 1
            elif char.isdigit():
                piece_images['empty'].append(figure_only)
                file_idx += int(char)

    return (width, height), piece_images


def calculate_hu_moments(image):
    """
    Calculates the Hu Moments for the given image.
    Returns None if image has no non-zero pixels.
    """
    non_zero = np.count_nonzero(image)
    if non_zero < 5:
        return None

    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Use log-scaled absolute values for stability
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)


def match_figure_by_moments(square_image, templates):
    """
    Matches a square image to the closest figure template using Hu Moments.

    Parameters:
    - square_image: Grayscale image of the square with figure extracted.
    - templates: Dict mapping piece type names to lists of template images.

    Returns:
    - Matched piece type name (e.g., "white_pawn"), or "empty" for no match.
    """
    # Calculate Hu Moments for the square image
    square_moments = calculate_hu_moments(square_image)
    if square_moments is None:
        return ""

    best_match = ""
    min_distance = float("inf")

    # Iterate over each piece type and its template images
    for piece_type, image_list in templates.items():
        if piece_type == "empty":
            continue  # Handle empty separately
        for template_image in image_list:
            template_moments = calculate_hu_moments(template_image)
            if template_moments is None:
                continue

            # Compute Euclidean distance between moment vectors
            distance = np.sqrt(np.sum((square_moments - template_moments) ** 2))

            if distance < min_distance:
                min_distance = distance
                best_match = piece_type

    # If the best match distance is too high, or the square is mostly empty,
    # classify as empty
    non_zero = np.count_nonzero(square_image)
    if non_zero < 10:
        return ""

    # Reject if the best match is too far (very different from any template)
    # This threshold may need tuning
    if min_distance > 50:
        return ""

    return best_match


def extract_fen_from_image(image_path, templates, player="W", reference_path=None):
    """
    Extracts the FEN string from a board image using figure templates.

    Parameters:
    - image_path: Path to the board image to analyze.
    - templates: Dict of figure templates from extract_piece_images().
    - player: "W" (white at bottom) or "B" (black at bottom).
    - reference_path: Optional path to reference board image for histogram matching.

    Returns:
    - FEN string representing the board position.
    """
    # Load the board image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Histogram-match to the reference board if provided
    if reference_path is not None:
        reference = cv2.imread(reference_path)
        if reference is not None:
            image = match_histogram(image, reference)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the size of each square (assuming an 8x8 board)
    height, width = gray_image.shape[:2]
    square_size = width // 8

    # Initialize the FEN string components
    fen_rows = []

    # Iterate over each square on the board
    for rank in range(8):
        fen_row = ""
        empty_count = 0

        for file in range(8):
            # Adjust the coordinates for the player perspective
            actual_rank = rank if player == "W" else 7 - rank
            actual_file = file if player == "W" else 7 - file

            # Calculate the coordinates of the current square
            margin = 3
            x_start = actual_file * square_size + margin
            y_start = actual_rank * square_size + margin
            x_end = x_start + square_size - 2 * margin
            y_end = y_start + square_size - 2 * margin

            # Extract the square image
            square_image = gray_image[y_start:y_end, x_start:x_end]

            # Extract the figure from the square
            figure_only = extract_figure(square_image)

            # Check if the square is empty (very few non-zero pixels)
            if np.count_nonzero(figure_only) < 15:
                empty_count += 1
            else:
                matched_piece = match_figure_by_moments(figure_only, templates)
                if matched_piece and matched_piece in fen_map:
                    fen_char = fen_map[matched_piece]
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += fen_char
                else:
                    empty_count += 1

        # Append remaining empty squares if any
        if empty_count > 0:
            fen_row += str(empty_count)

        # Add the row to the FEN string
        fen_rows.append(fen_row)

    # Join the rows with slashes to form the FEN string
    fen = "/".join(fen_rows)
    return fen


if __name__ == "__main__":
    # Example usage
    image_path = "startboard.png"
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    # Extract piece images based on the FEN string
    board_size, piece_images = extract_piece_images(image_path, fen)

    # Print the number of images extracted for each piece type
    for piece_type, images in piece_images.items():
        print(f"{piece_type}: {len(images)} images extracted")

    # Example usage
    new_board_image_path = "extracted_chessboard.png"
    fen = extract_fen_from_image(new_board_image_path, piece_images,
                                  reference_path=image_path)
    print("Extracted FEN:", fen)
