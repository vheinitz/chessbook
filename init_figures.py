import cv2
import numpy as np


# Define piece mappings for FEN notation
piece_map = {
    'p': "black_pawn", 'r': "black_rook", 'n': "black_knight", 'b': "black_bishop", 'q': "black_queen", 'k': "black_king",
    'P': "white_pawn", 'R': "white_rook", 'N': "white_knight", 'B': "white_bishop", 'Q': "white_queen", 'K': "white_king",
    ' ': "empty"
}

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import cv2


def extract_figure(square_image):
            # Apply Gaussian blur to reduce noise
            blurred_square = cv2.GaussianBlur(square_image, (5, 5), 0)

            # Apply adaptive thresholding to get a binary image (figure contours)
            binary_square = cv2.adaptiveThreshold(
                blurred_square, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 7, 2
            )

            # Perform morphological operations to clean up the binary image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #binary_square = cv2.morphologyEx(binary_square, cv2.MORPH_CLOSE, kernel)

            # Find contours of the figure
            contours, _ = cv2.findContours(binary_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask of the same size as the square image
            mask = np.zeros_like(binary_square)

            # Fill the detected contours on the mask
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=2)


            # Subtract the background using the filled mask
            figure_only = cv2.bitwise_and(square_image, mask)
            return figure_only

def show_images_interactive(image1, image2, title1="Image 1", title2="Image 2"):
    """
    Display two images side by side interactively using matplotlib.
    Waits for a key press or mouse click to update the images.

    Parameters:
    - image1 (ndarray): First image to display.
    - image2 (ndarray): Second image to display.
    - title1 (str): Title for the first image.
    - title2 (str): Title for the second image.
    """
    # Convert images to RGB format if they are in BGR format
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Enable interactive mode
    plt.ion()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    img_plot1 = ax1.imshow(image1, cmap="gray" if len(image1.shape) == 2 else None)
    ax1.set_title(title1)
    ax1.axis("off")

    # Display the second image
    img_plot2 = ax2.imshow(image2, cmap="gray" if len(image2.shape) == 2 else None)
    ax2.set_title(title2)
    ax2.axis("off")

    # Draw the initial images
    plt.show()

    while True:
        # Wait for a key press or mouse click
        print("Press any key or click the figure to update the images.")
        plt.waitforbuttonpress()

        # Update the images (for demonstration, we'll just refresh the same images)
        img_plot1.set_data(image1)
        img_plot2.set_data(image2)

        # Redraw the figure
        fig.canvas.draw()

        # Optionally, break the loop with a specific key press (e.g., 'q' to quit)
        key = input("Press 'q' to quit or Enter to refresh: ")
        if key.lower() == 'q':
            break

    # Disable interactive mode
    plt.ioff()
    plt.close(fig)


def extract_piece_images(image_path, fen):
    """
    Extracts images of chess pieces based on the given FEN string and creates a map of piece type to list of images.

    Parameters:
    - image_path (str): Path to the chessboard image.
    - fen (str): FEN string representing the board position.

    Returns:
    - piece_images (dict): A dictionary where each key is a piece type (e.g., "black_pawn") and the value is a list of image patches.
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
            x_start = file_idx * square_size + 2
            y_start = rank_idx * square_size + 2
            x_end = x_start + square_size - 2
            y_end = y_start + square_size - 2

            # Extract the square image
            square_image = gray_image[y_start:y_end, x_start:x_end]

            figure_only = extract_figure( square_image )


            if char in piece_map:
                # Get the piece type
                piece_type = piece_map[char]
                # Add the square image to the list for this piece type
                piece_images[piece_type].append(figure_only)
                # Move to the next file (column)
                file_idx += 1

            elif char.isdigit():
                piece_images['empty'].append(figure_only)
                file_idx += int(char)

    return (width, height), piece_images



import cv2
import numpy as np

import cv2
import numpy as np

def save_images_side_by_side(image1, image2, output_path="combined_image.png", info=""):
    """
    Combines two images side by side, annotates them with information, and saves the result.

    Parameters:
    - image1 (ndarray): First image to be placed on the left.
    - image2 (ndarray): Second image to be placed on the right.
    - output_path (str): File path for saving the combined image.
    - info (str): Information text to overlay on the combined image.
    """
    # Ensure both images have the same height
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Resize the images if their heights are different
    if height1 != height2:
        new_height = min(height1, height2)
        image1 = cv2.resize(image1, (int(width1 * new_height / height1), new_height))
        image2 = cv2.resize(image2, (int(width2 * new_height / height2), new_height))

    # Concatenate the images horizontally
    combined_image = np.hstack((image1, image2))

    # Add text overlay for info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .5
    thickness = 2
    color = (255, 255, 255)  # White text
    text_size = cv2.getTextSize(info, font, font_scale, thickness)[0]

    # Calculate text position
    text_x = (combined_image.shape[1] - text_size[0]) // 2
    text_y = 10  # Text height from the top of the image

    # Add a background rectangle for text visibility
    cv2.rectangle(combined_image, 
                  (text_x - 10, text_y - text_size[1] - 10), 
                  (text_x + text_size[0] + 10, text_y + 10), 
                  (0, 0, 0), 
                  -1)  # Black rectangle

    # Overlay the text
    cv2.putText(combined_image, info, (text_x, text_y), font, font_scale, color, thickness)

    # Save the combined image
    cv2.imwrite(output_path, combined_image)
    print(f"Combined image saved as {output_path}")


# Example usage:
# save_images_side_by_side(template_image, square_image, "combined_output.png")


# Initialize ORB detector
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

filenum=1;

def calculate_hu_moments(image):
    """
    Calculates the Hu Moments for the given binary image.

    Parameters:
    - image (ndarray): Input binary image.

    Returns:
    - hu_moments (ndarray): Array of Hu Moments.
    """
    # Compute image moments
    moments = cv2.moments(image)
    
    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()
    
    return hu_moments



def match_figure_by_moments(square_image, templates):
    global filenum
    """
    Matches the given square image to the closest figure template using Hu Moments.

    Parameters:
    - square_image (ndarray): Extracted binary image of the square.
    - templates (dict): Dictionary where each key is a FEN character (e.g., 'p', 'R')
      and the value is a list of template images for that piece type.

    Returns:
    - matched_fen_char (str): The FEN character of the matched piece (e.g., 'p' for black pawn).
      Returns an empty string if no match is found.
    """
    # Calculate Hu Moments for the square image
    square_moments = calculate_hu_moments(square_image)

    best_match = ""
    min_distance = float("inf")

    #optmap={}
    # Iterate over each FEN character and its list of template images
    for fen_char, image_list in templates.items():
        for template_image in image_list:
            # Calculate Hu Moments for the template image
            template_moments = calculate_hu_moments(template_image)
            

            # Calculate the distance between Hu Moments (log scale for stability)
            distance = np.sum(np.abs(np.log(np.abs(square_moments)) - np.log(np.abs(template_moments))))
            try:
                #save_images_side_by_side( template_image, square_image, "/tmp/img/compare_%d.png" % (filenum), str(int(distance*10)) )
                pass
            except:
                pass
            filenum += 1

            # Update the best match if the distance is smaller
            if distance < min_distance:
                min_distance = distance
                best_match = fen_char

    # Return the matched FEN character or an empty string if no match is found
    return best_match


def match_figure(square_image, templates):
    global filenum
    """
    Matches the given square image to the closest figure template.

    Parameters:
    - square_image (ndarray): Extracted grayscale image of the square.
    - templates (dict): Dictionary where each key is a FEN character (e.g., 'p', 'R')
      and the value is a list of template images for that piece type.

    Returns:
    - matched_fen_char (str): The FEN character of the matched piece (e.g., 'p' for black pawn).
      Returns an empty string if no match is found.
    """
    keypoints, descriptors = orb.detectAndCompute(square_image, None)
    if descriptors is None:
        return ""  # No descriptors found, likely an empty square

    best_match = ""
    max_matches = 0

    
    
    # Iterate over each FEN character and its list of template images
    for fen_char, image_list in templates.items():
        for template_image in image_list:
            # Detect keypoints and descriptors for the template image
            template_keypoints, template_descriptors = orb.detectAndCompute(template_image, None)
            if template_descriptors is None:
                continue
            
            save_images_side_by_side( template_image, square_image, "/tmp/img/compare_%d.png" % (filenum) )
            filenum += 1

            # Match descriptors using BFMatcher
            matches = bf.match(descriptors, template_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            print( fen_char , matches , max_matches )
            # Use the number of good matches as the score
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_match = fen_char

    return best_match



def extract_fen_from_image(image_path, templates, player="W"):
    """
    Extracts the FEN string from the given board image using figure templates.

    Parameters:
    - image_path (str): Path to the board image.
    - templates (dict): Dictionary of figure templates and descriptors.
    - player (str): "W" if the player is white (bottom), "B" if the player is black (bottom).

    Returns:
    - fen (str): The FEN string representing the board position.
    """
    # Load the board image and preprocess it
    image = cv2.imread(image_path)

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
            x_start = actual_file * square_size + 2
            y_start = actual_rank * square_size + 2
            x_end = x_start + square_size - 2
            y_end = y_start + square_size - 2

            # Extract the square image
            square_image = gray_image[y_start:y_end, x_start:x_end]

            figure_only = extract_figure(square_image)

            

            matched_piece = match_figure_by_moments(figure_only, templates)
            #matched_piece = match_figure(figure_only, templates)

            if matched_piece:
                # Map the matched piece to the FEN character
                fen_char = [k for k, v in piece_map.items() if v == matched_piece][0]
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += fen_char
            else:
                # Count empty squares
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
    fen = extract_fen_from_image(new_board_image_path, piece_images)
    print("Extracted FEN:", fen)