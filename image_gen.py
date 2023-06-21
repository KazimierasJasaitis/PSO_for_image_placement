import os
import random
from PIL import Image
import math


def cut_image(image_path, output_folder, n_pieces):
    if n_pieces < 2 or n_pieces > 8:
        raise ValueError("n_pieces should be between 2 and 8.")

    # Load the image
    image = Image.open(image_path)
    width, height = image.size
    original_area = width * height

    pieces = []
    total_area = 0

    while total_area != original_area:
        # Store the cutting coordinates
        horizontal_cuts = [0, height]
        vertical_cuts = [0, width]

        # Calculate the number of horizontal and vertical cuts needed
        n_horizontal_cuts = int(math.sqrt(n_pieces))
        n_vertical_cuts = (n_pieces + n_horizontal_cuts - 1) // n_horizontal_cuts

        # Select random points for horizontal cuts
        horizontal_cuts += random.sample(range(1, height - 1), n_horizontal_cuts - 1)
        # Select random points for vertical cuts
        vertical_cuts += random.sample(range(1, width - 1), n_vertical_cuts - 1)

        # Sort the lists
        horizontal_cuts.sort()
        vertical_cuts.sort()

        # Cut the image into pieces
        pieces = []
        for i in range(1, len(horizontal_cuts)):
            for j in range(1, len(vertical_cuts)):
                left = vertical_cuts[j - 1]
                upper = horizontal_cuts[i - 1]
                right = vertical_cuts[j]
                lower = horizontal_cuts[i]
                cropped_image = image.crop((left, upper, right, lower))
                pieces.append(cropped_image)
        
        # Combine pieces if needed
        while len(pieces) > n_pieces:
            idx_to_combine = random.randint(0, len(pieces) - 2)
            combined_piece = Image.new(
                'RGB',
                (pieces[idx_to_combine].width + pieces[idx_to_combine + 1].width, pieces[idx_to_combine].height)
            )
            combined_piece.paste(pieces[idx_to_combine], (0, 0))
            combined_piece.paste(pieces[idx_to_combine + 1], (pieces[idx_to_combine].width, 0))
            del pieces[idx_to_combine + 1]
            pieces[idx_to_combine] = combined_piece
        
        # Calculate total area of the pieces
        total_area = sum(piece.width * piece.height for piece in pieces)
    
    # Save the final pieces
    piece_counter = 1
    for piece in pieces:
        piece.save(os.path.join(output_folder, f'0{piece_counter}.png'))
        piece_counter += 1

# Example usage when imported:
if __name__ == "__main__":

    n_pieces = 7  # Can be any integer between 2 and 8
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "0" + str(n_pieces) + "/")
    image_path = output_folder + "image.png"


    cut_image(image_path, output_folder, n_pieces)
