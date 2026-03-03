#!/usr/bin/python
import cv2
import numpy as np
import os
import logging
import glob
from pystackreg import StackReg
from tqdm import tqdm
import sys
import csv
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def validate_coordinates(coords):
    if coords[1] <= coords[0] or coords[3] <= coords[2]:
        raise argparse.ArgumentTypeError("Invalid cropping coordinates. Ensure that left_x < right_x and top_y < bottom_y.")
    return coords

def adjust_cropping_coords(cropping_coords, video_width, video_height):
    """
    Adjusts cropping coordinates to ensure they are within the video dimensions.
    Args:
        cropping_coords (list): Cropping coordinates [left_x, right_x, top_y, bottom_y].
        video_width (int): Width of the video.
        video_height (int): Height of the video.
    Returns:
        list: Adjusted cropping coordinates.
    """
    left_x, right_x, top_y, bottom_y = cropping_coords
    if left_x < 0 or right_x > video_width or top_y < 0 or bottom_y > video_height:
        logging.warning("Cropping coordinates are out of video dimensions. Adjusting them accordingly.")
        left_x = max(left_x, 0)
        right_x = min(right_x, video_width)
        top_y = max(top_y, 0)
        bottom_y = min(bottom_y, video_height)
    return [left_x, right_x, top_y, bottom_y]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply motion correction to video.")
    parser.add_argument('video_path', help="Path to the input video file.")
    parser.add_argument('--movement_output_path', help="Path to save the movement data in CSV format.", default=None)
    parser.add_argument('--frame_start', help="The first frame for reference image creation", default=None)
    parser.add_argument('--frame_end', help="The last frame for reference image creation", default=None)
    parser.add_argument('--algorithm_type', help="One of four possible algorithms as implemented by pystackreg. Options are Translation (T), Rigid Body (R), Scaled Rotation (S), and Affine (A)", default='S')
    parser.add_argument('--cropping_coords', nargs=4, type=int, help="Cropping coordinates in the format left_x right_x top_y bottom_y.", default=None)

    args = parser.parse_args()

    # If provided, test if they have the correct shape
    if args.cropping_coords:
        args.cropping_coords = validate_coordinates(args.cropping_coords)


    
    # If provided, convert cropping_coords to a list of integers
    if args.cropping_coords:
        args.cropping_coords = list(map(int, args.cropping_coords))

        # Map the single-character algorithm input to the appropriate StackReg constant
    algorithm_map = {
        'T': StackReg.TRANSLATION,
        'R': StackReg.RIGID_BODY,
        'S': StackReg.SCALED_ROTATION,
        'A': StackReg.AFFINE
    }
    
    if args.algorithm_type.upper() in algorithm_map:
        args.algorithm_type = algorithm_map[args.algorithm_type.upper()]
    else:
        raise ValueError(f"Invalid algorithm type: {args.algorithm_type}. Choose one of 'T', 'R', 'S', 'A'.")

    
    return args


def calculate_mean_image(video_path, frame_start, frame_end, cropping_coords=None):
    logging.info("Calculating mean image...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return None
    
    # Check for valid frame range to avoid division by zero
    if frame_end <= frame_start or frame_start < 0 or frame_end > int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        logging.error("Invalid frame range. Ensure that frame_end is greater than frame_start and within the video length.")
        cap.release()
        return None

    sum_image = None

    
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if cropping_coords:
        cropping_coords = adjust_cropping_coords(cropping_coords, video_width, video_height)
        logging.info(f"Using following adjusted cropping coordinates: {cropping_coords}")


    # Set the starting frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    frame_count = frame_end - frame_start
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames to read

        if cropping_coords:
            left_x, right_x, top_y, bottom_y = cropping_coords
            frame = frame[top_y:bottom_y, left_x:right_x]  # Apply cropping

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if sum_image is None:
            sum_image = np.zeros_like(gray_frame, dtype=np.float32)

        sum_image += gray_frame

    mean_image = sum_image / frame_count
    cap.release()
    logging.info("Mean image calculation completed.")
    return mean_image.astype(np.uint8)



def apply_motion_correction(input_video_path, output_video_path, mean_image, cropping_coords, algorithm_type=StackReg.SCALED_ROTATION, movement_output_path=None):
    """
    Applies motion correction to a video and optionally saves information about detected movements.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path where the corrected video will be saved.
        mean_image (numpy.ndarray): The mean image used for registration.
        cropping_coords (list): Cropping coordinates in the format [left_x, right_x, top_y, bottom_y].
        algorithm_type (StackReg constant): Type of transformation. Options are StackReg.TRANSLATION, StackReg.RIGID_BODY, 
                                            StackReg.SCALED_ROTATION, StackReg.AFFINE. Default is StackReg.SCALED_ROTATION.
        movement_output_path (str, optional): Path to save the movement data in CSV format.

    Returns:
        None: The function saves the corrected video and optionally the movement data.
    """
    try:
        logging.info("Starting motion correction...")

        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")

        
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if cropping_coords:
            cropping_coords = adjust_cropping_coords(cropping_coords, video_width, video_height)
            logging.info(f"Using following adjusted cropping coordinates: {cropping_coords}")

        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        movements = []

        # Setup Video Writer
        if cropping_coords:
            crop_width = cropping_coords[1] - cropping_coords[0]
            crop_height = cropping_coords[3] - cropping_coords[2]
        else:
            crop_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            crop_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (crop_width, crop_height))

        if not out.isOpened():
            raise IOError(f"Cannot write video file: {output_video_path}")


        # Initialize StackReg object
        sr = StackReg(algorithm_type)

        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break

            if cropping_coords:
                frame = frame[cropping_coords[2]:cropping_coords[3], cropping_coords[0]:cropping_coords[1]]

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Register the current frame to the mean image and apply transformation
            transformation_matrix = sr.register(mean_image, gray_frame)

            # Extract x and y movements
            dx = transformation_matrix[0, 2]
            dy = transformation_matrix[1, 2]
            movements.append((dx, dy))

            umat_frame = cv2.UMat(frame)
            
            corrected_frame = cv2.warpAffine(umat_frame, transformation_matrix[:2], (crop_width, crop_height),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)

            out.write(corrected_frame.get())

        logging.info(f"Motion corrected video saved at {output_video_path}")

        # Optionally save the movement data
        if movement_output_path:
            with open(movement_output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Frame', 'Delta_X', 'Delta_Y'])
                for i, (dx, dy) in enumerate(movements):
                    writer.writerow([i, dx, dy])
                logging.info(f"Movement data saved to {movement_output_path}")

        cap.release()
        out.release()
    
    except Exception as e:
        logging.error(f"An error occurred in motion correction: {e}")



        
def correct_motion(video_path, cropping_coords=None, frame_start_end=None, algorithm_type=StackReg.SCALED_ROTATION, movement_output_path=None):
    """
    Processes the given video for motion correction.

    Args:
        video_path (str): Path to the video file.
        cropping_coords (list, optional): Coordinates for cropping and motion correction.
        frame_start_end (tuple, optional): Tuple of (frame_start, frame_end) for mean image calculation.
        algorithm_type (StackReg constant, optional): Algorithm type for motion correction.
        movement_output_path (str, optional): Path to save the movement data.

    Returns:
        str: Path to the motion-corrected video.
    """

    # Validate cropping coordinates
    if cropping_coords is not None:
        if not (isinstance(cropping_coords, list) and len(cropping_coords) == 4):
            raise ValueError("Cropping coordinates must be a list of four values.")

    # Validate frame start and end
    if frame_start_end is not None:
        if not (isinstance(frame_start_end, tuple) and len(frame_start_end) == 2):
            raise ValueError("frame_start_end must be a tuple of two values (frame_start, frame_end).")

    # Validate algorithm type
    if algorithm_type not in [StackReg.TRANSLATION, StackReg.RIGID_BODY, StackReg.SCALED_ROTATION, StackReg.AFFINE]:
        raise ValueError("Invalid algorithm type for motion correction.")

    # Validate movement output path
    if movement_output_path is not None and not isinstance(movement_output_path, str):
        raise ValueError("Invalid movement output path.")

    # Validate existence of the directory for movement_output_path
    if movement_output_path:
        movement_output_dir = os.path.dirname(movement_output_path)
        if not os.path.exists(movement_output_dir):
            raise ValueError(f"The directory for movement output path does not exist: {movement_output_dir}")


    logging.info(f"Starting motion correction process for {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Default frame start and end if not provided
    mid_frame = frame_count // 2
    if frame_start_end is None:
        frame_start, frame_end = max(0, mid_frame - 1000), min(frame_count, mid_frame + 1000)
    else:
        frame_start, frame_end = frame_start_end
        logging.info(f"Using supplied range for mean image calculation {frame_start_end}")

    mean_image = calculate_mean_image(video_path, frame_start, frame_end, cropping_coords)
    bare_video_path = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(os.path.dirname(video_path), f"{bare_video_path}-moco.mp4")

    apply_motion_correction(video_path, output_video_path, mean_image, cropping_coords, algorithm_type, movement_output_path)

    return output_video_path



def main():
    args = parse_arguments()

    # Ensure that a video path is provided
    if not args.video_path:
        logging.error("No video path provided. Please specify the path to the input video file.")
        sys.exit(1)  # Exit the script with an error status


    frame_start_end = (args.frame_start,args.frame_end) if args.frame_start is not None and args.frame_end is not None else None
    
    corrected_video_path = correct_motion(args.video_path, args.cropping_coords, frame_start_end, args.algorithm_type, args.movement_output_path)
    logging.info(f"Motion correction completed. Corrected video saved at {corrected_video_path}")

if __name__ == '__main__':
    main()
