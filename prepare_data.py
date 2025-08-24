import os
import cv2
import glob
import numpy as np

# --- CONFIGURATION ---
# Define the path to your downloaded Avenue dataset
AVENUE_PATH = 'Avenue Dataset'

# Define the output folder and image properties
OUTPUT_FOLDER = 'unified_normal_frames'
IMG_SIZE = (256, 256)

def process_avenue():
    """Finds, extracts, preprocesses, and saves the normal training frames from the Avenue dataset."""
    print("Processing Avenue dataset...")
    video_paths = glob.glob(os.path.join(AVENUE_PATH, 'training_videos', '*.avi'))
    
    if not video_paths:
        print("Warning: Avenue training videos not found. Please check the path in AVENUE_PATH.")
        return 0

    frame_count = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).replace('.avi', '')
        
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess: Grayscale and Resize
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, IMG_SIZE)
            
            # Save the frame with a unique name
            filename = f"avenue_{video_name}_{frame_num:04d}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(output_path, frame_resized)
            frame_num += 1
            frame_count += 1
        cap.release()
        
    print(f"Finished processing Avenue. {frame_count} frames prepared.")
    return frame_count

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("Starting data preparation process for the Avenue dataset...")
    total_frames = process_avenue()
    
    print(f"\nData preparation complete! A total of {total_frames} frames have been saved to the '{OUTPUT_FOLDER}' directory.")