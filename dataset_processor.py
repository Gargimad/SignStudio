import json
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
INDEX_FILE = "WLASL_v1.json"        # Path to your index file
OUTPUT_FILE = "wlasl_dictionary.json" # Output destination for coordinates
MODEL_PATH = "hand_landmarker.task"
MAX_WORDS = 10                       # Set how many total words you want to extract
MAX_INSTANCES_PER_WORD = 1           # Number of clean variations needed per word

def setup_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Please place {MODEL_PATH} in this directory.")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )
    return vision.HandLandmarker.create_from_options(options)

def extract_coordinates_from_video(video_path, detector):
    """Opens a video file and processes it safely through MediaPipe."""
    cap = cv2.VideoCapture(video_path)
    video_sequence = []
    
    try:
        frame_timestamp_ms = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_duration_ms = int(1000 / fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            frame_timestamp_ms += frame_duration_ms
            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
            
            frame_coords = []
            if detection_result.hand_landmarks:
                # Extract first detected hand sequence
                first_hand = detection_result.hand_landmarks[0]
                for landmark in first_hand:
                    frame_coords.append([landmark.x, landmark.y, landmark.z])
            
            if frame_coords:
                video_sequence.append(frame_coords)
    finally:
        # Crucial Fix: Forces Windows to unlock the video file handle no matter what
        cap.release()
        
    return video_sequence

def main():
    print("🚀 Initializing MediaPipe Hand Detector...")
    try:
        detector = setup_detector()
    except Exception as e:
        print(f"Error: {e}")
        return

    if not os.path.exists(INDEX_FILE):
        print(f"Error: Could not find your index file named '{INDEX_FILE}'")
        return

    with open(INDEX_FILE, "r") as f:
        wlasl_data = json.load(f)

    # Load existing dictionary data if it exists to allow appending safely
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                processed_dictionary = json.load(f)
            except json.JSONDecodeError:
                processed_dictionary = {}
    else:
        processed_dictionary = {}

    temp_video = "temp_download.mp4"
    words_processed = 0

    for entry in wlasl_data:
        if words_processed >= MAX_WORDS:
            break
            
        gloss_word = entry["gloss"].upper()
        
        # Skip if we already successfully processed this word in a prior run
        if gloss_word in processed_dictionary:
            continue
            
        instances = entry["instances"]
        
        print(f"\nProcessing word [{len(processed_dictionary) + 1}]: '{gloss_word}'")
        word_sequences = []
        instances_count = 0

        for inst in instances:
            if instances_count >= MAX_INSTANCES_PER_WORD:
                break
                
            video_url = inst["url"]
            
            # Skip YouTube links and Adobe Flash .swf files which urllib can't parse natively
            if "youtube.com" in video_url or "youtu.be" in video_url or video_url.endswith(".swf"):
                continue
                
            print(f"  -> Downloading instance from: {video_url}")
            
            try:
                urllib.request.urlretrieve(video_url, temp_video)
                coords = extract_coordinates_from_video(temp_video, detector)
                
                if coords:
                    word_sequences.append(coords)
                    instances_count += 1
                    print(f"     ✅ Successfully extracted {len(coords)} coordinate tracking frames.")
                else:
                    print("     ⚠️ No hands detected in this video structure.")
                    
            except Exception as e:
                print(f"     ❌ Failed to download or process this video resource.")
            
            finally:
                # Safely delete the temporary video file now that cap.release() is guaranteed
                if os.path.exists(temp_video):
                    try:
                        os.remove(temp_video)
                    except OSError:
                        pass 

        if word_sequences:
            processed_dictionary[gloss_word] = word_sequences[0]
            words_processed += 1
            
            # Intermediate checkpoint write-out so you don't lose progress if you stop execution
            with open(OUTPUT_FILE, "w") as f:
                json.dump(processed_dictionary, f, indent=4)

    print(f"\n🎉 Process Complete! Clean coordinate file saved to '{OUTPUT_FILE}' with {len(processed_dictionary)} total entries.")

if __name__ == "__main__":
    main()