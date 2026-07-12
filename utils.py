import cv2

# Define the explicit connections between MediaPipe hand landmarks to form bones
HAND_CONNECTIONS = [
    # Wrist to base of fingers
    (0, 1), (0, 5), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index Finger
    (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (9, 10), (10, 11), (11, 12),
    # Ring Finger
    (13, 14), (14, 15), (15, 16),
    # Pinky Finger
    (17, 18), (18, 19), (19, 20),
    # Knuckle joints connections across the palm
    (5, 9), (9, 13), (13, 17)
]

def draw_styled_skeleton(image, hand_landmarks):
    """
    Draws a fully connected bone skeleton wireframe over the hand using native OpenCV,
    bypassing the broken mp.solutions.drawing_utils.
    """
    h, w, _ = image.shape
    
    # 1. Convert normalized landmarks into pixel coordinate tuples
    pixel_points = {}
    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        pixel_points[idx] = (cx, cy)
        
    # 2. Draw the connection lines (Bones) first so they sit behind the dots
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx in pixel_points and end_idx in pixel_points:
            pt1 = pixel_points[start_idx]
            pt2 = pixel_points[end_idx]
            # Draw vibrant blue/pink bone lines
            cv2.line(image, pt1, pt2, (255, 75, 75), 3, cv2.LINE_AA)
            
    # 3. Draw the joint markers (Dots) over the lines
    for idx, pt in pixel_points.items():
        # Draw clean, bright green tracking nodes
        cv2.circle(image, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)