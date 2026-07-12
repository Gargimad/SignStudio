import numpy as np

# Define the connected bone structure pairs of a hand to calculate angles
# Each tuple represents (Joint_Base, Joint_Middle, Joint_Tip)
BONE_ANGLES = [
    # Thumb
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index Finger
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle Finger
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring Finger
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky Finger
    (0, 17, 18), (17, 18, 19), (18, 19, 20)
]

def calculate_joint_angles(landmarks):
    """Transforms 21 3D landmarks into a scale-invariant list of bone angles."""
    matrix = np.array(landmarks)
    angles = []
    
    for base, mid, tip in BONE_ANGLES:
        # Build vectors between joints
        v1 = matrix[base] - matrix[mid]
        v2 = matrix[tip] - matrix[mid]
        
        # Normalize vectors
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        
        # Calculate angle in degrees
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        angles.append(np.degrees(angle))
        
    return np.array(angles)

def get_top_predictions(live_landmarks, reference_library, handedness):
    """Compares joint angles using Euclidean distance and returns ranked results."""
    if not reference_library:
        return []
        
    live_angles = calculate_joint_angles(live_landmarks)
    predictions = []
    
    for word, ref_data in reference_library.items():
        # Check if reference data contains handedness info to avoid comparing Left to Right
        ref_handedness = ref_data.get("handedness", "Right")
        if ref_handedness != handedness:
            continue
            
        ref_angles_list = np.array(ref_data["angles"])
        
        # Find closest matching frame inside the blueprint library
        distances = [np.linalg.norm(live_angles - ref_angles) for ref_angles in ref_angles_list]
        min_distance = min(distances) if len(distances) > 0 else float('inf')
        
        # Convert Euclidean distance to a cleaner calibrated confidence percentage
        confidence = max(0.0, 100.0 - (min_distance * 1.5))
        predictions.append({"word": word, "confidence": confidence})
        
    # Sort predictions by highest confidence score
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions[:3]