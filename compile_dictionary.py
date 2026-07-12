import os
import pickle
import json

# --- CONFIGURATION ---
DATASET_PATH = r"C:\Users\rabim\.cache\kagglehub\datasets\nguyenchitinh\asl-citizen\versions\21\keypoints-100"
OUTPUT_FILE = "wlasl_dictionary.json"

# Let's try matching words we KNOW exist from your screenshots (like BASEMENT and APPLE)
TARGET_WORDS = ["BASEMENT", "APPLE", "BOOK", "HELLO", "ACTION"]

def compile_asl_dict():
    app_dictionary = {}
    all_available_folders = []
    
    print("🔍 Scanning keypoints-100 directory...")

    # Walk through the file tree to find what folders exist
    for root, dirs, files in os.walk(DATASET_PATH):
        for d in dirs:
            folder_name_upper = d.upper()
            if folder_name_upper not in all_available_folders:
                all_available_folders.append(folder_name_upper)
            
            # Check for a match
            matched_word = None
            for word in TARGET_WORDS:
                if folder_name_upper == word or folder_name_upper.startswith(word):
                    matched_word = word
                    break
            
            if matched_word and matched_word not in app_dictionary:
                target_folder_path = os.path.join(root, d)
                
                # Grab all .pkl files right inside this folder
                pkl_files = [f for f in os.listdir(target_folder_path) if f.endswith(".pkl")]
                
                if pkl_files:
                    print(f"  🎯 Found match! Extracting {len(pkl_files)} files from folder: '{d}'")
                    app_dictionary[matched_word] = []
                    
                    for pkl_file in pkl_files:
                        target_file = os.path.join(target_folder_path, pkl_file)
                        try:
                            with open(target_file, 'rb') as f:
                                data = pickle.load(f)
                            
                            if hasattr(data, 'tolist'):
                                clean_sequence = data.tolist()
                            else:
                                clean_sequence = list(data)
                                
                            app_dictionary[matched_word].append(clean_sequence)
                        except Exception as e:
                            pass

    # Save whatever we found
    if app_dictionary:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(app_dictionary, f, indent=4)
        print(f"\n🎉 Success! Built `{OUTPUT_FILE}` with {len(app_dictionary)} words.")
    else:
        print("\n❌ Could not find any of your TARGET_WORDS in this dataset folder.")
        # Print out the first 15 words that actually exist so you can pick from them!
        print(f"💡 Here are some words that DO exist in your dataset: {sorted(list(set(all_available_folders)))[:15]}")

if __name__ == "__main__":
    compile_asl_dict()