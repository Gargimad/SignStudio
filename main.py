import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
from landing import show_landing_page

# Assuming these exist in your project environment
from recognition import calculate_joint_angles, get_top_predictions
from utils import draw_styled_skeleton

DATA_DIR = "./data"          # Creator Studio (custom words)
ASL_DATA_DIR = "./asl_data"  # Pre-defined ASL dictionary words
HISTORY_DIR = "./data_history"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASL_DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

st.set_page_config(page_title="SignStudio", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0E0E11;
}
.sentence-box {
    background: linear-gradient(135deg, #1F1F24, #2A2A31);
    border: 1px solid #FF4B4B;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.sentence-box h2 {
    font-size: 1.4rem !important;
    margin: 5px 0 0 0;
    font-weight: 600;
}
.workspace-title {
    margin: 0;
    font-size: 0.8rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
}
.prediction-box {
    background-color: #1A1A1E;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #FF4B4B;
}
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stSidebar"] {
    background-color: #1A1A1E !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

st.title("SignStudio")
st.markdown("Create, Learn, Translate")

@st.cache_data(ttl=3600)
def load_reference_library(path):
    library = {}
    if not os.path.exists(path):
        return library
    for file in os.listdir(path):
        if file.endswith(".json"):
            word = file.replace(".json", "")
            with open(os.path.join(path, file), "r") as f:
                library[word] = json.load(f)
    return library

def refresh_library():
    st.cache_data.clear()

# Session state
if "assembled_sentence" not in st.session_state:
    st.session_state.assembled_sentence = []
if "word_timestamps" not in st.session_state:
    st.session_state.word_timestamps = []
if "state_machine" not in st.session_state:
    st.session_state.state_machine = "SEARCHING"
if "last_word" not in st.session_state:
    st.session_state.last_word = ""
if "hold_counter" not in st.session_state:
    st.session_state.hold_counter = 0
if "camera_active" not in st.session_state:
    st.session_state.camera_active = True
if "pending_delete_word" not in st.session_state:
    st.session_state.pending_delete_word = None
if "pending_delete_type" not in st.session_state:
    st.session_state.pending_delete_type = None

# NEW: buffers for confidence & stability
if "confidence_buffer" not in st.session_state:
    st.session_state.confidence_buffer = []
if "stability_buffer" not in st.session_state:
    st.session_state.stability_buffer = []
if "last_angles" not in st.session_state:
    st.session_state.last_angles = None

def smooth_value(buffer, new_value, max_len=8):
    buffer.append(new_value)
    if len(buffer) > max_len:
        buffer.pop(0)
    return sum(buffer) / len(buffer)

def calculate_stability(current, previous):
    if previous is None:
        return 1.0
    diff = np.mean(np.abs(np.array(current) - np.array(previous)))
    stability = max(0.0, min(1.0, 1.0 - diff * 3))
    return stability

# Load libraries
creator_library = load_reference_library(DATA_DIR)
asl_library = load_reference_library(ASL_DATA_DIR)

# --- ROUTING FROM LANDING PAGE ---
mode = show_landing_page()

# Stop rendering the rest of the app if user is on landing page
if mode == "Landing Page":
    st.stop()

# --- SIDEBAR CONTROLS & HISTORY ---
st.sidebar.title("Controls")
app_mode = st.sidebar.selectbox("Select Mode", ["Custom Language (Live)", "ASL Video Translation (3s Capture)"])
st.sidebar.markdown("---")

if st.session_state.camera_active:
    if st.sidebar.button("Stop Camera", use_container_width=True):
        st.session_state.camera_active = False
        st.rerun()
else:
    if st.sidebar.button("Start Camera", use_container_width=True, type="primary"):
        st.session_state.camera_active = True
        st.rerun()

st.sidebar.markdown("---")

with st.sidebar.expander("Creator Studio Dictionary", expanded=False):
    if creator_library:
        for idx, w in enumerate(sorted(creator_library.keys()), 1):
            st.write(f"**{idx}.** {w.upper()} ({creator_library[w].get('handedness', 'Right')})")
    else:
        st.caption("No custom JSON datasets found in /data/ directory.")

with st.sidebar.expander("ASL Dictionary", expanded=False):
    if asl_library:
        for idx, w in enumerate(sorted(asl_library.keys()), 1):
            st.write(f"**{idx}.** {w.upper()} ({asl_library[w].get('handedness', 'Right')})")
    else:
        st.caption("No ASL JSON datasets found in /asl_data/ directory.")

with st.sidebar.expander("Sentence History", expanded=False):
    history_files = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")], reverse=True)
    if history_files:
        for h_file in history_files:
            with open(os.path.join(HISTORY_DIR, h_file), "r") as f:
                record = json.load(f)
            st.markdown(f"🗓️ **Saved at:** `{record['saved_at']}`")
            st.markdown(f"💬 **Sentence:** `{record['full_text']}`")
    else:
        st.caption("No saved sentences found.")

st.sidebar.markdown("---")

# Delete confirmation UI (global)
if st.session_state.pending_delete_word:
    st.sidebar.warning(f"Delete '{st.session_state.pending_delete_word.upper()}' from {st.session_state.pending_delete_type} dictionary?")
    col_c1, col_c2 = st.sidebar.columns(2)
    with col_c1:
        if st.button("Confirm delete", key="confirm_delete"):
            if st.session_state.pending_delete_type == "ASL":
                path = os.path.join(ASL_DATA_DIR, f"{st.session_state.pending_delete_word}.json")
            else:
                path = os.path.join(DATA_DIR, f"{st.session_state.pending_delete_word}.json")
            if os.path.exists(path):
                os.remove(path)
            st.session_state.pending_delete_word = None
            st.session_state.pending_delete_type = None
            refresh_library()
            st.sidebar.success("Deleted.")
            st.rerun()
    with col_c2:
        if st.button("Cancel", key="cancel_delete"):
            st.session_state.pending_delete_word = None
            st.session_state.pending_delete_type = None
            st.rerun()

# --- TRANSLATOR TAB ---
if mode == "Translator":
    # Setup triggers based on mode selection
    asl_translate_trigger = False
    record_button = False
    new_word = ""
    asl_new_word = ""
    asl_record_button = False

    if app_mode == "Custom Language (Live)":
        st.subheader("Creator Studio (Live Custom Language)")
        new_word = st.text_input("New Custom Word:", placeholder="e.g., family").strip().lower()
        record_button = st.button("Capture 60 Frame Average (Custom)", use_container_width=True)
    else:
        st.subheader("ASL Video Translation (3s Capture)")
        st.markdown("Press the button below to record a **3-second movement** to translate into an ASL word.")
        asl_translate_trigger = st.button("🎬 Record ASL Movement (3s)", use_container_width=True, type="primary")
        st.markdown("---")
        st.markdown("ASL Blueprint Creator (pre-defined dictionary)")
        asl_new_word = st.text_input("New ASL Word:", placeholder="e.g., hello").strip().lower()
        asl_record_button = st.button("Capture 60 Frame Average (ASL)", use_container_width=True)

    # Mediapipe initialization
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Camera Feed:")
        frame_slot = st.empty()

    with col2:
        current_sentence_str = " ".join(st.session_state.assembled_sentence)
        if not current_sentence_str:
            current_sentence_str = "Sign words to make a sentence!"

        st.markdown(f"""
        <div class="sentence-box">
        <p class="workspace-title">Current sentence</p>
        <h2 style='margin: 10px 0 0 0;'>{current_sentence_str}</h2>
        </div>
        """, unsafe_allow_html=True)

        predictions_slot = st.empty()
        progress_slot = st.empty()
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💾 Save Sentence", use_container_width=True, type="primary"):
                if st.session_state.assembled_sentence:
                    timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    readable_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_payload = {
                        "saved_at": readable_ts,
                        "full_text": " ".join(st.session_state.assembled_sentence),
                        "word_timeline": st.session_state.word_timestamps
                    }
                    with open(os.path.join(HISTORY_DIR, f"sentence_{timestamp_id}.json"), "w") as f:
                        json.dump(save_payload, f)
                    st.session_state.assembled_sentence = []
                    st.session_state.word_timestamps = []
                    st.rerun()
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.assembled_sentence = []
                st.session_state.word_timestamps = []
                st.rerun()
        with c3:
            if st.button("Delete last word", use_container_width=True):
                if st.session_state.assembled_sentence:
                    st.session_state.assembled_sentence.pop()
                    st.session_state.word_timestamps.pop()
                st.rerun()

    # Camera processing loop
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        with HandLandmarker.create_from_options(options) as landmarker:
            while st.session_state.camera_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect(mp_image)

                if detection_result.hand_landmarks:
                    draw_styled_skeleton(rgb_frame, detection_result.hand_landmarks[0])

                # MODE 1: Custom Language (Save blueprint word to /data/)
                if app_mode == "Custom Language (Live)" and record_button and new_word:
                    for countdown in range(3, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"STEADY IN: {countdown}", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        time.sleep(1)

                    captured_angles = []
                    saved_handedness = "Right"
                    for f_idx in range(60):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"PROCESSING SAMPLES ({f_idx+1}/60)", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        rgb_snap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_snap = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_snap)
                        res = landmarker.detect(mp_snap)
                        if res.hand_landmarks:
                            draw_styled_skeleton(rgb_snap, res.hand_landmarks[0])
                            live_landmarks = [[l.x, l.y, l.z] for l in res.hand_landmarks[0]]
                            angles = calculate_joint_angles(live_landmarks)
                            captured_angles.append(angles.tolist())
                            if res.handedness:
                                saved_handedness = res.handedness[0][0].category_name
                        frame_slot.image(rgb_snap, channels="RGB")
                        time.sleep(0.03)
                    if captured_angles:
                        out_payload = {"handedness": saved_handedness, "angles": captured_angles}
                        with open(os.path.join(DATA_DIR, f"{new_word}.json"), "w") as f:
                            json.dump(out_payload, f)
                        refresh_library()
                        st.sidebar.success(f"Saved custom blueprint for '{new_word.upper()}'!")
                        time.sleep(1)
                        st.rerun()
                    record_button = False

                # MODE 1b: ASL Blueprint Creator (Save blueprint word to /asl_data/)
                if app_mode == "ASL Video Translation (3s Capture)" and asl_record_button and asl_new_word:
                    for countdown in range(3, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"STEADY IN: {countdown}", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        time.sleep(1)

                    captured_angles = []
                    saved_handedness = "Right"
                    for f_idx in range(60):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"PROCESSING SAMPLES ({f_idx+1}/60)", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        rgb_snap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_snap = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_snap)
                        res = landmarker.detect(mp_snap)
                        if res.hand_landmarks:
                            draw_styled_skeleton(rgb_snap, res.hand_landmarks[0])
                            live_landmarks = [[l.x, l.y, l.z] for l in res.hand_landmarks[0]]
                            angles = calculate_joint_angles(live_landmarks)
                            captured_angles.append(angles.tolist())
                            if res.handedness:
                                saved_handedness = res.handedness[0][0].category_name
                        frame_slot.image(rgb_snap, channels="RGB")
                        time.sleep(0.03)
                    if captured_angles:
                        out_payload = {"handedness": saved_handedness, "angles": captured_angles}
                        with open(os.path.join(ASL_DATA_DIR, f"{asl_new_word}.json"), "w") as f:
                            json.dump(out_payload, f)
                        refresh_library()
                        st.sidebar.success(f"Saved ASL blueprint for '{asl_new_word.upper()}'!")
                        time.sleep(1)
                        st.rerun()
                    asl_record_button = False

                # MODE 2: ASL 3-Second Video Capture & Translate (uses /asl_data/)
                elif app_mode == "ASL Video Translation (3s Capture)" and asl_translate_trigger:
                    for countdown in range(3, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        cv2.putText(frame, f"START SIGNING IN: {countdown}", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 165), 4)
                        frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        time.sleep(1)

                    asl_buffer_angles = []
                    detected_handedness = "Right"
                    start_time = time.time()

                    while time.time() - start_time < 3.0:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        elapsed = time.time() - start_time
                        cv2.putText(frame, f"RECORDING MOVEMENT: {3.0 - elapsed:.1f}s", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

                        rgb_snap = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_snap = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_snap)
                        res = landmarker.detect(mp_snap)

                        if res.hand_landmarks:
                            draw_styled_skeleton(rgb_snap, res.hand_landmarks[0])
                            live_landmarks = [[l.x, l.y, l.z] for l in res.hand_landmarks[0]]
                            angles = calculate_joint_angles(live_landmarks)
                            asl_buffer_angles.append(angles.tolist())
                            if res.handedness:
                                detected_handedness = res.handedness[0][0].category_name

                        frame_slot.image(rgb_snap, channels="RGB")
                        time.sleep(0.02)

                    if asl_buffer_angles and asl_library:
                        mid_idx = len(asl_buffer_angles) // 2
                        representative_landmarks = [[l.x, l.y, l.z] for l in detection_result.hand_landmarks[0]] \
                            if detection_result.hand_landmarks else live_landmarks

                        ranked_matches = get_top_predictions(representative_landmarks, asl_library, detected_handedness)
                        if ranked_matches and ranked_matches[0]["confidence"] > 65.0:
                            matched_word = ranked_matches[0]['word'].upper()
                            current_time_str = datetime.now().strftime("%H:%M:%S")
                            st.session_state.assembled_sentence.append(matched_word)
                            st.session_state.word_timestamps.append({"word": matched_word, "time": current_time_str})
                            st.sidebar.success(f"Added ASL Word: {matched_word}")
                        else:
                            st.sidebar.error("Could not confidently translate that movement.")
                    else:
                        st.sidebar.error("No hand signals detected during the 3-second window.")

                    asl_translate_trigger = False
                    time.sleep(1)
                    st.rerun()

                # DEFAULT LIVE RUNNING PREDICTION (Custom mode, uses /data/)
                if app_mode == "Custom Language (Live)":
                    if detection_result.hand_landmarks and creator_library:
                        live_list = [[l.x, l.y, l.z] for l in detection_result.hand_landmarks[0]]
                        handedness_label = "Right" if not detection_result.handedness else detection_result.handedness[0][0].category_name

                        # Stability
                        stability_raw = calculate_stability(live_list, st.session_state.last_angles)
                        st.session_state.last_angles = live_list
                        stability_smooth = smooth_value(st.session_state.stability_buffer, stability_raw)

                        ranked_matches = get_top_predictions(live_list, creator_library, handedness_label)
                        if ranked_matches:
                            top_match = ranked_matches[0]
                            raw_conf = top_match["confidence"]
                            smooth_conf = smooth_value(st.session_state.confidence_buffer, raw_conf)

                            if smooth_conf >= 85:
                                label = "🟢 Excellent Match"
                                bar_color = "#00FF88"
                            elif smooth_conf >= 70:
                                label = "🟡 Strong Match"
                                bar_color = "#FFD447"
                            elif smooth_conf >= 50:
                                label = "🟠 Weak Match"
                                bar_color = "#FF8C42"
                            else:
                                label = "🔴 Uncertain Match"
                                bar_color = "#FF4B4B"

                            if stability_smooth >= 0.85:
                                stability_label = "🟢 Very Stable Hand"
                            elif stability_smooth >= 0.65:
                                stability_label = "🟡 Moderately Stable"
                            else:
                                stability_label = "🔴 Unstable / Moving"

                            pred_html = f"""
                            <div class='prediction-box'>
                                <h4>Recognition Engine</h4>
                                <p><b>{top_match['word'].upper()}</b></p>
                                <p>{label}</p>
                                <p>Confidence: {smooth_conf:.1f}%</p>
                                <p>{stability_label}</p>
                                <p>Stability: {stability_smooth*100:.1f}%</p>
                                <hr>
                            """

                            for rank, match in enumerate(ranked_matches, 1):
                                pred_html += f"<p><b>{rank}. {match['word'].upper()}</b> — {match['confidence']:.1f}%</p>"

                            pred_html += "</div>"

                            predictions_slot.markdown(pred_html, unsafe_allow_html=True)

                            progress_slot.markdown(
                                f"""
                                <div style="height:20px; width:100%; background:#333; border-radius:10px;">
                                    <div style="height:20px; width:{smooth_conf}%; background:{bar_color};
                                                border-radius:10px;"></div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # State machine logic
                            if smooth_conf > 80.0:
                                if st.session_state.state_machine == "WAIT_FOR_RELEASE" and top_match["word"] != st.session_state.last_word:
                                    st.session_state.state_machine = "SEARCHING"

                                if st.session_state.state_machine == "SEARCHING":
                                    st.session_state.last_word = top_match["word"]
                                    st.session_state.state_machine = "FOUND"
                                    st.session_state.hold_counter = 1

                                elif st.session_state.state_machine == "FOUND":
                                    if top_match["word"] == st.session_state.last_word:
                                        st.session_state.hold_counter += 1
                                        if st.session_state.hold_counter >= 6:
                                            st.session_state.state_machine = "CONFIRMED"
                                    else:
                                        st.session_state.state_machine = "SEARCHING"

                                elif st.session_state.state_machine == "CONFIRMED":
                                    word_upper = top_match["word"].upper()
                                    current_time_str = datetime.now().strftime("%H:%M:%S")
                                    st.session_state.assembled_sentence.append(word_upper)
                                    st.session_state.word_timestamps.append({"word": word_upper, "time": current_time_str})
                                    st.session_state.state_machine = "WAIT_FOR_RELEASE"
                                    st.rerun()
                            else:
                                st.session_state.state_machine = "SEARCHING"
                                st.session_state.hold_counter = 0
                        else:
                            predictions_slot.markdown(
                                "<div class='prediction-box'><h4>Recognition Engine</h4><p><i>No matches yet.</i></p></div>",
                                unsafe_allow_html=True
                            )
                            progress_slot.progress(0.0)
                            st.session_state.state_machine = "SEARCHING"
                    else:
                        predictions_slot.markdown(
                            "<div class='prediction-box'><h4>Recognition Engine</h4><p><i>No hand detected.</i></p></div>",
                            unsafe_allow_html=True
                        )
                        progress_slot.progress(0.0)
                        st.session_state.state_machine = "SEARCHING"
                else:
                    predictions_slot.markdown(
                        "<div class='prediction-box'><h4>ASL Phrase Builder Mode</h4><p>Use the buttons to insert words chronologically.</p></div>",
                        unsafe_allow_html=True
                    )

                frame_slot.image(rgb_frame, channels="RGB")
            cap.release()
    else:
        frame_slot.info("System Engine Offline. Click button to wake webcam feed arrays.")

# --- ASL DICTIONARY TAB (view + preview + delete) ---
if mode == "ASL Dictionary":
    st.header("ASL Dictionary (Pre-defined Blueprints)")

    asl_library = load_reference_library(ASL_DATA_DIR)

    if not asl_library:
        st.info("No ASL blueprints found...")
    else:
        for word, data in sorted(asl_library.items()):
            with st.expander(f"👐 {word.upper()}"):
                st.write(f"**Handedness:** {data.get('handedness', 'Unknown')}")
                st.write(f"**Frames:** {len(data.get('angles', []))}")
                if "angles" in data and data["angles"]:
                    st.line_chart(data["angles"])
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button(f"Delete {word.upper()}", key=f"delete_asl_{word}"):
                        st.session_state.pending_delete_word = word
                        st.session_state.pending_delete_type = "ASL"
                        st.rerun()
                with col_d2:
                    st.caption("Deletion requires confirmation in the sidebar.")

# --- CREATOR STUDIO DICTIONARY TAB (view + preview + delete) ---
if mode == "Creator Studio":
    st.header("Creator Studio Dictionary (Custom Blueprints)")
    creator_library = load_reference_library(DATA_DIR)
    if not creator_library:
        st.info("No custom blueprints found in /data/. Create some in the Translator tab (Custom mode).")
    else:
        for word, data in sorted(creator_library.items()):
            with st.expander(f"🎨 {word.upper()}"):
                st.write(f"**Handedness:** {data.get('handedness', 'Unknown')}")
                st.write(f"**Frames:** {len(data.get('angles', []))}")
                if "angles" in data and data["angles"]:
                    st.line_chart(data["angles"])
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button(f"Delete {word.upper()}", key=f"delete_custom_{word}"):
                        st.session_state.pending_delete_word = word
                        st.session_state.pending_delete_type = "Custom"
                        st.rerun()
                with col_d2:
                    st.caption("Deletion requires confirmation in the sidebar.")
