import streamlit as st
from PIL import Image
import os
import io
import numpy as np
import pandas as pd
from datetime import datetime
import face_recognition

# ========== CONFIG ==========
st.set_page_config(page_title="Network Attendance (Streamlit)", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = os.path.abspath(os.getcwd())
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ATT_DIR = os.path.join(BASE_DIR, "attendence_sheet")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

# Path you uploaded earlier (kept here for convenience)
EXISTING_SCRIPT_PATH = "/mnt/data/face_rec.py"

# ========== CSS for nicer full-screen cards ==========
st.markdown(
    """
    <style>
    .card { background: linear-gradient(180deg,#ffffff 0%, #f7fbff 100%); border-radius:12px; padding:18px; box-shadow: 0 12px 30px rgba(20,20,60,0.06); }
    .card:hover { transform: translateY(-6px); transition: .12s ease; }
    .big { font-size:18px; font-weight:600; }
    .muted { color:#6b7280; }
    /* make layout fill viewport */
    [data-testid="stAppViewContainer"] > .main { min-height: calc(100vh - 32px); }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== helpers ==========
def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in s).strip()

def load_registered_encodings(images_dir=IMAGES_DIR):
    known_encs = []
    known_names = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(images_dir, fn)
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if len(encs) == 0:
                # skip images without found faces
                continue
            known_encs.append(encs[0])
            known_names.append(os.path.splitext(fn)[0])
        except Exception as e:
            # skip problematic files
            st.write(f"Warning: failed to process {fn}: {e}")
    return known_encs, known_names

def append_attendance(name: str, csv_dir=ATT_DIR, prefix="attendance"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    csv_name = f"{prefix}_{date_str}.csv"
    csv_path = os.path.join(csv_dir, csv_name)
    row = {"name": name, "date": date_str, "time": time_str, "timestamp": now.isoformat()}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype=str)
        df = df.append(row, ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    return csv_path

def recognize_from_image_bytes(image_bytes, threshold=0.5):
    """
    Given image bytes, compute encodings and compare to registered encodings.
    Returns list of matches for each face found.
    """
    img = face_recognition.load_image_file(io.BytesIO(image_bytes))
    encs = face_recognition.face_encodings(img)
    if len(encs) == 0:
        return [{"matched": None, "reason": "no_face"}]
    known_encs, known_names = load_registered_encodings()
    if len(known_encs) == 0:
        return [{"matched": None, "reason": "no_registered_faces"}]
    results = []
    for e in encs:
        distances = face_recognition.face_distance(known_encs, e)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        if best_distance <= threshold:
            results.append({"matched": known_names[best_idx], "distance": best_distance})
            append_attendance(known_names[best_idx])
        else:
            results.append({"matched": None, "distance": best_distance})
    return results

# ========== UI ==========
st.title("📸 Automatic Attendance — Streamlit Network UI")
st.write("Full-screen manager for registrations, attendance marking (via client camera), and CSV viewing.")

cols = st.columns(3)

# ------- Register card -------
with cols[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🆕 Register")
    st.write("Upload a person's photo or capture one from your camera. Saved to the server `images/` folder.")
    name = st.text_input("Name / Roll number (used as filename)", key="reg_name")
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"], key="reg_upload")
    if st.button("Save uploaded image", key="save_upload"):
        if uploaded is None:
            st.warning("Choose an image to upload first.")
        elif not name.strip():
            st.warning("Enter a name/roll to use as filename.")
        else:
            safe = sanitize_filename(name)
            ext = os.path.splitext(uploaded.name)[1] or ".jpg"
            filename = f"{safe}{ext}"
            path = os.path.join(IMAGES_DIR, filename)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved as {filename}")
            st.image(Image.open(path), caption=filename, use_column_width=True)

    st.write("---")
    st.write("Or capture from your webcam:")
    cam_name = st.text_input("Name for camera capture", key="cam_reg_name")
    cam_image = st.camera_input("Take photo (browser will ask permission)", key="reg_camera_input")
    if cam_image is not None and st.button("Save camera capture", key="save_cam"):
        if not cam_name.strip():
            st.warning("Provide a name before saving the capture.")
        else:
            safe = sanitize_filename(cam_name)
            filename = f"{safe}.jpg"
            path = os.path.join(IMAGES_DIR, filename)
            img_bytes = cam_image.getvalue()
            with open(path, "wb") as f:
                f.write(img_bytes)
            st.success(f"Saved camera capture as {filename}")
            st.image(Image.open(path), caption=filename, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------- Mark attendance -------
with cols[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Mark attendance")
    st.write("Use the camera to take a snapshot (client-side) and send it to server to recognize.")
    threshold = st.slider("Matching threshold (lower = stricter)", 0.25, 0.85, 0.5, 0.01)
    st.write("Capture from camera:")
    snap = st.camera_input("Take a snapshot to mark attendance")
    if snap is not None and st.button("Run recognition on snapshot"):
        image_bytes = snap.getvalue()
        with st.spinner("Recognizing..."):
            res = recognize_from_image_bytes(image_bytes, threshold=threshold)
        st.write("Recognition result:")
        st.json(res)
        matched = [r for r in res if r.get("matched")]
        if matched:
            st.success(f"Marked attendance for: {', '.join([m['matched'] for m in matched])}")
        else:
            st.info("No matches found (or no faces detected).")

    st.write("---")
    st.write("Or upload a photo to mark attendance (useful if client can't use camera):")
    up_for_mark = st.file_uploader("Upload snapshot(s) for marking", type=["jpg","jpeg","png"], accept_multiple_files=True, key="mark_uploads")
    if up_for_mark and st.button("Run recognition on uploaded images"):
        all_results = {}
        for f in up_for_mark:
            bi = f.getbuffer().tobytes()
            res = recognize_from_image_bytes(bi, threshold=threshold)
            all_results[f.name] = res
        st.json(all_results)
    st.markdown('</div>', unsafe_allow_html=True)

# ------- See CSVs -------
with cols[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 See CSV attendance sheet")
    st.write("List, preview and download daily attendance CSVs from `attendence_sheet/`.")
    csv_files = [f for f in sorted(os.listdir(ATT_DIR), reverse=True) if f.lower().endswith(".csv")]
    if not csv_files:
        st.info("No CSV files found yet. Attendance will create files named like attendance_YYYY-MM-DD.csv")
    else:
        sel = st.selectbox("Choose CSV to preview", csv_files)
        if st.button("Load CSV"):
            path = os.path.join(ATT_DIR, sel)
            df = pd.read_csv(path)
            st.write(f"Rows: {len(df)}")
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name=sel, mime="text/csv")
    st.write("---")
    st.write("Quick maintenance:")
    if st.button("Reload registered encodings (debug)"):
        encs, names = load_registered_encodings()
        st.write(f"Found {len(encs)} registered faces: {names}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.markdown(
    f"""
    **Notes & extras**  
    - The app saves registered images into: `{IMAGES_DIR}`  
    - Attendance CSVs are saved into: `{ATT_DIR}` with names `attendance_YYYY-MM-DD.csv`  
    - You can inspect your existing face recognition script here: [face_rec.py]({f"file://{EXISTING_SCRIPT_PATH}"})  
    - To deploy publicly, run Streamlit with `--server.address=0.0.0.0` and configure TLS / auth in front proxy.
    """
)

# deployment instructions
st.info(
    "To make this accessible on the network run:\n\n"
    "`streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501`\n\n"
    "Then open http://<server-ip>:8501 from client browsers on your network."
)
