import cv2
import streamlit as st
import numpy as np
from datetime import datetime

# Global variable to store the static background for motion detection
static_back = None

def apply_filter(frame, filter_type):
    global static_back

    if filter_type == "Edge Detection":
        return cv2.Canny(frame, 100, 200)

    elif filter_type == "Face Detection":
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    elif filter_type == "Motion Detection":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize background frame
        if static_back is None:
            static_back = gray
            return frame

        # Compute absolute difference between background and current frame
        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh_frame.copy(),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    return frame


def main():
    st.title("📷 Webcam Capture & Analysis App")
    st.sidebar.title("⚙️ Settings")
    filter_option = st.sidebar.radio("Choose a Feature:", 
                                     ["None", "Edge Detection", "Face Detection", "Motion Detection"])

    # Initialize image history
    if 'captured_images' not in st.session_state:
        st.session_state['captured_images'] = []

    img_capture = cv2.VideoCapture(0)
    if not img_capture.isOpened():
        st.error("❌ Could not access webcam.")
        return

    stframe = st.empty()
    capture_button = st.sidebar.button("📸 Capture Image", key="capture_button")

    while True:
        res, frame = img_capture.read()
        if not res:
            st.error("Failed to capture image.")
            break

        processed_frame = apply_filter(frame.copy(), filter_option)
        is_gray = filter_option == "Edge Detection"
        stframe.image(processed_frame, channels="GRAY" if is_gray else "BGR")

        # Save image if button pressed
        if capture_button:
            filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.session_state['captured_images'].append(filename)
            st.success(f"✅ Image saved as `{filename}`")
            break

    img_capture.release()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🖼️ Captured Images")
    for img_path in st.session_state['captured_images']:
        st.sidebar.image(img_path, width=150)

if __name__ == "__main__":
    main()
