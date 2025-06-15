import cv2
import streamlit as st
import numpy as np
from datetime import datetime

def apply_filter(frame, filter_type):
    if filter_type == "Edge Detection":
        return cv2.Canny(frame, 100, 200)
    elif filter_type == "Face Detection":
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame
    elif filter_type == "Motion Detection":
        static_back = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if static_back is None:
            static_back = gray
            return frame
        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        return thresh_frame
    return frame

def main():
    st.title("Webcam Capture & Analysis Application")
    st.sidebar.title("Settings")
    filter_option = st.sidebar.radio("Choose a Feature:", ["None", "Edge Detection", "Face Detection", "Motion Detection"])
    
    img_capture = cv2.VideoCapture(0)
    if not img_capture.isOpened():
        st.error("Could not access webcam.")
        return
    
    stframe = st.empty()
    capture_button = st.sidebar.button("Capture Image", key="capture_button")
    
    while True:
        res, frame = img_capture.read()
        if not res:
            st.error("Failed to capture image.")
            break
        
        frame = apply_filter(frame, filter_option)
        stframe.image(frame, channels="BGR" if filter_option in ["None", "Face Detection"] else "GRAY")
        
        if capture_button:
            filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Image saved as {filename}")
            break
    
    img_capture.release()
    
if __name__ == "__main__":
    main()
