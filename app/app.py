import streamlit as st
import joblib
import speech_recognition as sr
import cv2
from PIL import Image
import base64
import numpy as np

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        b64_encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
            background-attachment: fixed;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background("backgroung.jpg")  # Ensure you have an image named background.jpg in your project folder

# Load models and vectorizer
lr_model = joblib.load('logistic_model.pkl')
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict using selected model
def predict(model, input_data, data_type='text'):
    try:
        if data_type == 'text':
            input_data_tfidf = vectorizer.transform([input_data])
            if model == svm_model:
                input_data_tfidf = input_data_tfidf.toarray()  # Convert to dense format
            prediction = model.predict(input_data_tfidf)
        elif data_type == 'audio':
            text = audio_to_text(input_data)
            if text is None:
                return "Error in recognizing audio"
            input_data_tfidf = vectorizer.transform([text])
            if model == svm_model:
                input_data_tfidf = input_data_tfidf.toarray()  # Convert to dense format
            prediction = model.predict(input_data_tfidf)
        elif data_type == 'video':
            frames = extract_frames(input_data)
            prediction = analyze_frames(frames)
        return "Hate Speech" if prediction == 1 else "Not Hate Speech"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to convert audio to text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioFile(audio_file)
    try:
        with audio_data as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Function to extract frames from video
def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Dummy function for frame analysis (to be replaced with actual model)
def analyze_frames(frames):
    # Placeholder for actual frame analysis logic
    return "No hate speech detected in video frames."

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Text Input'

# Function to set page in session state
def set_page(page):
    st.session_state.page = page

# Title of the app
st.markdown("<h1 style='text-align: center; color: white; font-size: 3em;'> Hate Hunter </h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Provide text, audio, or video input for analysis</h3>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.button("Text Input", on_click=set_page, args=("Text Input",))
st.sidebar.button("Audio Upload", on_click=set_page, args=("Audio Upload",))
st.sidebar.button("Video Upload", on_click=set_page, args=("Video Upload",))

if st.session_state.page == "Text Input":
    st.markdown("<h2 style='color: white;'>Enter Text for Prediction</h2>", unsafe_allow_html=True)
    text_input = st.text_area("Text Input", height=100)
    model_choice = st.selectbox("Choose a Model", ("Logistic Regression", "SVM"))
    if st.button("Submit Text"):
        if text_input:
            if model_choice == "Logistic Regression":
                prediction = predict(lr_model, text_input, 'text')
            elif model_choice == "SVM":
                prediction = predict(svm_model, text_input, 'text')
            st.markdown(f"<p style='color: white;'>Prediction: {prediction}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red;'>Please enter some text.</p>", unsafe_allow_html=True)

elif st.session_state.page == "Audio Upload":
    st.markdown("<h2 style='color: white;'>Upload an Audio File for Prediction</h2>", unsafe_allow_html=True)
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    model_choice = st.selectbox("Choose a Model", ("Logistic Regression", "SVM"))
    if st.button("Submit Audio"):
        if audio_file:
            if model_choice == "Logistic Regression":
                prediction = predict(lr_model, audio_file, 'audio')
            elif model_choice == "SVM":
                prediction = predict(svm_model, audio_file, 'audio')
            st.markdown(f"<p style='color: white;'>Prediction: {prediction}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red;'>Please upload an audio file.</p>", unsafe_allow_html=True)

elif st.session_state.page == "Video Upload":
    st.markdown("<h2 style='color: white;'>Upload a Video File for Prediction</h2>", unsafe_allow_html=True)
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    model_choice = st.selectbox("Choose a Model", ("Logistic Regression", "SVM"))
    if st.button("Submit Video"):
        if video_file:
            if model_choice == "Logistic Regression":
                prediction = predict(lr_model, video_file, 'video')
            elif model_choice == "SVM":
                prediction = predict(svm_model, video_file, 'video')
            st.markdown(f"<p style='color: white;'>Prediction: {prediction}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: red;'>Please upload a video file.</p>", unsafe_allow_html=True)
