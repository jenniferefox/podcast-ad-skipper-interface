# Libraries:
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from keras import models
from pydub import AudioSegment
import tempfile

# Import your ad detection and removal functions here
from utils.utils import create_spectrogram, detect_ads, remove_ads_from_podcast

# Load the model:
model = models.load_model('model.h5')

# Function to add background style
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Function to style buttons with CSS
def style_buttons():
    st.markdown("""
    <style>
    .stButton > button {
        color: black;
        background-color: grey;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: black;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

style_buttons()

# Main App layout with Sidebar
st.sidebar.title("Podcast Ad Skipper")
st.sidebar.write("An easy way to skip ads in your podcasts!")
st.sidebar.write("Upload a podcast and choose to play the original version, skip the ads, or listen to only the ads.")

# Upload audio file
st.sidebar.subheader("Upload Your Podcast")
podcast_file = st.sidebar.file_uploader("Choose a podcast file", type=["mp3", "wav"])

# Display the main header
st.title("🎧 Podcast Ad Skipper")
st.markdown("Play the original podcast, the podcast without ads, or listen to ads only.")


if podcast_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3" if podcast_file.name.endswith(".mp3") else ".wav") as tmp_file:
        tmp_file.write(podcast_file.read())
        tmp_file_path = tmp_file.name

    # Load the audio file with AudioSegment
    file_format = tmp_file_path.split('.')[-1]  # Get the file extension (either 'mp3' or 'wav')
    audio_file_before_ads = AudioSegment.from_file(tmp_file_path, format=file_format)

    # Play original podcast
    if st.button("▶️ Play Original Podcast"):
        st.audio(podcast_file, format='audio/wav')

    # Run ad detection and removal
    if st.button("🚫 Detect and Remove Ads"):
        st.info("Processing... Please wait.")

        # Initialize progress bar
        progress_bar = st.progress(0)

        # Step 1: Detect ads
        progress_bar.progress(25)
        # Run detect_ads and remove_ads_from_podcast functions
        ad_segments = detect_ads(tmp_file_path, model, clip_duration=5) # Detect ads using the model

        # Step 2: Remove ads from podcast
        progress_bar.progress(50)
        podcast_without_ads, ads_only = remove_ads_from_podcast(tmp_file_path, ad_segments)

        # Save to files for playback
        clean_podcast_path = "clean_podcast.mp3"
        ads_only_path = "ads_only.mp3"
        progress_bar.progress(75)
        podcast_without_ads.export(clean_podcast_path, format="mp3")
        ads_only.export(ads_only_path, format="mp3")

         # Final step: Completion
        progress_bar.progress(100)
        st.success("Ad detection and removal complete!")

 # Display two options for playing the processed podcasts

    if st.button("🎶 Play Podcast Without Ads"):
        clean_podcast_path = "clean_podcast.mp3"
        st.audio(clean_podcast_path, format="mp3")

    if st.button("🔊 Play Ads Only"):
        ads_only_path = "ads_only.mp3"
        st.audio(ads_only_path, format='mp3')
