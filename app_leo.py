# Libraries:
import streamlit as st
import numpy as np
import librosa
import soundfile as sf

# Import your ad detection and removal functions here
from utils.utils_leo import create_spectrogram, detect_ads, remove_ads_from_podcast

# Title:
st.title('Podcast Ad Skipper')
st.write("Play the original podcast, the podcast without ads, or listen to ads only.")

# Upload audio file
podcast_file = st.file_uploader("Choose a podcast file", type=["mp3", "wav"])

if podcast_file is not None:
    # Load the audio file
    y, sr = librosa.load(podcast_file, sr=22050)

    # Play original podcast
    if st.button("Play Original Podcast"):
        st.audio(podcast_file, format='audio/wav')

    # Run ad detection and removal
    if st.button("Detect and Remove Ads"):
        # Run detect_ads and remove_ads_from_podcast functions
        ad_segments = detect_ads(podcast_file, model, clip_duration=5) # We need to define the model !!! 
        podcast_without_ads, ads_only = remove_ads_from_podcast(podcast_file, ad_segments)

        # Save to files for playback
        sf.write("podcast_no_ads.wav", podcast_without_ads, sr)
        sf.write("ads_only.wav", ads_only, sr)
        st.success("Ad detection and removal complete!")

    # Play podcast without ads
    if st.button("Play Podcast Without Ads"):
        st.audio("podcast_no_ads.wav", format='audio/wav')

    # Play ads only
    if st.button("Play Ads Only"):
        st.audio("ads_only.wav", format='audio/wav')
