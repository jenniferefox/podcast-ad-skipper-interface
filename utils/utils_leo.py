import pandas as pd
import numpy as np
import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.ndimage import zoom
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import layers, models, Model
from keras.optimizers import Adam
import tensorflow as tf
from pydub import AudioSegment

# Function to create a mel spectrogram

def create_spectrogram(wav_path, sr=22050):
    y, sr = librosa.load(wav_path, sr=sr)
    # Create mel spectrogram
    mel_spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,  # Number of mel bands
        fmax=8000)    # Maximum frequency
    # Convert to log scale and return
    return np.array(librosa.power_to_db(mel_spect, ref=np.max))



# Function to detect ads within a podcast
def detect_ads(podcast_file, model, clip_duration=5):
    """
    This function splits the podcast into clips, creates spectrograms, and uses the model to detect ads.
    podcast_file: Path to the podcast audio file (mp3)
    model: The trained model for ad detection
    clip_duration: Duration of each clip in seconds (default 5)
    return: List of ad segments (start_time, end_time) in seconds

    EXPLANATION:
    ongoing_ad Flag: Tracks whether an ad sequence is currently in progress.
    ad_start_time Variable: Stores the start time of the ongoing ad segment.
    Conditional Check for Consecutive Ads: If an ad prediction is followed by another ad, it extends the ongoing ad segment.
    If the following clip is not an ad, it finalizes the current ad segment and appends it to ad_segments.
    Final Check After Loop: If the podcast ends while an ad segment is ongoing, it appends this last ad segment to ad_segments.
    This way, the function will detect ad segments lasting longer than one clip duration and merge consecutive ad clips into single segments.
    Adjust the 0.5 threshold in if prediction[0][0] > 0.5 as needed based on model performance.
    """
    podcast = AudioSegment.from_file(podcast_file)  # Load podcast file
    podcast_duration = len(podcast) / 1000  # Duration in seconds
    print(f"Podcast duration: {podcast_duration} seconds.")

    ad_segments = []  # Store detected ad segments
    ongoing_ad = False  # Track if an ad segment is ongoing
    ad_start_time = 0   # Start time of the ongoing ad segment
    previous_prediction = 0  # Track previous clip prediction (0 = no ad, 1 = ad)

    # Process the podcast in chunks
    for i in range(0, int(podcast_duration), clip_duration):
        start_time = i * 1000  # Convert to milliseconds
        end_time = (i + clip_duration) * 1000
        clip = podcast[start_time:end_time]

        # Save the clip as a temporary wav file for processing
        clip_file = "temp_clip.wav"
        clip.export(clip_file, format="wav")

        # Create and validate spectrogram
        spectrogram = create_spectrogram(clip_file)
        if spectrogram.shape == (128, 216):
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension
            spectrogram = np.expand_dims(spectrogram, axis=0)   # Add batch dimension
        else:
            print(f"Clip starting at {start_time / 1000} seconds has incorrect shape: {spectrogram.shape}")
            os.remove(clip_file)
            continue  # Skip this clip if shape is incorrect

        # Predict using the model
        prediction = model.predict(spectrogram)
        current_prediction = int(prediction[0][0] > 0.5)  # Convert to binary (0 or 1)
        print(f"Prediction for clip starting at {start_time / 1000} seconds: {prediction}")

        # Check if the previous clip was an ad
        if current_prediction == 1: # Current clip detected as an ad
            if previous_prediction == 0: # Previous clip was not an ad
                ad_start_time = i # Start a new ad sequence
            ongoing_ad = True

        elif current_prediction == 0 and previous_prediction == 1:
            # End of an ad sequence when the current clip is not an ad but the previous was
            ad_segments.append((ad_start_time, i))  # Record the ad segment
            ongoing_ad = False

        # Update previous prediction to the current one for the next iteration
        previous_prediction = current_prediction

        # Remove the temporary file
        os.remove(clip_file)

    # Append any remaining ad segment at the end of the podcast
    if ongoing_ad:
        ad_segments.append((ad_start_time, int(podcast_duration)))

    print(f"Total ad segments detected: {len(ad_segments)}")
    return ad_segments

# Function to remove ads from a podcast

def remove_ads_from_podcast(podcast_file, ad_segments):
    """
    Removes the ad segments from the podcast and saves two files:
    - An ad-free podcast file
    - A file with only the ad segments
    podcast_file: Path to the podcast audio file
    ad_segments: List of tuples with (start_time, end_time) of ads in seconds
    """

    # Load
    podcast = AudioSegment.from_file(podcast_file) # Load the podcast file
    podcast_duration = len(podcast)

    # Initialize segments for the ad-free podcast and for the ads only
    clean_podcast = AudioSegment.empty()
    ads_only = AudioSegment.empty()
    current_time = 0

    for ad_start, ad_end in ad_segments:
        ad_start_ms = ad_start * 1000 # Convert to milliseconds
        ad_end_ms = ad_end * 1000
        # Add non-ad segment to the clean podcast
        clean_podcast += podcast[current_time:ad_start_ms]

        # Add ad segment to the ads_only file
        ads_only += podcast[ad_start_ms:ad_end_ms]

        # Update the current time
        current_time = ad_end_ms

    # Add the last segment of the podcast
    clean_podcast += podcast[current_time:podcast_duration]  # Add the last segment of the podcast

    return clean_podcast, ads_only
