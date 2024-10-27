
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
import cv2 as cv
from keras import layers, models, Model
from keras.optimizers import Adam
import tensorflow as tf
from pydub import AudioSegment




# ----------------- Functions to create the spectrogram ----------------- #

def create_spectrogram(audio_file_wav, sr=16000):
    """
    Converts wav files to spectrograms.
    """

    #data: is an array representing the amplitude of the audio signal at each sample.
    #sample_rate: is the sampling rate (samples per second)
    data, sample_rate = librosa.load(audio_file_wav, sr=sr) # sr=None to keep the original sample rate (we can change this if needed)
    spectrogram = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_mels=128,  # Number of mel bands
        fmax=8000    # Maximum frequency
    )
    # Short-time Fourier transform
    return np.array(librosa.power_to_db(spectrogram, ref=np.max))  # Convert to decibel scale



# ----------------- Functions to process the spectrograms ----------------- #

def resize_spectrogram(spectrogram, output_size):
    sp_row, sp_col = spectrogram.shape
    out_row, out_col = output_size
    resized_spec = zoom(spectrogram, (out_row/sp_row, out_col/sp_col))
    return resized_spec

def minmax_scaler(spectrogram):
    min_val = np.min(spectrogram)
    max_val = np.max(spectrogram)

    normalised_spectrogram = (spectrogram - min_val) / (max_val - min_val)

    return normalised_spectrogram

def reshape_spectrogram(spectrogram):
    return np.stack((spectrogram, spectrogram, spectrogram), axis=2)



# ----------------- Functions to process the demo podcasts ----------------- #

def detect_ads(podcast_file, model, clip_duration=5):
    """
    This function splits the podcast into clips, creates spectrograms, and passes them to the model to detect ads.
    podcast_file: Path to the podcast audio file (mp3)
    model: The trained model for ad detection
    clip_duration: Duration of each clip in seconds (default 5)
    return: List of ad segments (start_time, end_time) in seconds
    """

    # Load the podcast file
    podcast = AudioSegment.from_file(podcast_file) # Load the new podcast file
    podcast_duration = len(podcast) / 1000  # Duration in seconds

    # List to hold the ad segments
    ad_segments = []

    # Process the podcast in chunks of clip_duration seconds
    for i in range(0, int(podcast_duration), clip_duration):
        start_time = i * 1000  # Convert to milliseconds
        end_time = (i + clip_duration) * 1000

        # Extract the clip from the podcast
        clip = podcast[start_time:end_time]

        # Save the clip as a temporary wav file (for librosa to process)
        clip_file = "temp_clip.wav"
        clip.export(clip_file, format="wav")

        # Create a spectrogram for the clip
        spectrogram = create_spectrogram(clip_file) # We already have this function
        resized_spectrogram =resize_spectrogram(spectrogram, (224,224))
        scaled_spectrogram = minmax_scaler(resized_spectrogram)
        reshaped_spectrogram = reshape_spectrogram(scaled_spectrogram)

        # Convert the spectrogram to a numpy array and pass it to the model
        spectrogram_np = np.expand_dims(reshaped_spectrogram, axis=0)  # Add batch dimension
        prediction = model.predict(spectrogram_np) # Use the model to predict

        # If the model predicts 'ad' it will mark this segment as an ad (1)
        if prediction == 1:
            ad_segments.append((i, i + clip_duration))

        # Clean up the temporary file
        os.remove(clip_file)

    return ad_segments


def remove_ads_from_podcast(podcast_file, ad_segments):
    """
    Removes the ad segments from the podcast and returns an ad-free podcast.
    podcast_file: Path to the podcast audio file
    ad_segments: List of tuples with (start_time, end_time) of ads in seconds
    return: An AudioSegment object, the podcast without ads
    """
    podcast = AudioSegment.from_file(podcast_file) # Load the podcast file
    podcast_duration = len(podcast)

    clean_podcast = AudioSegment.empty() # Create an empty AudioSegment object
    current_time = 0

    for ad_start, ad_end in ad_segments:
        ad_start_ms = ad_start * 1000 # Convert to milliseconds
        ad_end_ms = ad_end * 1000

        clean_podcast += podcast[current_time:ad_start_ms] # Add the non-ad segment to the clean podcast
        current_time = ad_end_ms  # Update the current time

    clean_podcast += podcast[current_time:podcast_duration]  # Add the last segment of the podcast

    return clean_podcast
