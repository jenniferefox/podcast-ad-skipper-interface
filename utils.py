
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




# ----------------- Functions to create the spectrogram ----------------- #

def create_spectrogram(audio_file_wav, sr=22050):
    """
    Converts wav files to spectrograms.
    sr=None to keep the original sample rate
    """

    #data: is an array representing the amplitude of the audio signal at each sample.
    #sample_rate: is the sampling rate (samples per second)
    data, sample_rate = librosa.load(audio_file_wav, sr=sr)
    spectrogram = librosa.feature.melspectrogram(
        y=data,
        sr=sample_rate,
        n_mels=128,  # Number of mel bands
        fmax=8000    # Maximum frequency
    )
    # Short-time Fourier transform
    return np.array(librosa.power_to_db(spectrogram, ref=np.max))  # Convert to decibel scale

# ----------------- Functions to process the demo podcasts ----------------- #



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
