
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

# ----------------- Functions to create the clips ----------------- #
def split_files(original_file, ad_list, podcast_name, output_directory, run_env="local"):

    """
    This function takes an original audio file name, list of integers showing
    when each ad starts and ends and a podcast name and splits up the original
    file into 5 second chunks, naming each one according to whether it contains
    ads or not.
    """

    # Create a folder for the podcast and their clips:
    podcast_folder = os.path.join(output_directory, podcast_name)

    if run_env == "local":
        #Check if the folder already exists and has any .mp3 files
        if os.path.exists(podcast_folder) and any(fname.endswith('.wav') for fname in os.listdir(podcast_folder)):
            print(f"Skipping {podcast_name} because it has already been processed.")
            return 'skipped'

        # Create the directory if doesnt exist:
        if not os.path.exists(podcast_name):
            os.makedirs(podcast_folder)
            print(f"Created folder: {podcast_folder}")


    # Determine the file extension and load the audio file accordingly
    file_extension = os.path.splitext(original_file)[1].lower()

    if file_extension == '.mp3':
        new_audio = AudioSegment.from_mp3(original_file)
    elif file_extension == '.wav':
        new_audio = AudioSegment.from_wav(original_file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only .mp3 and .wav are supported.")

    # Save duration
    duration = int(new_audio.duration_seconds)

    # Set default to no_ad
    is_ad = '0'

    # If the ad_list doesn't start with 0, then the ads don't start straight away.
    # in this case, insert '0' first in the list so that a segment is created at the start.
    if ad_list[0] != 0:
        ad_list.insert(0, 0)
        is_ad = '1'

    # Add duration at the end so that the end segments can be made.
    if ad_list[-1] != duration:
        ad_list.append(duration)

    #Go through each segement in the list, label whether the section is an ad or not
    for index in range(0,len(ad_list)-1):
        start = ad_list[index]
        end = ad_list[index+1]
        # Toggle between 'ad' and 'no_ad'
        if is_ad == '1':
            is_ad = '0'
        else:
            is_ad = '1'

        # Go through each second in the segment and create a new 5 second clip from here.
        # Stop before the end of the segment so that only 5 second clips are created
        for tc in range(start, (end-4)):
            start_clip = tc*1000 #pydub works with milliseconds, so seconds are converted here
            end_clip = (tc+5)*1000

            # Construct the file path for saving
            output_file = os.path.join(podcast_folder, f'{is_ad}_{tc}_{duration}_{podcast_name}.wav')

            if run_env == "local":
                # Save clip locally:
                new_audio[start_clip:end_clip].export(output_file, format='wav')
                print(f"Saved clip: {output_file}")


    is_ad = '0'
    return 'finished'

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

def get_features_model(clip_audio_files, run_env="gc"):
    """
    Creates spectrograms and converts into np arrays.
    """
    CORRECT_SPECTROGRAM_SHAPE = (128, 216)
    spectrograms = [] # This will store the spectrograms of each clip
    labels = []  # This will store the labels of each clip
    seconds = []  # Number of seconds to consider for each clip
    durations = []  # Duration of the full audio file
    podcast_names = []  # This will store the podcast names of each clip

    # Iterate over all files in the directory
    if run_env == "local":
        file_list = os.listdir(clip_audio_files)
    elif run_env == 'gc':
        file_list = clip_audio_files

    for filename in file_list:
        # Split the filename by underscore
        if run_env == "local":
            filename_parts = filename.split('_')
        elif run_env == 'gc':
            filename_parts = filename.name.split('/')[1].split('_')


        # Extract 0 or 1 from the first part of the filename (label: ad or no_ad)
        is_ad = int(filename_parts[0])  # First part is the label
        # Extract the start time in seconds (second part of the filename)
        start_time = int(filename_parts[1])  # Second part is the start time in seconds
        # Extract the total duration (third part of the filename)
        duration = int(filename_parts[2])  # Third part is the total duration of the podcast
         # Extract the podcast name (four part of the filename)
        podcast_name = filename_parts[3].replace('.wav', '')  # Third part is the total duration of the podcast

        if run_env == "local":
            file_path = os.path.join(clip_audio_files, filename)
        elif run_env == 'gc':
            file_path = filename.open('rb')

        spectrogram = create_spectrogram(file_path)

        if spectrogram.shape == CORRECT_SPECTROGRAM_SHAPE:

            # Append the numpy array to the list
            spectrograms.append(spectrogram)
            labels.append(is_ad)
            seconds.append(start_time)
            durations.append(duration)
            podcast_names.append(podcast_name)

        else:
            print(f'{filename_parts} is not correct shape. Instead shape is {spectrogram.shape}')

    return spectrograms, labels, seconds, durations, podcast_names

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
