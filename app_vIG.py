import streamlit as st
from utils import *
import requests
import sys




st.markdown("""
    <style>
    .stAudio {
        margin: 20px 0;
    }
    #Podcast Ad Skipper {
        text-align: center
    }
    </style>
""")

st.title("Podcast Ad Skipper")

st.title("Pick a podcast to play")

# Allow user to upload audio files
audio_file_before_ads = st.file_uploader("Choose an audio file:", type=["mp3"])

# Display audio player
st.audio(audio_file_before_ads)

# Create a button that will trigger the API call
st.button("Remove the ads!")



if st.button("Remove the ads!"):
# Make the API call to get predictions

    clip_duration=5

    podcast_duration = len(audio_file_before_ads) / 1000  # Duration in seconds

    # List to hold the ad segments
    ad_segments = []

    # Process the podcast in chunks of clip_duration seconds
    for i in range(0, int(podcast_duration), clip_duration=5):
        start_time = i * 1000  # Convert to milliseconds
        end_time = (i + clip_duration) * 1000

        # Extract the clip from the podcast
        clip = audio_file_before_ads[start_time:end_time]

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


        try:
            # To update URL and save eit as an env variable
            api_url = "http://localhost:8000/predict"
            response = requests.post(api_url, json={"spectrogram": spectrogram.tolist()})

            params={"spectrogram":spectrogram_np}

            response = requests.get(api_url, params=params)
            prediction = response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            sys.exit(1)

        # If the model predicts 'ad' it will mark this segment as an ad (1)
        if prediction == 1:
            ad_segments.append((i, i + clip_duration))

        # Clean up the temporary file
        os.remove(clip_file)

    clean_podcast = remove_ads_from_podcast(audio_file_before_ads, ad_segments)
    st.audio(clean_podcast)



# # Sample audio URL - replace with your actual audio file
# audio_url_before_ads = "https://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/audio-nondrm-download/proto/https/vpid/p0bw7py0.mp3"
