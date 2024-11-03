import streamlit as st
from utils import *
import requests
import sys
from pydub import AudioSegment


# st.markdown("""
#     <style>
#     .stAudio {
#         margin: 20px 0;
#     }
#     #Podcast Ad Skipper {
#         text-align: center
#     }
#     </style>
# """)

st.title("Podcast Ad Skipper")

st.title("Play the podcast")

st.image("images/NSTAAF.png", caption="No Such Thing As a Fish", width=200)
if st.button("Pick Fish"):
    file_path_audio_before_ads = "downloaded_podcasts/FridayNightComedyFromBBCRadio4-20220318.mp3"
    st.audio(file_path_audio_before_ads)


# Valid if we decide to import the podcast from a local folder

# # Specify the path to your audio file
# audio_file_path = "path/to/your/audio_file.mp3"

# # Load audio file using pydub
# audio_file_before_ads = AudioSegment.from_file(audio_file_path, format="mp3")



# Display audio player


# # Create a button that will trigger the API call
# st.button("Remove the ads!")


if st.button("Remove the ads!"):
# Make the API call to get predictions
    file_path_audio_before_ads = "downloaded_podcasts/FridayNightComedyFromBBCRadio4-20220318.mp3"

    audio_file_before_ads = AudioSegment.from_file(file_path_audio_before_ads, format="mp3")

    clip_duration=5

    podcast_duration = len(audio_file_before_ads) / 1000  # Duration in seconds

    # List to hold the ad segments
    ad_segments = []

    # Process the podcast in chunks of clip_duration seconds
    for i in range(0, int(podcast_duration), clip_duration):
        start_time = i * 1000  # Convert to milliseconds
        end_time = (i + clip_duration) * 1000

        # Extract the clip from the podcast
        clip = audio_file_before_ads[start_time:end_time]

        # Save the clip as a temporary wav file (for librosa to process)
        clip_file = "temp_clip.wav"
        clip.export(clip_file, format="wav")

        # Create a spectrogram for the clip
        spectrogram = create_spectrogram(clip_file) # We already have this function

        # Convert the spectrogram to a numpy array and pass it to the model
        spectrogram_np = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

        model = models.load_model('model_folder/ad_detection_model.h5')
        prediction = model.predict(spectrogram_np)

        # If the model predicts 'ad' it will mark this segment as an ad (1)
        if prediction == 1:
            ad_segments.append((i, i + clip_duration))

        # Clean up the temporary file
        os.remove(clip_file)

    clean_podcast = remove_ads_from_podcast(file_path_audio_before_ads, ad_segments)

    output_file_path = "clean_podcast_file_path.mp3"  # Update with your desired path

    clean_podcast.export(output_file_path, format="mp3")

    clean_podcast_mp3_file_path ="clean_podcast_file_path.mp3"

    st.audio(clean_podcast_mp3_file_path)



# # Sample audio URL - replace with your actual audio file
# audio_url_before_ads = "https://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/audio-nondrm-download/proto/https/vpid/p0bw7py0.mp3"
