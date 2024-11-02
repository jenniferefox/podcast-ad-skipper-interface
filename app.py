import streamlit as st
from utils.utils import get_features_model, split_files
import os

st.markdown("""
    <style>
    .stAudio {
        margin: 20px 0;
    }
    #Podcast Ad Skipper {
        text-align: center
    }
    </style>
""", unsafe_allow_html=True)

'''
# Podcast Ad Skipper
'''

st.markdown('''
## Pick a podcast to play
''')

col1, col2, col3 = st.columns(3)

with col1:
    st.image("images/NSTAAF.png", caption="No Such Thing As a Fish", width=200)
    if st.button("Pick Halloween"):
        info = [os.path.join('.raw_data/', "spirithalloween.mp3"), [0,30, ((22*60)+51), ((23*60)+53),((33*60)+49), ((34*60)+50)], "spirit_halloween"]
        split_files(info[0], info[1], info[2], info[3])
        for file in get_features_model()

# with col2:
#     st.image("images/vergecast.jpg", caption="The Vergecast", width=200)
#     if st.button("Pick World"):

# with col3:
#     st.image("images/parentinghell.jpeg", caption="Parenting Hell", width=200)
#     if st.button("Pick Parenting"):



# Sample audio URL - replace with your actual audio file
audio_url_before_ads = "https://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/audio-nondrm-download/proto/https/vpid/p0bw7py0.mp3"

# Display audio player
st.audio(audio_url_before_ads)

# Create a button that will trigger the API call
st.button("Remove the ads!")

#prep local vs cloud options!

    #split into 5 sec clips
    #transform for predict input
    #run through model.predict
    #output list of where ads are



# Sample audio URL - replace with your actual audio file
audio_url_after_ads = "https://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/audio-nondrm-download/proto/https/vpid/p0bw7py0.mp3"

# Display audio player
st.audio(audio_url_after_ads)




# if st.button("Make API Call"):
    # try:
        # Replace with your actual API endpoint
        # api_url = ""

        # Make the API call
        # response = requests.get(api_url)

        # Check if the request was successful
        # if response.status_code == 200:
        #     # Parse the JSON response
        #     data = response.json()

        #     # Display the API response
        #     st.success("API call successful!")
        #     st.json(data)
        # else:
        #     st.error(f"API call failed with status code: {response.status_code}")

    # except Exception as e:
    #     st.error(f"An error occurred: {str(e)}")
