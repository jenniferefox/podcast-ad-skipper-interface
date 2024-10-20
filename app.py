import streamlit as st

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
    st.button("Pick Fish")
with col2:
    st.image("images/vergecast.jpg", caption="The Vergecast", width=200)
    st.button("Pick Verge")
with col3:
    st.image("images/parentinghell.jpeg", caption="Parenting Hell", width=200)
    st.button("Pick Parenting")


# Sample audio URL - replace with your actual audio file
audio_url_before_ads = "https://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/audio-nondrm-download/proto/https/vpid/p0bw7py0.mp3"

# Display audio player
st.audio(audio_url_before_ads)

# Create a button that will trigger the API call
st.button("Remove the ads!")

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
