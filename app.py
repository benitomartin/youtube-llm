import streamlit as st
from pytube import YouTube
import moviepy.editor
import whisper
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import os
import torch
from transformers import pipeline  # Import the summarization pipeline
from dotenv import load_dotenv

# Load environmental variables from a .env file
load_dotenv()

# Get the OpenAI API key from environmental variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the title of the Streamlit app
st.title("YouTube Q&A and Summarization App")
st.markdown("""
    #### Description
    Upload a YouTube video and get a Q&A Assistant or a Summary. Select an option on the left side.
    """)

# Get the YouTube URL from the user
url = st.text_input("Add the URL of the YouTube Video:")

# Create or retrieve session state variables
state = st.session_state
if "model" not in state:
    state.model = None
if "url" not in state:
    state.url = ""

# Sidebar
st.sidebar.subheader("About the App")
st.sidebar.info("This App has been developed using Whisper and Langchain (LLM). For the Q&A it uses gpt-3.5-turbo (4,097 tokens)")
st.sidebar.subheader("Select an Option")
selected_option = st.sidebar.selectbox("PLease select an option", ["Q&A", "Summarize"])


# Get the directory of the script (your app.py)
script_directory = os.path.dirname(__file__)

# Specify the folder for audio files within the script's directory
audio_folder = os.path.join(script_directory, "audio_output")

# Detect the appropriate device for Whisper
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base Whisper model
model = whisper.load_model("base", device=DEVICE)


# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")


if selected_option == "Q&A":

    # Button to initiate audio extraction and transcription
    if st.button("Extract Audio and Transcribe"):
        if url:
            try:
                # Create a YouTube object for the given URL
                yt = YouTube(url)

                # Define file names for the video and audio files
                video_extension = "mp4"
                audio_extension = "mp3"
                video_filename = f"video.{video_extension}"
                audio_filename = os.path.join(audio_folder, f"audio.{audio_extension}")

                # Create the audio folder if it doesn't exist
                os.makedirs(audio_folder, exist_ok=True)

                # Download the highest resolution video in MP4 format
                yt.streams.filter(progressive=True, file_extension=video_extension).order_by('resolution').desc().first().download(filename=video_filename)

                # Load the downloaded video using MoviePy
                video = moviepy.editor.VideoFileClip(filename=video_filename)

                # Extract Audio if Audio is Available
                try:
                    # Extract the audio from the video
                    audio = video.audio

                    # Export the audio to MP3 format in the specified folder
                    audio.write_audiofile(filename=audio_filename)

                    # Get the absolute path of the audio file
                    audio_filepath = os.path.normpath(os.path.abspath(audio_filename))

                    st.success("Audio extraction complete!")
                    

                    my_bar = st.progress(0, text="Getting Transcription from Audio...")

                    # Transcribe the audio using Whisper
                    if os.path.exists(audio_filepath):
                        result = model.transcribe(audio_filepath, word_timestamps=False)

                        # Set your OpenAI API key
                        OPENAI_API_KEY = OPENAI_API_KEY

                        # Initialize the OpenAIEmbeddings
                        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

                        # Initialize the Chroma vector store with source data
                        vStore = Chroma.from_texts([result["text"]], embeddings, metadatas=[{"source": "Video Transcription"}])

                        # Model name
                        model_name = "gpt-3.5-turbo"

                        # Create a retriever from the vector store
                        retriever = vStore.as_retriever()
                        retriever.search_kwargs = {'k': 2}

                        # Initialize the ChatOpenAI model
                        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name)

                        # Create a RetrievalQAWithSourcesChain
                        model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

                        my_bar.progress(100, text="The Q&A is Ready!")
                        
                        # Save the model and URL in the session state
                        state.model = model
                        state.url = url
                    
                    else:
                        st.error("Audio file not found. Please check the audio extraction process.")

                except Exception as audio_error:
                    st.error(f"Error extracting audio: {audio_error}")

            except Exception as error:
                st.error(f"Error downloading video: {error}")

        else:
            st.warning("Please enter a YouTube video URL.")

    # Check if a model and URL are available in the session state
    if state.model and state.url:
        question = st.text_input("Ask a question about the video:")

        if question:
            result = state.model({"question": question}, return_only_outputs=True)

            # Display the response
            st.subheader("Answer:")
            st.write(result["answer"])

elif selected_option == "Summarize":

# Button to initiate audio extraction and transcription
    if st.button("Summarize Video"):
        if url:
            try:
                # Create a YouTube object for the given URL
                yt = YouTube(url)

                # Define file names for the video and audio files
                video_extension = "mp4"
                audio_extension = "mp3"
                video_filename = f"video.{video_extension}"
                audio_filename = os.path.join(audio_folder, f"audio.{audio_extension}")

                # Create the audio folder if it doesn't exist
                os.makedirs(audio_folder, exist_ok=True)

                # Download the highest resolution video in MP4 format
                yt.streams.filter(progressive=True, file_extension=video_extension).order_by('resolution').desc().first().download(filename=video_filename)

                # Load the downloaded video using MoviePy
                video = moviepy.editor.VideoFileClip(filename=video_filename)

                # Extract Audio if Audio is Available
                try:
                    # Extract the audio from the video
                    audio = video.audio

                    # Export the audio to MP3 format in the specified folder
                    audio.write_audiofile(filename=audio_filename)

                    # Get the absolute path of the audio file
                    audio_filepath = os.path.normpath(os.path.abspath(audio_filename))

                    st.success("Audio extraction complete!")
                    
                    my_bar = st.progress(0, text="Getting Transcription from Audio...")

                        
                    if os.path.exists(audio_filepath):
                        result = model.transcribe(audio_filepath, word_timestamps=False)
                        
                        # Extract the transcribed text
                        transcribed_text = result["text"]
                        
                        # Generate a summary
                        summary = summarizer(transcribed_text, max_length=150, min_length=30, do_sample=False)
                        
                        st.success("Text extraction and summarization complete!")
                        st.subheader("Summary:")
                        st.write(summary[0]["summary_text"])

                    else:
                        st.error("Audio file not found. Please check the audio extraction process.")

                except Exception as audio_error:
                    st.error(f"Error extracting audio: {audio_error}")

            except Exception as error:
                st.error(f"Error downloading video: {error}")

