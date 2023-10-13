# YOUTUBE Q&A AND SUMMARIZATION APP 📺

<p align="center">
    <img src="https://cdn.pixabay.com/photo/2016/05/30/14/23/detective-1424831_1280.png" width="500" height="400"/>
</p>

This repository hosts an app developed using **Whisper** and **Langchain** that allows to create a Q&A assistant and video summarization. The model's maximum context length is 4097 tokens (gpt-3.5-turbo).

The App can be run locally but requires an `OPENAI_API_KEY` in the `.env` file

## 👨‍💻 **Tech Stack**


![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## 💬 Set Up

 I recommend to install the modules in the following order. The `ffmpeg` module is required for the proper functioning of the application. You can install it using Conda as follows:

```bash
conda install -c conda-forge ffmpeg
```

```bash
pip install git+https://github.com/openai/whisper.git
```

```bash
pip install -r requirements.txt
```

## 🫵 App Deployment

The up can be used running `streamlit run app.py` in the terminal. There are 2 options on the side bar, Q&A or Summarize. I recommend using videos no longer than 5 min of speech due to the model tokens' limitations.

The first option allows a Q&A assistant to ask questions about the video.

<p align="center">
    <img src="images/qa.png" />
</p>

The second option allows us to get a summary of the video.

<p align="center">
    <img src="images/summary.png" />
</p>

