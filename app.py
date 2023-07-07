from dotenv import find_dotenv, load_dotenv
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st
import xml.etree.ElementTree as ET
from tts import mainSeq, get_SSML
import asyncio

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def img2text(filename):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    res_data = response.json()
    return res_data[0]['generated_text']


def generate_story(scenario):

    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 100 words?
    Answer in Chinese
    
    CONTEXT: {scenario}
    STORY: 
    """
    prompt = PromptTemplate(template=template, input_variables=['scenario'])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def text2xml(message):
    speak = ET.Element('speak')
    speak.set('xmlns', 'http://www.w3.org/2001/10/synthesis')
    speak.set('xmlns:mstts', 'http://www.w3.org/2001/mstts')
    speak.set('xmlns:emo', 'http://www.w3.org/2009/10/emotionml')
    speak.set('version', '1.0')
    speak.set('xml:lang', 'en-US')

    voice = ET.SubElement(speak, 'voice')
    voice.set('name', 'zh-CN-liaoning-XiaobeiNeural')

    prosody = ET.SubElement(voice, 'prosody')
    prosody.set('rate', '20%')
    prosody.set('pitch', '20%')
    prosody.text = message

    tree = ET.ElementTree(speak)
    tree.write('SSML.xml', encoding='utf-8')


async def tts():
    SSML_text = get_SSML('SSML.xml')
    await mainSeq(SSML_text, "audio")


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="🍰")
    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "png", "webp"])
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',
                 use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2xml(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        asyncio.run(tts())
        st.audio("audio.mp3")


if __name__ == '__main__':
    main()
