import speech_recognition as sr
import pyttsx3
# from my_translators_here import GPT_translate_text
import time
# from gtts import gTTS
import os
from transformers import MarianMTModel, MarianTokenizer
import os
import openai
import sys
# import nltk
# from translate import Translator
# from google.cloud import translate
# from google.oauth2 import service_account  # type: ignore
# import googleapiclient.discovery  # type: ignore

from creds import OPENAI_API_KEY


def translate_text(text, source_lang, target_lang):
    # Map the language names to the corresponding language codes
    language_codes = {
    "French": "fr","Spanish": "es","Italian": "it","Romanian": "ro","Catalan": "ca",
    "German": "de","Dutch": "nl","Russian": "ru",
    "Chinese": "zh","Arabic": "ar","Swedish": "sv","Finnish": "fi","Greek": "el",
    "Hebrew": "he","Hindi": "hi","Indonesian": "id","Vietnamese": "vi","Bengali": "bn","English": "en"
}

    # Get the language code for the target language
    target_lang_code = language_codes[target_lang]
    source_lang_code = language_codes[source_lang]

    # Use the correct model for the target language
    model_name = f"Helsinki-NLP/opus-mt-{source_lang_code}-{target_lang_code}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Chunk the input text into segments of around 500 tokens each
    chunk_size = 500
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Translate each chunk and concatenate the translations
    translated_text_chunks = []
    for chunk in text_chunks:
        # Tokenize the chunk
        input_ids = tokenizer.encode(chunk, return_tensors="pt")

        # Translate the chunk
        translated_ids = model.generate(input_ids)

        # Decode the translated chunk
        translated_chunk = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        translated_text_chunks.append(translated_chunk)

    # Concatenate the translated chunks to form the final translated text
    translated_text = " ".join(translated_text_chunks)

    return translated_text


def GPT_translate_text(text, source_language, target_language):
    """
    Translate text from source language to target language using OpenAI API.
    Args:
        text (str): Text to be translated.
        source_language (str): Source language code.
        target_language (str): Target language code.
    Returns:
        str: Translated text.
    """
    # Constants
    MAX_TOKENS_PER_REQUEST = 4096  # Maximum tokens allowed per request
    CHUNK_SIZE = 3000  # Chunk size to ensure safe margin
    
    # Initialize translated text
    translated_text = ""
    
    # Set up the API request
    openai.api_key = OPENAI_API_KEY
    
    # Prompt for additional context
    additional_context = (
        "Additional context: Please translate this text with a formal tone suitable for "
        "professional communication. Avoid slang or colloquial expressions. Ensure accuracy "
        "and maintain the original meaning as closely as possible.\n\n"
        "Example translations:\n- Previous translations of similar texts.\n- Reference materials "
        "or glossaries for domain-specific terms.\n\n"
        "Clarifications:\n- If any terms are ambiguous, please provide context or "
        "clarifications to ensure accurate translation.\n"
    )
    
    # Check if the text exceeds maximum token limit
    if len(text) > MAX_TOKENS_PER_REQUEST:
        # Chunk the text into smaller segments
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        
        # Translate each chunk separately
        for chunk in chunks:
            # Construct prompt for translation
            prompt = f"Translate the following text from {source_language} to {target_language}:\n\n"\
                     f"Text to be translated:\n\"{chunk}\"\n\n{additional_context}"
            
            # Make API request
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7,
            )
            
            # Extract translated text from response
            translated_text += response.choices[0].text.strip() + " "
    else:
        # Construct prompt for translation
        prompt = f"Translate the following text from {source_language} to {target_language}:\n\n"\
                 f"Text to be translated:\n\"{text}\"\n\n{additional_context}"
        
        # Make API request
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        # Extract translated text from response
        translated_text = response.choices[0].text.strip()
    
    return translated_text



def record_and_transcribe():
    # Set the duration for recording in seconds
    duration = 10

    # Record audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        audio = r.listen(source, timeout=duration)

    # Transcribe the recorded audio
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand your speech.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def read_text(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Set the speech rate (optional)
    engine.setProperty('rate', 150)

    # Read the transcribed text
    engine.say(text)
    engine.runAndWait()

# def read_text(text):
#     # Create a gTTS object
#     tts = gTTS(text=text, lang='en')

#     # Play the audio file
#     tts.save('temp.mp3')
#     os.system('start temp.mp3')
    



if __name__ == "__main__":
    transcribed_text = record_and_transcribe()
    translated_text = GPT_translate_text(transcribed_text, "English", "French")
    read_text(translated_text)





# import pyttsx3

# def save_speech(text, filename):
#   """
#   This function reads the provided text using text-to-speech and saves it as an MP3 file.

#   Args:
#       text: The text string to be converted to speech.
#       filename: The desired filename (including path) for the saved MP3 audio.
#   """
  
#   # Initialize the text-to-speech engine
#   engine = pyttsx3.init()

#   # Set the speech rate (optional)
#   engine.setProperty('rate', 150)

#   # Read the transcribed text
#   engine.say(text)

#   # Save the speech to a file
#   engine.save_to_file(text, filename)
#   engine.runAndWait()

# # # Example usage
# text = "The monkey is in the house."
# filename = "speech.mp3"

# save_speech(text, filename)

# print(f"Speech saved as: {filename}")

# from gtts import gTTS

# def save_speech(text, filename):
#   """
#   This function reads the provided text using Google Text-to-Speech and saves it as an MP3 file.

#   Args:
#       text: The text string to be converted to speech.
#       filename: The desired filename (including path) for the saved MP3 audio.
#   """
  
#   # Create a Google Text-to-Speech object with the text and language (optional) 
#   tts = gTTS(text=text, lang='en')

#   # Save the audio data to a file
#   tts.save(filename)

# # Example usage
# # text = "This is an example text to convert to speech and save as MP3 using gTTS."
# # filename = "saved_speech_gTTS.mp3"

# save_speech(text, filename)

# print(f"Speech saved as: {filename}")