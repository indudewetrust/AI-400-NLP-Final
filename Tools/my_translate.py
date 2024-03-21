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

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Translate the input text
    translated_ids = model.generate(input_ids)

    # Decode the translated text
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

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



if __name__ == "__main__":
    text_to_translate = "Hello, how are you?"
    source_language = "English"
    target_language = "French"
    translated_text = translate_text(text_to_translate, source_language, target_language)
    print("Translated text:", translated_text)
    translated_text = GPT_translate_text(text_to_translate, source_language, target_language)
    print("Translated text:", translated_text)



# Example usage
# print(translate_text("Hello, how are you?", "English", "Japanese")) # Output: "Hola, ¿cómo estás?"


#  NOT GOOD FOR SPEECH TRANSLATION. DO NOT LIKE!!!
# use NLTK for translation
# def translate_with_nltk(text, target_lang):
#     translator = Translator(to_lang=target_lang)
#     translated_text = translator.translate(text)
#     return translated_text

# # Example usage
# print(translate_with_nltk("Hello, how are you?", "English", "Japanese")) # Output: "こんにちは、お元気ですか？"

# # use Google AutoML for translation
# def automl_translate(text):
#     client = translate.TranslationServiceClient()

#     # project_id = '260122611251'
#     project_id="boxwood-ellipse-417405"
#     # text = 'YOUR_SOURCE_CONTENT'
#     location = 'us-central1'
#     model = f'projects/{project_id}/locations/us-central1/models/NMb2b54cc566c94c0a'

#     parent = client.location_path(project_id, location)

#     response = client.translate_text(
#         parent=parent,
#         contents=[text],
#         model=model,
#         mime_type='text/plain',  # mime types: text/plain, text/html
#         source_language_code='en',
#         target_language_code='bn')

#     for translation in response.translations:
#         print('Translated Text: {}'.format(str(translation).encode('utf8')))

# Example usage
# automl_translate("Hello, I love you!")