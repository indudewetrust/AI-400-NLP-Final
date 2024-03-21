import streamlit as st
# from Tools.my_translators_here import translate_text, GPT_translate_text
from Tools.analysis import sentiment_analysis, compute_bleu
from Tools.speech import record_and_transcribe, read_text, translate_text, GPT_translate_text
from Tools.scrapper import scrape_paragraphs
import time

language_codes = {"English": "en",
    "French": "fr","Spanish": "es","Italian": "it","Romanian": "ro","Catalan": "ca",
    "German": "de","Dutch": "nl","Russian": "ru",
    "Chinese": "zh","Arabic": "ar","Swedish": "sv","Finnish": "fi","Greek": "el",
    "Hebrew": "he","Hindi": "hi","Indonesian": "id","Vietnamese": "vi","Bengali": "bn",
}


def main():
    st.title('LinguaWave Translate')
    st.write("""Welcome to LinguaWave Translate! 
             This is a simple web app that allows you to perform various natural language processing tasks. 
             You can use this app to translate text, convert text to speech, convert speech to text, and chat with a chatbot. 
             To get started, select a task from the sidebar.""")

    # Tab selection
    seletcted_tab = st.sidebar.selectbox("Select a task", ["Report", "MarianMT Text Translation", "Google AutoML Text Translation", "ChatGPT Text Translation","Speech Translation", "Webpage Translation"])

    if seletcted_tab == "MarianMT Text Translation":
        translated_text = ""
        st.subheader("Text Translation")
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox("Select the source language", list(language_codes.keys()), key="source_lang")
        with col2:
            dest_lang = st.selectbox("Select the destination language", list(language_codes.keys()), key="dest_lang")
        col3, col4 = st.columns(2)
        with col3:
            text = st.text_area("Enter the text to translate", height=200)
        
            translate_button = st.button("Translate")
        with col4:
            if translate_button:
                if text:
                    # st.subheader("Translated Text")
                    with st.spinner("Translating..."):
                        translated_text = translate_text(text, source_lang, dest_lang)
                        st.text_area(label="Translated Text",value=translated_text, height=200)
        col5, col6 = st.columns(2)
        with col5:
            if text:
                st.write(f"Original Text Sentiment Analysis: {sentiment_analysis(text)}")
                st.write(f"Translated Text Sentiment Analysis: {sentiment_analysis(translated_text)}")
        with col6:       
            if text:
                score = compute_bleu(text, translated_text)
                st.write("BLEU score:", score)

    elif seletcted_tab == "Google AutoML Text Translation":
        st.subheader("Google AutoML Text Translation")
        st.write("This feature is currently under development. Please check back later.")
        # translated_text = ""
        # st.subheader("Google AutoML Text Translation")
        # col1, col2 = st.columns(2)
        # with col1:
        #     source_lang = st.selectbox("Select the source language", list(language_codes.keys()), key="source_lang_nltk")
        # with col2:
        #     dest_lang = st.selectbox("Select the destination language", list(language_codes.keys()), key="dest_lang_nltk")
        # col3, col4 = st.columns(2)
        # with col3:
        #     text = st.text_area("Enter the text to translate using NLTK", height=200)
        
        #     translate_button = st.button("Translate using NLTK")
        # with col4:
        #     if translate_button:
        #         if text:
        #             # st.subheader("Translated Text")
        #             with st.spinner("Translating using NLTK..."):
        #                 translated_text = automl_translate(text, dest_lang)
        #                 st.text_area(label="Translated Text (NLTK)",value=translated_text, height=200)
        # col5, col6 = st.columns(2)
        # with col5:
        #     if text:
        #         st.write(f"Original Text Sentiment Analysis: {sentiment_analysis(text)}")
        #         st.write(f"Translated Text Sentiment Analysis: {sentiment_analysis(translated_text)}")
        # with col6:       
        #     if text:
        #         score = compute_bleu(text, translated_text)
        #         st.write("BLEU score:", score)
                
    elif seletcted_tab == "ChatGPT Text Translation":
        translated_text = ""
        st.subheader("ChatGPT Text Translation")
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox("Select the source language", list(language_codes.keys()), key="source_lang")
        with col2:
            dest_lang = st.selectbox("Select the destination language", list(language_codes.keys()), key="dest_lang")
        col3, col4 = st.columns(2)
        with col3:
            text = st.text_area("Enter the text to translate", height=200)
        
            translate_button = st.button("Translate")
        with col4:
            if translate_button:
                if text:
                    # st.subheader("Translated Text")
                    with st.spinner("Translating..."):
                        translated_text = GPT_translate_text(text, source_lang, dest_lang)
                        st.text_area(label="Translated Text",value=translated_text, height=200)
        col5, col6 = st.columns(2)
        with col5:
            if text:
                st.write(f"Original Text Sentiment Analysis: {sentiment_analysis(text)}")
                st.write(f"Translated Text Sentiment Analysis: {sentiment_analysis(translated_text)}")
        with col6:       
            if text:
                score1 = compute_bleu(text, translated_text)
                st.write("BLEU score:", score1)

    elif seletcted_tab == "Speech Translation":
        st.subheader("Speech Translation")
        st.write("Click the button below and speak into the microphone to translate your speech to text.")
        col7, col8 = st.columns(2)
        with col7:
            rec_lang = st.selectbox("Select the source language", list(language_codes.keys()), key="rec_lang")
        with col8:
            speak_lang = st.selectbox("Select the destination language", list(language_codes.keys()), key="speak_lang")
        speech_to_text_button = st.button("Start Recording")
        text = ""  # Initialize text to an empty string
        if speech_to_text_button:
            st.write("Recording starting in...")
            countdown_seconds = 3  # Set the countdown time in seconds
            countdown_text = st.empty()  # Placeholder to display the countdown
            for i in range(countdown_seconds, 0, -1):
                countdown_text.text(f"Recording... {i} seconds left")
                time.sleep(1)  # Wait for 1 second
            countdown_text.text("Recording... 0 seconds left")
            text = record_and_transcribe()
            st.write(f"Speech to Text: {text}")
                    
            translated_text = GPT_translate_text(text, rec_lang, speak_lang)
            st.write(f"Translated Text: {translated_text}")
            read_text(translated_text)
            st.write("Recording stopped.")

            col9, col10 = st.columns(2)
            with col9:
                if text:
                    st.write(f"Original Text Sentiment Analysis: {sentiment_analysis(text)}")
                    st.write(f"Translated Text Sentiment Analysis: {sentiment_analysis(translated_text)}")
            with col10:       
                score3 = compute_bleu(text, translated_text)
                st.write("BLEU score:", score3)
            
    elif seletcted_tab == "Webpage Translation":
        st.subheader("Webpage Translation")
        col12, col13 = st.columns(2)
        with col12:
            source_lang = st.selectbox("Select the source language", list(language_codes.keys()), key="source_lang")
        with col13:
            dest_lang = st.selectbox("Select the destination language", list(language_codes.keys()), key="dest_lang")
        url = st.text_input("Enter the URL of the webpage to translate")
        translate_button = st.button("Translate")

        if translate_button:
            if url:
                with st.spinner("Scraping webpage..."):
                    article_paragraphs, headline = scrape_paragraphs(url)
                    article_text = ' '.join(article_paragraphs)
                    st.text_area(label="Webpage Text:", value=headline+"\n\n"+article_text)
                with st.spinner("Translating..."):
                    translated_headline = translate_text(headline, source_lang, dest_lang)
                    translated_text = translate_text(article_text, source_lang, dest_lang)
                    st.text_area(label="Translated Text:", value=translated_headline+"\n\n"+translated_text, height=500)
                if article_text:
                    st.write(f"Original Text Sentiment Analysis: {sentiment_analysis(article_text)}")
                    st.write(f"Translated Text Sentiment Analysis: {sentiment_analysis(translated_text)}")
                if article_text:
                    score4 = compute_bleu(article_text, translated_text)
                    st.write("BLEU score:", score4)


    if seletcted_tab == "Report":
        with open("report.md", "r") as file:
            report = file.read()
        st.markdown(report)


if __name__ == '__main__':
    main()