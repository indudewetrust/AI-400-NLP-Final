import  streamlit as st
from sum_tools.scrapper import scrape_paragraphs
from sum_tools.summarizer import generate_summary, generate_abstractive_summary, sent_analysis
from sum_tools.huggingfaceTranslation import translate_text

# def main():

#     st.title('''
#     Project NLP Summarizer and Sentiment Analysis
#     ''')

#     st.write('''
#     Insert a web URL into the text box and hit enter. 
#     The textbot will give you a summary of the text and analyze the sentiment as generally negative or positive.
#     It will also translate five languages to an English summary.
#     ''')



#     with st.form("url"):
#         url_input = st.text_input("URL: ", key="url")

#         submitted = st.form_submit_button("Submit")
#         if submitted:
#             paragraphs, headline = scrape_paragraphs(url_input)
#             text = "\n".join(paragraphs)  # Combine paragraphs into a single text
#             st.header(headline)
#             sent = sent_analysis(text)
#             st.write("Original Text Sentiment Analysis", sent)
#             st.write("Extractive Summary:")
#             summary = generate_summary(text)
#             st.write(summary)
#             summary_sent = sent_analysis(summary)
#             st.write("Extractive Summary Sentiment Analysis", summary_sent)

#             st.write("Abstractive Summary:")
#             abs_summary = generate_abstractive_summary(text)
#             st.write(abs_summary)
#             abs_sent = sent_analysis(abs_summary)
#             st.write("Abstractive Summary Sentiment Analysis", abs_sent)

def main():

    st.title('''
    Project NLP Summarizer and Sentiment Analysis
    ''')

    st.write('''
    Insert a web URL into the text box and hit enter. 
    The textbot will give you a summary of the text and analyze the sentiment as generally negative or positive.
    It will also translate five languages to an English summary.
    ''')

    # Tab selection
    selected_tab = st.sidebar.selectbox("Select a task", ["Original Text", "Extractive Summary", "Abstractive Summary", "Translate French to English"])

    with st.form("url"):
        url_input = st.text_input("URL: ", key="url")
        submitted = st.form_submit_button("Submit")

        if submitted:
            paragraphs, headline = scrape_paragraphs(url_input)
            text = "\n".join(paragraphs)  # Combine paragraphs into a single text

            if selected_tab == "Original Text":
                st.header(headline)
                sent = sent_analysis(text)
                st.write("Original Text Sentiment Analysis", sent)

            elif selected_tab == "Extractive Summary":
                st.write("Extractive Summary:")
                summary = generate_summary(text)
                st.write(summary)
                summary_sent = sent_analysis(summary)
                st.write("Extractive Summary Sentiment Analysis", summary_sent)

            elif selected_tab == "Abstractive Summary":
                st.write("Abstractive Summary:")
                abs_summary = generate_abstractive_summary(text)
                st.write(abs_summary)
                abs_sent = sent_analysis(abs_summary)
                st.write("Abstractive Summary Sentiment Analysis", abs_sent)

            elif selected_tab == "Translate French to English":
                # st.write("Original Text:")
                # st.write(text)
                
                st.write("Romance languages to English Translation:")
                translation = translate_text(text)
                st.write(translation)

                st.write("Romance languages to English Summary:")
                summary = generate_abstractive_summary(translation)
                st.write(summary)

                st.write("Romance languages to English Summary:")
                asummary = generate_summary(translation)
                st.write(asummary)


                translation_sent = sent_analysis(translation)
                st.write("Translated Text Sentiment Analysis", translation_sent)



if __name__=="__main__":
    main()