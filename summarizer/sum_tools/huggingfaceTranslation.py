from transformers import MarianMTModel, MarianTokenizer

def translate_text(text):
    model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translation = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

    return translated_text[0]


# translated_text = translate_text("Va chercher du pain.")
# print(translated_text)