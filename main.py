import spellchecker
import ocr
from translate import Translator

def main():
    extracted_data=ocr.read_text(image_path)
    for item in extracted_data:
        words = [item['text'] for item in extracted_data]
    
    fixed_words = []
    for word in words:
        fixed_word = fix_spelling(word)
        fixed_words.append(fixed_word)

    translator = Translator(from_lang="ru", to_lang="ka")
    translated_words = [translator.translate(word) for word in fixed_words]

    for i in extracted_data:
        i['text'] = translated_words[words.index(i['text'])]

    blurred_image = blur_text_regions(image_path, extracted_data, blur_strength=(99, 99))

    for box in extracted_data:
        output_image = draw_text(blurred_image, box['text'], box['coordinates'])

    return output_image

