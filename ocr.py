import easyocr

def read_text(image_path):
    reader = easyocr.Reader(['ru'])  
    results = reader.readtext(image_path)

    extracted_data = []
    for (bbox, text, confidence) in results:
        extracted_data.append({
            "text": text,
            "coordinates": bbox,
            "confidence": confidence
        })

    return extracted_data
