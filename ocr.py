import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

def blur_text_regions(image_path, text_boxes, blur_strength=(15, 15)):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or invalid path.")
    
    for box in text_boxes:
        coordinates = np.array(box['coordinates'], dtype=np.int32)
        
        # Create a mask for the text area
        x_min = min(pt[0] for pt in coordinates)
        y_min = min(pt[1] for pt in coordinates)
        x_max = max(pt[0] for pt in coordinates)
        y_max = max(pt[1] for pt in coordinates)
        
        # Extract the region to blur
        roi = image[y_min:y_max, x_min:x_max]
        
        # Apply blur
        averaged_roi = cv2.blur(roi, blur_strength, 0)
        blurred_roi = cv2.GaussianBlur(averaged_roi, blur_strength, 0)
        # Replace the original text area with the blurred one
        image[y_min:y_max, x_min:x_max] = blurred_roi
    
    return image

def draw_text(image, text, coordinates, font_path):
    x_min = int(min(pt[0] for pt in coordinates))  
    y_min = int(min(pt[1] for pt in coordinates))  
    x_max = int(max(pt[0] for pt in coordinates))  
    y_max = int(max(pt[1] for pt in coordinates))  
    
    box_width = x_max - x_min
    box_height = y_max - y_min
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_size = int(box_height * 0.8)
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.Layout.BASIC)

    box_region = np.array(img_pil)[y_min:y_max, x_min:x_max]
    #avg_color = np.mean(box_region, axis=(0, 1)).astype(int)
    #inverse_color = tuple(255 - avg_color)
    
    inverse_color = (0, 0, 0)
    

    draw.text((center_x, center_y), text, fill=inverse_color, font=font, anchor="mm")

    return np.array(img_pil)