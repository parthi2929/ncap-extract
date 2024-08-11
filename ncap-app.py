import streamlit as st
import os
import zipfile
from PIL import Image
import numpy as np
import cv2
import io

def process_image_for_ratings(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    height, width = img.shape[:2]
    
    # Define the coordinates for the adult and child rating areas
    adult_crop_box = (70, height - 100, 370, height)
    child_crop_box = (width - 320, height - 100, width, height)
    
    # Crop the images
    adult_rating_img = img[adult_crop_box[1]:adult_crop_box[3], adult_crop_box[0]:adult_crop_box[2]]
    child_rating_img = img[child_crop_box[1]:child_crop_box[3], child_crop_box[0]:child_crop_box[2]]
    
    # Function to count yellow stars
    def count_yellow_stars(img):
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for yellow color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        
        # Create a mask for yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count yellow stars (filter out small contours)
        yellow_stars = sum(1 for cnt in contours if cv2.contourArea(cnt) > 100)
        
        return yellow_stars
    
    adult_stars = count_yellow_stars(adult_rating_img)
    child_stars = count_yellow_stars(child_rating_img)
    
    return adult_stars, child_stars

def main():
    st.title("Car Safety Rating Extractor")
    
    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    
    if uploaded_file is not None:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            results = []
            
            for filename in zip_ref.namelist():
                if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                    with zip_ref.open(filename) as file:
                        image_bytes = file.read()
                        
                    car_model = os.path.splitext(os.path.basename(filename))[0].replace("-", " ").strip()
                    adult_rating, child_rating = process_image_for_ratings(image_bytes)
                    
                    results.append({
                        "Car Model": car_model,
                        "Adult Safety Rating": adult_rating,
                        "Child Safety Rating": child_rating
                    })
                    
        # Create HTML table
        html_table = f"<table><tr><th>Car Model</th><th>Adult Safety Rating</th><th>Child Safety Rating</th></tr>"
        for result in results:
            html_table += f"<tr><td>{result['Car Model']}</td><td>{result['Adult Safety Rating']} stars</td><td>{result['Child Safety Rating']} stars</td></tr>"
        html_table += "</table>"
        
        st.markdown(html_table, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
