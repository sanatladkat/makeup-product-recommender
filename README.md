# Skin Tone Detection & Product Shade Recommendation

This project utilizes a pre-trained MobileNetV2 model to predict the skin tone of an uploaded image. The model has been fine-tuned on a custom dataset to classify skin tones into four categories: dark, light, mid_dark, and mid_light. After detecting the skin tone, the app recommends suitable lipstick shades, compact shades, blush shades, and eyeliner shades.

![Alt Text](Prod_UI.gif)

## Project Structure

- **skindata20 folder**: Contains images used for training the model.
- **test_folder**: Contains images for testing the model.
- **app.py**: Streamlit UI for the application.
- **trained_model.pth**: Fine-tuned model on the custom dataset.
- **skin_prod_shade**: Jupyter notebook for model development.

## Training the Skin Tone Detection Model

The model is trained using PyTorch, with the MobileNetV2 architecture. The code for training the model is included in the `skin_prod_shade` notebook. The training data is divided into training and validation sets, and the model is trained to classify images into one of the four skin tone categories.

## Project Overview

This project consists of two main components:

1. **Skin Tone Detection:** 
   - Utilizes a pre-trained MobileNetV2 model to predict the skin tone of an uploaded image.
   - The model has been fine-tuned on a custom dataset to classify skin tones into four categories: dark, light, mid_dark, and mid_light.

2. **Product Recommendation:**
   - Recommends makeup products based on the detected skin tone.
   - Uses TF-IDF vectorization and cosine similarity to compute product similarities.
   - Retrieves makeup product recommendations from a dataset based on the detected skin tone.

## How to Use the App

1. **Upload an Image:**
   - Use the sidebar file uploader to upload an image containing a face.
   - Supported image formats: JPG, JPEG, PNG.

2. **Detect Skin Tone:**
   - Click the "Detect Skin Tone" button to predict the skin tone of the uploaded image.
   - The app will display the predicted skin tone in the second column.

3. **View Product Recommendations:**
   - After detecting the skin tone, the app will recommend suitable lipstick shades, compact shades, blush shades, and eyeliner shades in the second column.
   - Each recommended product will be displayed separately.

## Requirements

- Python 3.x
- Streamlit
- PyTorch
- torchvision
- scikit-learn
- pandas
- Pillow

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sanatladkat/makeup-product-recommender.git
   ```

2. Navigate to the project directory:

   ```bash
   cd makeup-product-recommender
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser. Upload an image and detect the skin tone to receive product recommendations.
