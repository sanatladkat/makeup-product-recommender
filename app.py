import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load skin tone recommendation data
data = {
    "Skin Tone": ["light", "mid_light", "mid_dark", "dark"],
    "Lipstick": ["Nude Beach, Peachy Keen, Vanilla Cream", "Coral Crush, Peach Fuzz, Cinnamon Spice", "Mocha Madness, Toasted Almond, Cocoa Butter", "Plum Royale, Rich Raisin, Deep Burgundy"],
    "Compact": ["Fair Porcelain, Light Ivory, Alabaster Glow", "Natural Buff, Buff Beige, Medium Light", "Caramel, Medium Tan, Warm Beige", "Cocoa, Deep Bronze, Dark Walnut"],
    "Blush": ["Soft Rose, Petal Pink, Cotton Candy", "Peachy Glow, Warm Apricot, Coralista", "Terracotta, Spiced Coral, Raspberry Rush", "Berry Bliss, Rich Spice, Mahogany"],
    "Eyeliner": ["Champagne Shimmer, Pearl, Golden Bronze", "Amber Glow, Toffee, Bronze Beam", "Copper Glow, Mahogany, Espresso", "Black Opal, Onyx, Ebony"]
}

df = pd.DataFrame(data)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(
    df["Lipstick"] + ", " + df["Compact"] + ", " + df["Blush"] + ", " + df["Eyeliner"])

# Compute similarity scores
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend products based on skin tone


def recommend_products(skin_tone, cosine_sim=cosine_sim):
    idx = df[df["Skin Tone"] == skin_tone].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:2]
    product_indices = [i[0] for i in sim_scores]
    return df["Lipstick"][product_indices], df["Compact"][product_indices], df["Blush"][product_indices], df["Eyeliner"][product_indices]


# Define class names
class_names = ['dark', 'light', 'mid_dark', 'mid_light']

# Load the model
model = models.mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.7),
    nn.Linear(num_ftrs, 50),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(50, 4)
)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to predict skin tone


def predict_skin_tone(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Streamlit app


def main():
    st.title("Product Recommendation App")
    st.sidebar.title("Upload & Detect")

    # Sidebar upload button
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    # Create two columns layout
    col1, col2 = st.columns(2)

    # Open and display the uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        col1.write("Please upload an image.")

    if uploaded_file is not None:
        # Predict skin tone when button clicked
        if st.sidebar.button('Detect Skin Tone'):
            # Predict skin tone
            prediction = predict_skin_tone(image)
            # Display predicted skin tone output in second column
            col1.write(f"Predicted Skin Tone: {class_names[prediction]}")

            with col2:
                st.subheader("Recommended Shades")

            # Recommend products
            lipsticks, compacts, blushes, eyeliners = recommend_products(
                class_names[prediction])
            col2.write("Lipstick :")
            for lipstick in lipsticks:
                col2.write("- " + lipstick)
            col2.write("Compact :")
            for compact in compacts:
                col2.write("- " + compact)
            col2.write("Blush :")
            for blush in blushes:
                col2.write("- " + blush)
            col2.write("Eyeliner :")
            for eyeliner in eyeliners:
                col2.write("- " + eyeliner)


if __name__ == '__main__':
    main()
