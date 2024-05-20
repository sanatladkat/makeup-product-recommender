import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

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
    st.title("Skin Tone Detection App")
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
            col2.write(f"Predicted Skin Tone: {class_names[prediction]}")

if __name__ == '__main__':
    main()
