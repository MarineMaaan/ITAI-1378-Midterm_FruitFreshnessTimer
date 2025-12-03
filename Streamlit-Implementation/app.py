import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import math
import os
import re

st.set_page_config(layout="wide", page_title="Fruit Freshness Predictor")

# Define the base directory
data_dir = '/content/Fruit_Dataset_Final'

# Define the list of class names
class_names = [
    'fresh_apple', 'fresh_banana', 'fresh_grape', 'fresh_mango', 'fresh_orange',
    'rotten_apple', 'rotten_banana', 'rotten_grape', 'rotten_mango', 'rotten_orange'
]

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Function to convert time to days
def convert_to_days(value, metric):
    """Converts a value and metric to total days."""
    if pd.isna(value):
        return 0
    if metric == 'Weeks':
        return int(value * 7)
    elif metric == 'Days':
        return int(value)
    elif metric == 'Months':
        return int(value * 30) # Assuming 30 days per month for simplicity
    else:
        return 0

def parse_duration_string(duration_string):
    """
    Parses a duration string (e.g., '1-3 Weeks', '5 Days', '8-12 Months')
    and returns an estimated number of days as an integer.
    """
    if not isinstance(duration_string, str):
        return 0

    duration_string = duration_string.strip()

    match = re.match(r'(\d+)(?:\s*-\s*(\d+))?\s*(Days|Weeks|Months)', duration_string, re.IGNORECASE)

    if not match:
        return 0 # Unrecognized format

    value1 = int(match.group(1))
    value2 = match.group(2)
    unit = match.group(3).lower()

    if value2:
        avg_value = (value1 + int(value2)) / 2
    else:
        avg_value = value1

    if unit == 'days':
        return int(avg_value)
    elif unit == 'weeks':
        return int(avg_value * 7)
    elif unit == 'months':
        return int(avg_value * 30) # Assuming 30 days per month
    else:
        return 0

# Create the FALLBACK_STORAGE_DURATIONS dictionary
FALLBACK_STORAGE_DURATIONS = {
    'apple': {
        'pantry': '1-3 Weeks',
        'fridge': '3-4 Weeks',
        'freezer': '8-12 Months'
    },
    'banana': {
        'pantry': '3-7 Days',
        'fridge': '1 Week',
        'freezer': '2-3 Months'
    },
    'orange': {
        'pantry': '1-2 Weeks',
        'fridge': '3-4 Weeks',
        'freezer': '10-12 Months'
    },
    'grape': {
        'pantry': '3-5 Days',
        'fridge': '1-2 Weeks',
        'freezer': '2-3 Months'
    },
    'mango': {
        'pantry': '5-7 Days',
        'fridge': '1-2 Weeks',
        'freezer': '6-8 Months'
    }
}

# Cache the FOODKEEPER_DB creation
@st.cache_data
def load_foodkeeper_db():
    file_path = '/content/FoodKeeper-Data.csv'

    # Load the .csv file into a pandas DataFrame
    foodkeeper_df = pd.read_csv(file_path, encoding='latin-1')

    # Define a label_to_id dictionary
    label_to_id = {'apple': 248, 'banana': 251, 'orange': 256, 'grape': 261, 'mango': 265}

    # Create an empty dictionary named FOODKEEPER_DB
    FOODKEEPER_DB = {}

    # Loop through the label_to_id dictionary
    for fruit, item_id in label_to_id.items():
        # Find the correct row in the DataFrame
        fruit_data = foodkeeper_df[foodkeeper_df['ID'] == item_id].iloc[0]

        # Extract the required data, using .get() for potentially missing columns
        pantry_max = fruit_data.get('Pantry_Max', np.nan)
        pantry_metric = fruit_data.get('Pantry_Metric', None)
        refrigerate_max = fruit_data.get('Refrigerate_Max', np.nan)
        refrigerate_metric = fruit_data.get('Refrigerate_Metric', None)

        # New: Extract Freezer_Max and Freezer_Metric using .get() to handle potential KeyError
        freezer_max = fruit_data.get('Freeze_Max', np.nan)
        freezer_metric = fruit_data.get('Freeze_Metric', None)

        # Convert to days
        pantry_days = convert_to_days(pantry_max, pantry_metric)
        fridge_days = convert_to_days(refrigerate_max, refrigerate_metric)
        freezer_days = convert_to_days(freezer_max, freezer_metric) # Calculate freezer_days

        # Populate the FOODKEEPER_DB with days and hardcoded tips
        FOODKEEPER_DB[fruit] = {
            'pantry_days': pantry_days,
            'fridge_days': fridge_days,
            'freezer_days': freezer_days, # Added freezer_days
            'pantry_tip': "Store in a cool, dark place.", # Hardcoded pantry tip
            'fridge_tip': "Refrigerate for extended freshness.", # Hardcoded fridge tip
            'freezer_tip': "Freeze for long-term storage." # Hardcoded freezer tip
        }
    return FOODKEEPER_DB

FOODKEEPER_DB = load_foodkeeper_db()

# Load the trained model
@st.cache_resource
def load_model():
    model = torchvision.models.resnet50(weights=None) # Start with no pre-trained weights

    # Freeze all layers except the final one
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features

    # Replace the final layer with a new one matching our number of classes
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Load the state dictionary from the best performing model during training
    model.load_state_dict(torch.load('/content/fruit_freshness_resnet50.pth', map_location=torch.device('cpu')))

    model = model.to('cpu') # Ensure model is on CPU for Streamlit deployment
    model.eval() # Set to evaluation mode
    return model

model = load_model()

# Prediction function, adapted for Streamlit
def predict_freshness(image, model, storage_type, class_names, data_transforms, FOODKEEPER_DB, device):
    """
    Predicts the freshness of an image and provides storage information.

    Args:
        image (PIL.Image): PIL Image object.
        model (torch.nn.Module): Trained PyTorch model.
        storage_type (str): 'pantry', 'fridge', or 'freezer'.
        class_names (list): List of class names.
        data_transforms (dict): Dictionary of image transformations.
        FOODKEEPER_DB (dict): Database of fruit storage information.
        device (torch.device): The device to run the model on (cpu/cuda).
    """
    # Ensure the model is on the correct device and in evaluation mode
    model = model.to(device)
    model.eval()

    # Apply the 'val' transformation and add a batch dimension
    transformed_image = data_transforms['val'](image).unsqueeze(0)

    # Move the image tensor to the device
    transformed_image = transformed_image.to(device)

    # Run the image through the model to get the output logits
    with torch.no_grad():
        outputs = model(transformed_image)

    # Apply torch.softmax to get probabilities
    probabilities = torch.softmax(outputs, dim=1)

    # Find the highest probability (top_p) and its index (top_i)
    top_p, top_i = probabilities.topk(1, dim=1)

    # Get the predicted_class_name
    predicted_class_name = class_names[top_i.item()]

    # Parse the predicted class name
    parts = predicted_class_name.split('_')
    state = parts[0].capitalize() # 'Fresh' or 'Rotten'
    fruit = parts[1]

    # Query the FOODKEEPER_DB for storage information
    if fruit in FOODKEEPER_DB:
        total_days = FOODKEEPER_DB[fruit].get(f'{storage_type}_days', 0)
        tip = FOODKEEPER_DB[fruit].get(f'{storage_type}_tip', "Information not available for this storage type.")

        # Fallback logic: if total_days from FOODKEEPER_DB is 0 and fruit is Fresh
        if total_days == 0 and state == 'Fresh':
            fallback_duration_str = FALLBACK_STORAGE_DURATIONS.get(fruit, {}).get(storage_type)
            if fallback_duration_str:
                total_days = parse_duration_string(fallback_duration_str)
                tip = f"Fallback storage tip: {FOODKEEPER_DB[fruit].get(f'{storage_type}_tip', 'Store properly.')}"
            else:
                tip = "Information not available, even with fallback."

    else:
        total_days = 0
        tip = "Information not available for this fruit."

    # Calculate the "best guess" for remaining days based on predicted state and confidence
    if state == 'Rotten':
        remaining_days_guess = 0
        remaining_days_text = "0 (Spoiled)"
    else:
        # Calculate "best guess" by scaling total days by confidence
        remaining_days_guess = total_days * top_p.item()
        if total_days == 0 and state == 'Fresh': # Special handling for 0 days from main DB, even with fallback
             remaining_days_text = f"0 (Best guess based on confidence - no duration data)"
        elif 'Fallback storage tip' in tip:
            remaining_days_text = f"{math.ceil(remaining_days_guess)} (Fallback estimate based on confidence)"
        else:
            remaining_days_text = f"{math.ceil(remaining_days_guess)} (Best guess based on confidence)"

    return fruit, state, top_p.item(), remaining_days_text, tip

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.title("Fruit Freshness Predictor")

st.write("Upload an image of a fruit to predict its freshness and get storage tips!")

# Sidebar for image upload and selection
st.sidebar.header("Upload or Select Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

storage_option = st.sidebar.radio(
    "Select Storage Type:",
    ('pantry', 'fridge', 'freezer') # Added freezer option
)

# Main content area
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
    st.write("")
    st.write("Classifying...")

    fruit, state, confidence, remaining_days_text, tip = predict_freshness(image, model, storage_option, class_names, data_transforms, FOODKEEPER_DB, device)

    st.subheader(f"Prediction for: {fruit.capitalize()}")
    st.write(f"**Status:** {state}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Estimated Remaining Days:** {remaining_days_text}")
    st.write(f"**Storage Tip:** {tip}")

else:
    st.write("Or select an example image:")

    # New dynamic example image selection
    selected_fruit_type = st.selectbox("Select Fruit Type", ['apple', 'banana', 'grape', 'mango', 'orange'])
    selected_freshness_state = st.selectbox("Select Freshness State", ['fresh', 'rotten'])

    # Dynamic example image selection logic
    if selected_fruit_type and selected_freshness_state:
        # Construct the target directory based on selections
        target_class_dir = os.path.join(data_dir, 'val', f'{selected_freshness_state}_{selected_fruit_type}')

        # Check if the directory exists and contains images
        if os.path.exists(target_class_dir) and os.listdir(target_class_dir):
            # Get the first image file from the directory
            example_image_filename = os.listdir(target_class_dir)[0]
            example_path = os.path.join(target_class_dir, example_image_filename)

            example_image = Image.open(example_path)
            st.image(example_image, caption=f'Example: {selected_freshness_state.capitalize()} {selected_fruit_type.capitalize()}', use_column_width=False, width=300)

            st.write("")
            st.write("Classifying example...")
            fruit, state, confidence, remaining_days_text, tip = predict_freshness(example_image, model, storage_option, class_names, data_transforms, FOODKEEPER_DB, device)

            st.subheader(f"Prediction for: {fruit.capitalize()}")
            st.write(f"**Status:** {state}")
            st.write(f"**Confidence:** {confidence:.2f}")
            st.write(f"**Estimated Remaining Days:** {remaining_days_text}")
            st.write(f"**Storage Tip:** {tip}")
        else:
            st.write(f"No example images found for {selected_freshness_state} {selected_fruit_type}.")
