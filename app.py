import streamlit as st
import nbformat
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Dog Breed Classification App")

# File uploader for .ipynb file
notebook_file = st.file_uploader("Upload Jupyter Notebook (.ipynb)", type=["ipynb"])

if notebook_file is not None:
    st.subheader("Notebook Content")
    try:
        # Read and parse the notebook
        notebook_content = notebook_file.getvalue().decode("utf-8")
        notebook = nbformat.reads(notebook_content, as_version=4)
        
        # Display markdown and code cells
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                st.markdown(cell.source)
            elif cell.cell_type == "code":
                st.code(cell.source, language="python")
                if cell.outputs:
                    for output in cell.outputs:
                        if output.output_type == "stream":
                            st.text(output.text)
                        elif output.output_type == "display_data" or output.output_type == "execute_result":
                            if "text/plain" in output.data:
                                st.text(output.data["text/plain"])
                            elif "text/html" in output.data:
                                st.write(output.data["text/html"])
                            elif "image/png" in output.data:
                                st.image(io.BytesIO(base64.b64decode(output.data["image/png"])))
    except Exception as e:
        st.error(f"Error reading notebook: {e}")

# File uploader for .h5 model file
model_file = st.file_uploader("Upload Model (.h5)", type=["h5"])

if model_file is not None:
    st.subheader("Model Information")
    try:
        # Save model temporarily
        with open("temp_model.h5", "wb") as f:
            f.write(model_file.getvalue())
        
        # Load the model
        model = tf.keras.models.load_model("temp_model.h5")
        
        # Display model summary
        st.write("Model Summary:")
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        st.text("\n".join(summary))
    except Exception as e:
        st.error(f"Error loading model: {e}")

# File uploader for custom image
image_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "jpeg", "png"])

if image_file is not None and model_file is not None:
    st.subheader("Dog Breed Prediction")
    try:
        # Load and preprocess image
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Assuming model expects 224x224 input
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Load labels from the notebook's labels.csv (assuming it's available)
        labels_csv = pd.read_csv("labels.csv")  # Update path if hosted elsewhere
        unique_breeds = np.unique(labels_csv["breed"])
        
        # Make prediction
        pred_probs = model.predict(img_array)
        pred_label = unique_breeds[np.argmax(pred_probs[0])]
        
        # Display image and prediction
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write(f"Predicted Breed: **{pred_label}**")
        st.write("Prediction Probabilities:")
        for i, breed in enumerate(unique_breeds):
            st.write(f"{breed}: {pred_probs[0][i]:.4f}")
    except Exception as e:
        st.error(f"Error processing image or prediction: {e}")
