import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Dog breeds list
dog_breeds = [
    "Chihuahua", "Japanese Spaniel", "Maltese Dog", "Pekinese", "Shih-Tzu", "Blenheim Spaniel", "Papillon",
    "Toy Terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset", "Beagle", "Bloodhound", "Bluetick",
    "Black-and-Tan Coonhound", "Walker Hound", "English Foxhound", "Redbone", "Borzoi", "Irish Wolfhound",
    "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
    "Scottish Deerhound", "Weimaraner", "Staffordshire Bullterrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
    "Norwich Terrier", "Yorkshire Terrier", "Wire-Haired Fox Terrier", "Lakeland Terrier", "Sealyham Terrier",
    "Airedale", "Cairn", "Australian Terrier", "Dandie Dinmont", "Boston Bull", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scotch Terrier", "Tibetan Terrier", "Silky Terrier",
    "Soft-Coated Wheaten Terrier", "West Highland White Terrier", "Lhasa", "Flat-Coated Retriever",
    "Curly-Coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever",
    "German Short-Haired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter",
    "Brittany Spaniel", "Clumber", "English Springer", "Welsh Springer Spaniel", "Cocker Spaniel",
    "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard",
    "Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "Collie", "Border Collie",
    "Bouvier des Flandres", "Rottweiler", "German Shepherd", "Doberman", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller", "Entlebucher", "Boxer",
    "Bull Mastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "Saint Bernard", "Eskimo Dog",
    "Malamute", "Siberian Husky", "Affenpinscher", "Basenji", "Pug", "Leonberg", "Newfoundland",
    "Great Pyrenees", "Samoyed", "Pomeranian", "Chow", "Keeshond", "Brabancon Griffon", "Pembroke",
    "Cardigan", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican Hairless", "Dingo",
    "Dhole", "African Hunting Dog"
]

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_2.h5")
    return model

model = load_model()

# App UI
st.title("🐶 Dog Breed Classifier")
st.markdown("Upload a dog image and get the predicted breed using a pre-trained EfficientNetB0 model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = tf.argmax(predictions[0]).numpy()
    predicted_breed = dog_breeds[predicted_index]
    confidence = tf.reduce_max(predictions[0]).numpy() * 100

    # Output
    st.markdown(f"### 🐾 Predicted Breed: `{predicted_breed}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
