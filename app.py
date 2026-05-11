import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Load the LSTM model and tokenizer
model = load_model('next_word_prediction_model.h5') # Load the trained model from the file named 'next_word_prediction_model.h5'
with open('tokenizer.pickle', 'rb') as handle: # Open the file named 'tokenizer.pickle' in read-binary mode
    tokenizer = pickle.load(handle) # Load the tokenizer object from the file using pickle

def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0] # Convert the input text into a sequence of integers using the tokenizer
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre') # Pad the sequence to ensure it has the same length as the model's input
    predicted = model.predict(sequence, verbose=0) # Use the model to predict the next word in the sequence
    predicted_word_index = np.argmax(predicted) # Get the index of the predicted word with the highest probability
    for word, index in tokenizer.word_index.items(): # Iterate through the tokenizer's word index to find the corresponding word for the predicted index
        if index == predicted_word_index: # If the index matches the predicted word index
            return word # Return the predicted word
    return None # Return None if no matching word is found

# Streamlit app
st.title("Next Word Prediction with LSTM") # Set the title of the Streamlit app
input_text = st.text_input("Enter a text sequence:") # Create a text input field
if st.button("Predict Next Word"): # Create a button to trigger the prediction
    if input_text: # Check if the input text is not empty
        max_sequence_len = model.input_shape[1] + 1 # Get the maximum sequence length from the model's input shape
        predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_len) # Predict the next word in the input text sequence
        if predicted_word: # Check if a predicted word was returned
            st.write(f"Predicted next word: {predicted_word}") # Display the predicted next word
        else:
            st.write("Could not predict the next word.") # Display a message if no prediction could be made
    else:
        st.write("Please enter a text sequence.") # Prompt the user to enter a text sequence if the input is empty



