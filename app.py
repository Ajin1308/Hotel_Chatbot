import json
import random
import streamlit as st
import torch
import nltk
from src.model import NeuralNet
from src.utils import (
    preprocess_user_input,
    predict, load_chat_history, save_chat_history
)

# Downloading required NLTK data
nltk.download('punkt_tab')

# Load intents file
with open(r'artifacts/intents.json', 'r') as f:
    intents = json.load(f)

# Load pre-trained model data
FILE = r"artifacts/Hotel_model.pth"
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Set device for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Bot name
bot_name = "HOTBOT"

# Streamlit Chatbot Interface
st.title("Hotel Assistant ChatBot")
st.markdown("Type a message and press Send to chat. Type 'exit' to quit.")

# Load chat history
chat_history = load_chat_history()

# Get user input
user_input = st.text_area("You:", key="user_input", height=100)

if st.button("Send"):
    # If user types 'exit', clear chat history
    if user_input.lower() == "exit":
        chat_history = []
        save_chat_history(chat_history)
        st.stop()

    # Append user input to chat history
    chat_history.append(("You", user_input))

    # Process user input and get model prediction
    output = model(preprocess_user_input(user_input, all_words, device))
    tag, prob = predict(output, tags)

    # Check if the model's confidence is above 75%
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                chat_history.append((bot_name, bot_response))
    else:
        chat_history.append((bot_name, "I do not understand..."))

    # Save chat history after response
    save_chat_history(chat_history)

# Display chat history in reverse order (newest first)
st.markdown("## Chat")
for speaker, message in chat_history[::-1]:
    st.markdown(f"<span style='font-size:20px;'>**{speaker}**: {message}</span>", unsafe_allow_html=True)
