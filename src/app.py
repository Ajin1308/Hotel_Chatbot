import streamlit as st
import random
import json
import torch
from model.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
import nltk
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(r'artifacts\intents.json', 'r') as f:
    intents = json.load(f)

FILE = r"artifacts\Hotel_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Assistant"

# List to store chat history
chat_history_file = r"artifacts\chat_history.json"
booking_history_file = r"artifacts\booking_history.json"

def save_chat_history(chat_history):
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file)

def load_chat_history():
    try:
        with open(chat_history_file, "r") as file:
            data = file.read()
            if data:
                return json.loads(data)
            else:
                return []  # Return an empty list if file is empty
    except FileNotFoundError:
        return []  # Return an empty list if file doesn't exist
    except json.JSONDecodeError:
        return []  # Return an empty list if file contains invalid JSON


def chat():
    st.title("Hotel Assistant ChatBot")
    st.markdown("Type a message and press Cntrl+Enter to chat. Type 'exit' to quit.")
    chat_history = load_chat_history()
    user_input = st.text_area("You:", key="user_input", height = 100)

    if st.button("Send"):
        if user_input.lower() == "exit":
            chat_history = []
            save_chat_history(chat_history)
            st.stop()

        chat_history.append(("You", user_input))
        
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    bot_response = random.choice(intent['responses'])
                    chat_history.append((bot_name, bot_response))
                    
        else:
            chat_history.append((bot_name, "I do not understand..."))
        save_chat_history(chat_history)
        st.empty()  # Clear the text area after sending the message

    # Display chat history
    st.markdown("## Chat")
    for speaker, message in chat_history[::-1]:
        st.text(f"{speaker}: {message}")
    for i in chat_history:
        print(i)


if __name__ == "__main__":
    chat()
