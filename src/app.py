import streamlit as st
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import nltk
nltk.download('punkt')


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

def preprocess_user_input(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    return X

def predict(output):
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag, prob

def app():
    st.title("Hotel Assistant ChatBot")
    st.markdown("Type a message and press Send to chat. Type 'exit' to quit.")
    chat_history = load_chat_history()
    user_input = st.text_area("You:", key="user_input", height = 100)

    if st.button("Send"):
        if user_input.lower() == "exit":
            chat_history = []
            save_chat_history(chat_history)
            st.stop()

        chat_history.append(("You", user_input))
        
        output = model(preprocess_user_input(user_input))

        tag, prob = predict(output)

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    bot_response = random.choice(intent['responses'])
                    chat_history.append((bot_name, bot_response))            
        else:
            chat_history.append((bot_name, "I do not understand..."))
        save_chat_history(chat_history)

    # Display chat history
    st.markdown("## Chat")
    for speaker, message in chat_history[::-1]:
        st.markdown(f"<span style='font-size:20px;'>**{speaker}**: {message}</span>", unsafe_allow_html=True)
    # for i in chat_history:
    #     print(i)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(r'C:\Users\user\Desktop\Hotel_chatbot\Hotel_Chatbot\artifacts\intents.json', 'r') as f:
        intents = json.load(f)

    FILE = r"C:\Users\user\Desktop\Hotel_chatbot\Hotel_Chatbot\artifacts\Hotel_model.pth"
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

    bot_name = "HOTBOT"

    chat_history_file = r"C:\Users\user\Desktop\Hotel_chatbot\Hotel_Chatbot\artifacts\chat_history.json"

    app()
