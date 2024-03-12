import streamlit as st
import random
import json
import torch
from model.model import NeuralNet
from utils.nltk_utils import bag_of_words, tokenize
import nltk
nltk.download('punkt')


def save_chat_history(chat_history):
    '''
    This function is used to save chat history on specified file. 'w' -> write

    '''
    with open(chat_history_file, "w") as file:
        json.dump(chat_history, file)

def load_chat_history():
    '''
    This function is used to load chat history. 'r' -> read
    
    '''
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
    '''
    This function takes user input and tokenize them pass through bag of words function, reshape, return numpy array.

    '''
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    return X

def predict(output):
    '''
    This function takes in the output of the model, extracts the predicted tag by applying the softmax activation function,
    and returns the predicted tag along with its probability.

    '''
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag, prob

def app():
    '''
    This function is use to make streamlit app.

    '''

    st.title("Hotel Assistant ChatBot")
    st.markdown("Type a message and press Send to chat. Type 'exit' to quit.")
    #load chat history that is stoed in the chat_history json file
    chat_history = load_chat_history()
    #ask for user input
    user_input = st.text_area("You:", key="user_input", height = 100)

    if st.button("Send"):
        #if user gives input as "exit", clears chat history
        if user_input.lower() == "exit":
            chat_history = []
            save_chat_history(chat_history)
            st.stop()

        #appending user input into chat history
        chat_history.append(("You", user_input))
        
        output = model(preprocess_user_input(user_input))

        tag, prob = predict(output)

        #checks if probability is higher that 0.75
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                #checking if predicted tag is there in the intent file
                if tag == intent["tag"]:
                    #randomly chooses a response from intent file
                    bot_response = random.choice(intent['responses'])
                    #appends the bot reply in the chat hsitory
                    chat_history.append((bot_name, bot_response))     
        #if less that 0.75       
        else:
            chat_history.append((bot_name, "I do not understand..."))

        #after going through all loops, save the chat history
        save_chat_history(chat_history)

    # Display chat history
    st.markdown("## Chat")
    #reversing the chat history so that we can see the new chat in the top
    for speaker, message in chat_history[::-1]:
        st.markdown(f"<span style='font-size:20px;'>**{speaker}**: {message}</span>", unsafe_allow_html=True)
    # for i in chat_history:
    #     print(i)


if __name__ == "__main__":
    #checks if our system has gpu or not. if gpu the it will use gpu else will use cpu
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

    bot_name = "HOTBOT"

    chat_history_file = r"artifacts\chat_history.json"

