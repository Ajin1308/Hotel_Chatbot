import json
from utils.nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from training.evaluation import evaluate_model
from model.model import NeuralNet


def open_dataset(file_path):
    with open(file_path, 'r') as f:
        intent_data = json.load(f)
    return intent_data


def preprocess_intent_data(intents):

    all_words = []
    tags = []
    xy = []


    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w,tag))

    ignore_words = ['?', ',', '!', '.']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return all_words,tags,xy


def split_data(all_words,tags,xy):

    X_train = []
    y_train = []

    for (patterns_sentence, tag) in xy:
        bag = bag_of_words(patterns_sentence, all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label) 

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return np.array(X_train), np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self,X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

def train_model(dataset, input_size, output_size, hidden_size, learning_rate, num_epochs, batch_size):

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = NeuralNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            labels = labels.to(torch.long)
            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss: {loss.item():.4f}')
    return model


def save_model(model, input_size, output_size, hidden_size, all_words, tags, filename):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    torch.save(data, filename)
    print(f'Model saved to {filename}')


    
if __name__ == "__main__":
    #Hyperparameters
    batch_size = 8 
    hidden_size = 8
    learning_rate = 0.001
    num_epochs = 1000
    file_path = 'artifacts/intents.json'
    model_file = "artifacts/hotel_bot.pth"

    # Load and preprocess data
    intents = open_dataset(file_path)
    all_words, tags, xy = preprocess_intent_data(intents)
    X_train, y_train = split_data(all_words, tags, xy)
    dataset = ChatDataset(X_train, y_train)

    input_size = len(X_train[0])
    output_size = len(tags)
    model = train_model(dataset, input_size, output_size, hidden_size, learning_rate, num_epochs, batch_size)

    # Evaluate model
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    accuracy, precision, recall, f1 = evaluate_model(model, dataloader)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Save model
    save_model(model, input_size, output_size, hidden_size, all_words, tags, model_file)




