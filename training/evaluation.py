import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def evaluate_model(model, dataloader):
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for (words, labels) in dataloader:
            outputs = model(words)
            _, predicted = torch.max(outputs, dim=1)

            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1

