import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN, self).__init__()
        
        # First block
        self.conv1 = nn.Conv1d(1, 256, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        
        # Second block
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding='same')
        self.dropout1 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        
        # Third block
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.conv5 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(kernel_size=8, stride=8, padding=0)
        
        # Fourth block
        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.flatten = nn.Flatten()
        self.dropout3 = nn.Dropout(0.2)
        
        # Dense layer
        self.fc1 = nn.Linear((input_length // 64) * 128, num_classes)
        self.bn4 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # First block
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # Second block
        x = self.pool1(self.bn2(self.dropout1(F.relu(self.conv2(x)))))
        
        # Third block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(self.bn3(self.dropout2(x)))
        
        # Fourth block
        x = F.relu(self.conv6(x))
        x = self.flatten(x)
        x = self.dropout3(x)
        
        # Dense layer
        x = self.fc1(x)
        x = self.bn4(x)
        return F.softmax(x, dim=1)  # Class probabilities
    
def eval_model(model, test_loader:DataLoader,criterion,device:str)-> tuple:
    """
    Evaluate the model on the test set.

    Args:
        model: The model to evaluate.
        test_loader (DataLoader): The DataLoader for the test set.
        criterion: The loss function.
        device (str): The device to run the model on, e.g., 'cpu', 'cuda', or 'mps'.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss on the test set.
            - accuracy (float): The accuracy on the test set.
    
    """
    correct = 0
    cur_loss = 0

    model.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            cur_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    # Compute average loss and accuracy
    avg_loss = cur_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    return avg_loss, accuracy

def train_model(model, train_loader:DataLoader,test_loader:DataLoader, criterion, optimizer,scheduler, device:str, num_epochs:int=10,verbos:bool=True)-> tuple:
    """
    Trains a given model using the provided training and testing data loaders, optimizer, and scheduler.

    Args:
        model: The model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        criterion: The loss function.
        optimizer: The optimizer to use for training.
        scheduler: The scheduler to use for training.
        device (str): The device to run the model on, e.g., 'cpu', 'cuda', or 'mps'.
        num_epochs (int): The number of epochs to train the model for.
        verbos (bool): Whether to print training progress.

    Returns:

        tuple: A tuple containing:
            - train_losses (list): The training losses for each epoch.
            - test_losses (list): The test losses for each epoch.
            - train_accuracies (list): The training accuracies for each epoch.
            - test_accuracies (list): The test accuracies for each epoch.
    """
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Test the model
        test_loss, test_accuracy = eval_model(model, test_loader,criterion,device)
        scheduler.step(test_accuracy)


        

        if (epoch+1) % 10 == 0:
            if verbos:
                print(f"Epoch [{epoch+1}/{num_epochs}], TRLoss: {total_loss/len(train_loader):.4f}, TRAccuracy: {correct/total:.4f}, TSLoss: {test_loss:.4f}, TSAccuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)   
            train_losses.append(total_loss/len(train_loader.dataset))
            train_accuracies.append(correct/total)
            test_accuracies.append(test_accuracy)

    return train_losses, test_losses, train_accuracies, test_accuracies 


def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    Plot the loss and accuracy curves.

    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        train_accuracies (list): List of training accuracies.
        test_accuracies (list): List of testing accuracies.
    """

    plt.figure(figsize=(8, 8))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Testing Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(test_accuracies, label='Testing Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.show()

def _label_to_emotion_for_metrics(label:int)->str:
    if label == 0:
        return "neutral"
    elif label == 1:
        return "calm"
    elif label == 2:
        return "happy"
    elif label == 3:
        return "sad"
    elif label == 4:
        return "angry"
    elif label == 5:
        return "fearful"
    elif label == 6:
        return "disgust"
    elif label == 7:
        return "surprised"
    else:
        return "unknown"

def get_all_metrics(model, test_loader, device):
    """
    Compute and print all metrics for the given model and test set.
    
    Args:
        model: The model to evaluate.
        test_loader (DataLoader): The DataLoader for the test set.
        device (str): The device to run the model on, e.g., 'cpu', 'cuda', or 'mps'.
    
    """

# Move model to evaluation mode
    model.eval()

    # Initialize lists to store true labels and predictions
    all_y_true = []
    all_y_pred = []

    # Process the entire test dataset
    with torch.no_grad():  # No need to compute gradients
        for batch in test_loader:  # Assuming you have a DataLoader
            x_batch, y_batch = batch  # Extract features and labels
            x_batch = x_batch.to(device)  # Move features to device
            
            # Get predictions
            y_pred_batch = model(x_batch).argmax(dim=1)  # Get predicted class
            
            # Store results (move to CPU and convert to list)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(y_pred_batch.cpu().numpy())

    # Convert to NumPy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    # print(np.array(np.unique(all_y_true, return_counts=True)).T)
    # print(np.array(np.unique(all_y_pred, return_counts=True)).T)

    # Compute metrics
    precision = precision_score(all_y_true, all_y_pred, average='macro',zero_division=1) #Unweighted Average Precision
    recall = recall_score(all_y_true, all_y_true, average='macro',zero_division=1) # UAR 
    f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=1)

    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    # Print results
    print(f"Macro Precision: {precision:.4f}")
    print(f"UAR (Macro Recall): {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    # Display Confusion Matrix

    label_names = [_label_to_emotion_for_metrics(i) for i in range(8)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=label_names))
