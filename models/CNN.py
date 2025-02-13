import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from models.Best_Model_Saver import BestModelSaver


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
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        # Fourth block
        self.conv6 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.flatten = nn.Flatten()
        self.dropout3 = nn.Dropout(0.2)
        
        # Dense layer
        self.fc1 = nn.Linear((input_length // 8) * 128,num_classes)
        self.bn4 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # First block
        x = self.relu1(self.bn1(self.conv1(x)))
        # Second block
        x = self.pool1(self.bn2(self.dropout1(F.relu(self.conv2(x)))))
        # Third block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.dropout2(self.relu2(self.bn3(x)))
        
        # Fourth block
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.dropout3(x)
        
        # Dense layer
        x = self.fc1(x)
        x = self.bn4(x)
        return F.softmax(x, dim=1)  # Class probabilities

class EarlyStoppingAccuracy:
    """Stop training if validation accuracy does not improve after a certain number of epochs."""
    def __init__(self, patience=10, verbose=False, delta=0, path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement before stopping.
            verbose (bool): If True, prints messages when stopping.
            delta (float): Minimum change to qualify as an improvement.
            path (str): Filepath to save the best model.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = 0.0

    def __call__(self, val_acc, model):
        """
        Args:
            val_acc (float): Current validation accuracy.
            model (torch.nn.Module): Model to save if accuracy improves.
        """
        score = val_acc  # We maximize accuracy, so higher is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0  # Reset counter

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy improves."""
        torch.save(model.state_dict(), self.path)
        self.best_acc = val_acc
        if self.verbose:
            print(f"Validation accuracy improved. Saving model to {self.path}")


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

def train_model(model, train_loader:DataLoader,test_loader:DataLoader,
                 criterion, optimizer,scheduler,
                 device:str,
                 num_epochs:int=10,
                 use_early_stopping:bool=False,
                 verbos:bool=True,
                 print_best = True)-> tuple:
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
    early_stopping = EarlyStoppingAccuracy(patience=10, verbose=False, path='best_CNN_model.pth',delta=0.001)
    model_saver = BestModelSaver()
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
        if scheduler is not None:
            scheduler.step(test_accuracy)
        # Early stopping
        if use_early_stopping:
            early_stopping(test_accuracy, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        #  Best Model saving
        model_saver(test_accuracy,model)

        

        if (epoch+1) % 10 == 0:
            if verbos:
                print(f"Epoch [{epoch+1}/{num_epochs}], TRLoss: {total_loss/len(train_loader):.4f}, TRAccuracy: {correct/total:.4f}, TSLoss: {test_loss:.4f}, TSAccuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)   
            train_losses.append(total_loss/len(train_loader.dataset))
            train_accuracies.append(correct/total)
            test_accuracies.append(test_accuracy)
        if epoch+1 == num_epochs:
            print(f"Best_model Acc:{model_saver.best_acc}")

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

    label_names = [_label_to_emotion_for_metrics(i) for i in range(8)]


    # Print classification report
    print("Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=label_names))

    # Display Confusion Matrix

    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    fig ,ax= plt.subplots(figsize=(8, 6))

    cm_display = ConfusionMatrixDisplay(conf_matrix,display_labels=label_names)
    cm_display.plot(cmap='magma',xticks_rotation = 45,ax=ax,values_format='.1f')

    plt.show()


