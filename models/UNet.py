import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

from models.Best_Model_Saver import BestModelSaver


class UNet(nn.Module):
    def __init__(self, num_classes=8, threshold_value=0.5):
        super(UNet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down_conv_1 = _double_conv(1, 16)
        self.down_conv_2 = _double_conv(16, 32)
        self.down_conv_3 = _double_conv(32, 64)

        self.up_trans_1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2)
        self.up_conv_1 = _double_conv(64, 32)

        self.up_trans_2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.up_conv_2 = _double_conv(32, 16)

        self.out = nn.Conv1d(16, num_classes, kernel_size=1)  # Output for 8 classes
        self.threshold_value = threshold_value

    def forward(self, image):
        # Encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)



        # Decoder
        x = self.up_trans_1(x6)
        if x.size() != x3.size():
            diff = x3.size(2) - x.size(2)
            x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.up_conv_1(torch.cat([x, x3], 1))

        x = self.up_trans_2(x)
        if x.size() != x1.size():
            diff = x1.size(2) - x.size(2)
            x = F.pad(x, [diff // 2, diff - diff // 2])
        x = self.up_conv_2(torch.cat([x, x1], 1))

        # Output layer
        x = self.out(x)  # Shape: [batch_size, num_classes, seq_length]
        x = torch.mean(x, dim=2)  # Global average pooling: [batch_size, num_classes]
        x = F.softmax(x, dim=1)  # Class probabilities
        return x

def _double_conv(in_channels, out_channels, kernel_size=5):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.Dropout(p=0.3),  # Dropout after activation
        nn.BatchNorm1d(out_channels, track_running_stats=False),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.Dropout(p=0.5),  # Dropout after activation
        nn.BatchNorm1d(out_channels, track_running_stats=False),
        nn.ReLU(inplace=True),
    )

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

def train_model(model, train_loader:DataLoader,test_loader:DataLoader,scheduler, criterion, optimizer,device:str, num_epochs:int=10,verbos:bool=True)-> tuple:
    """
        Trains a given model using the provided training and testing data loaders, optimizer, and scheduler.
        
        Args:
            model: The model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the testing dataset.
            scheduler: Learning rate scheduler.
            criterion: Loss function.
            optimizer (torch.optim): Optimizer for updating model weights.
            device:(str) The device to run the model on, e.g., 'cpu', 'cuda', or 'mps'.
            num_epochs (int, optional): Number of epochs to train the model. Default is 10.
            verbos (bool, optional): If True, print training progress. Default is True.

        Returns:

            tuple: A tuple containing four lists:
            
                - train_losses: List of training losses per epoch.
                - test_losses: List of testing losses per epoch.
                - train_accuracies: List of training accuracies per epoch.
                - test_accuracies: List of testing accuracies per epoch.    
    """

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    model_saver = BestModelSaver()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            # Move data to GPU if available
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

        # Save best model
        model_saver(test_accuracy,model)
        

        if (epoch+1) % 10 == 0 :
            if verbos:
                print(f"Epoch [{epoch+1}/{num_epochs}], TRLoss: {total_loss/len(train_loader):.4f}, TRAccuracy: {correct/total:.4f}, TSLoss: {test_loss:.4f}, TSAccuracy: {test_accuracy:.4f}")
            test_losses.append(test_loss)   
            train_losses.append(total_loss/len(train_loader.dataset))
            train_accuracies.append(correct/total)
            test_accuracies.append(test_accuracy)
        if epoch+1 == num_epochs:
            print(f"Best Model Acc: {model_saver.best_acc}")

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

    plt.figure(figsize=(8,8))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Testing Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,8))
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
    cm_display.plot(cmap='Blues',xticks_rotation = 45,colorbar=False,ax=ax)

    ax.set_title("Confusion Matrix for CNN Model")

    plt.show()


