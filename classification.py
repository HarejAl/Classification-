"""
An example of using a CNN for recognising digits in the MNIST dataset
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# Download the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert numpy arrays to PyTorch tensors
x_train = torch.tensor((np.float32(x_train)/255).reshape(-1,1,28,28))
y_train = torch.tensor(y_train, dtype=torch.long)
x_test  = torch.tensor((np.float32(x_test)/255).reshape(-1,1,28,28))
y_test  = torch.tensor(y_test, dtype=torch.long)

# Ploute the training dataset with noise 
std_dev = 0.1
# Generate noise with the same shape as x_train
noise = torch.randn_like(x_train) * std_dev
# Add noise to x_train
x_train = x_train + noise

# Create DataLoader objects for batching
batch_size = 128
train_dataset = TensorDataset(x_train, y_train)
test_dataset  = TensorDataset(x_test, y_test)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a simple CNN for MNIST classification
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1   = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: (32, 28, 28)
        self.pool    = nn.MaxPool2d(2, 2)                          # Halves the spatial dimensions
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: (64, 14, 14)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)                  # After two poolings: 28/2/2 = 7
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 10)                         # 10 classes (digits 0-9)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits (no softmax needed)
        return x

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Decide if train a new model
new_model = True; training = True

if new_model:
    model = CNN().to(device)
else:
    model = torch.load('mnist_cnn.pth', weights_only=False)
    model.to(device)

if training:

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()          # Zero the gradients
            outputs = model(inputs)        # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()                # Backward pass
            optimizer.step()               # Update weights
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(test_loader.dataset)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {accuracy:.2f}%')

    # Optionally, save the trained model
    torch.save(model, 'mnist_cnn.pth')

# ----------------- Confusion Matrix and Example ------------------
# Compute predictions on the test set to build the confusion matrix
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute the confusion matrix using sklearn
cm = confusion_matrix(all_labels, all_preds)

# Normalize each row (i.e., by true label count)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix with percentages
plt.figure(figsize=(8, 6))
plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (%)")
plt.colorbar()

tick_marks = np.arange(cm.shape[0])
plt.xticks(tick_marks, [str(i) for i in range(cm.shape[1])])
plt.yticks(tick_marks, [str(i) for i in range(cm.shape[0])])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Annotate the cells with % values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm_percent[i, j]:.1f}%", 
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > 50 else "black")

plt.tight_layout()
plt.show()

# Display an example image with its predicted and true label
# Select a random test image
idx = random.randint(0, len(x_test) - 1)
example_img = x_test[idx].unsqueeze(0).to(device)  # Add batch dimension
model.eval()
with torch.no_grad():
    output = model(example_img)
    _, pred = torch.max(output, 1)
true_label = y_test[idx].item()

plt.figure()
plt.imshow(x_test[idx].cpu().numpy().squeeze(), cmap='gray')
plt.title(f"True Label: {true_label} | Predicted: {pred.item()}")
plt.axis('off')
plt.show()
