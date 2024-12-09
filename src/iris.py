import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X, y = shuffle(X, y, random_state=42)  # Shuffle to randomize data order
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Neural Network
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 inputs to a hidden layer with 16 neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)  # 16 hidden to 3 output neurons (classes)
        self.softmax = nn.Softmax(dim=1)  # Output layer activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Instantiate the model, define loss function and optimizer
model = IrisNet()
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
epochs = 100
for epoch in range(epochs):
    model.train()  # Set to training mode
    optimizer.zero_grad()  # Zero out gradients
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()  # Set to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Accuracy: {accuracy:.4f}")






# Example single data point (e.g., [sepal length, sepal width, petal length, petal width])
single_sample = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your own input

# Preprocess the sample (standardize using the same scaler)
single_sample = scaler.transform(single_sample)  # StandardScaler expects 2D input
single_sample_tensor = torch.tensor(single_sample, dtype=torch.float32)

# Predict
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation for inference
    output = model(single_sample_tensor)  # Forward pass
    _, predicted_class = torch.max(output, 1)  # Get class with highest probability

print(output)
print(predicted_class)

# Convert class index to class name
predicted_class_name = iris.target_names[predicted_class.item()]

print(f"Predicted class: {predicted_class_name}")
