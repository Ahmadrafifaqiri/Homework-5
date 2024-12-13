import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv('/Users/u1503285/CS-6350-ML/NeuralNetworks/bank-note/train.csv', header=None)
df_test = pd.read_csv('/Users/u1503285/CS-6350-ML/NeuralNetworks/bank-note/test.csv', header=None)
df_train_X = df_train.iloc[:, 0:4].values
df_train_y = df_train.iloc[:, 4].values
df_test_X = df_test.iloc[:, 0:4].values
df_test_y = df_test.iloc[:, 4].values

train_data = TensorDataset(torch.tensor(df_train_X, dtype=torch.float32), torch.tensor(df_train_y, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(df_test_X, dtype=torch.float32), torch.tensor(df_test_y, dtype=torch.float32))

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, depth, width, activation):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_layer_size = input_size
        for _ in range(depth):
            layers.append(nn.Linear(prev_layer_size, width))
            if activation == 'tanh':
                nn.init.xavier_uniform_(layers[-1].weight)  # Xavier initialization for tanh
            elif activation == 'relu':
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')  # He initialization for ReLU
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
            prev_layer_size = width
        layers.append(nn.Linear(prev_layer_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    accuracy = correct / total
    return accuracy

input_size = df_train_X.shape[1]
output_size = 1
batch_size = 32
learning_rate = 1e-3

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ['tanh', 'relu']

for depth in depths:
    for width in widths:
        for activation in activations:
            model = NeuralNetwork(input_size, output_size, depth, width, activation)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            print(f"Training with depth={depth}, width={width}, activation={activation}")
            train_model(model, train_loader, criterion, optimizer)
            train_accuracy = evaluate_model(model, train_loader)
            test_accuracy = evaluate_model(model, test_loader)
            print(f"Train error: {1 - train_accuracy}")
            print(f"Test error: {1 - test_accuracy}")
