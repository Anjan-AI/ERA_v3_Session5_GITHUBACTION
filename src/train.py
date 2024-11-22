import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.model import MnistCNN
from src.utils import count_parameters, save_model

def train_model(epochs=1, batch_size=8):
    device = torch.device('cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                             train=True,
                                             transform=transform,
                                             download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.015)

    # Training
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    accuracy = 100. * correct / total
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = save_model(model, accuracy)
    
    return model, accuracy, model_path

if __name__ == "__main__":
    model, accuracy, model_path = train_model()
    print(f"Training completed with accuracy: {accuracy:.2f}%")
    print(f"Model saved as: {model_path}") 