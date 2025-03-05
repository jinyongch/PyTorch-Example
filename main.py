import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name, e.g., resnet50, resnet18, etc.")
    parser.add_argument("--epochs", type=int, default=100000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    return parser.parse_args()

def get_model(model_name, num_classes=10):
    if model_name == "resnet18":
        return models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet50":
        return models.resnet50(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet101":
        return models.resnet101(pretrained=False, num_classes=num_classes)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported")

def train_and_evaluate(model_name, epochs, batch_size, learning_rate):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = get_model(model_name).cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_iterations = 0
        epoch_iterations = len(trainloader)
        for inputs, labels in trainloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_iterations += 1
            
            current_average_loss = running_loss / total_iterations

            print(f"[Epoch {epoch + 1:03d}/{epochs}] Iteration {total_iterations:03d}/{epoch_iterations} | Current Average Loss: {current_average_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the 10000 test images: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            os.makedirs("output", exist_ok=True)
            torch.save(model.state_dict(), f"output/best_{model_name}_cifar10.pth")
            print(f"Best model saved as output/best_{model_name}_cifar10.pth")
        
        scheduler.step()

    print(f"Final best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    args = parse_args()
    print(f"Training {args.model} for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.lr}")
    train_and_evaluate(args.model, args.epochs, args.batch_size, args.lr)
