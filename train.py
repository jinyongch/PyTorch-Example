import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torchvision import models


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name, e.g., resnet50, resnet18, etc.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

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
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"Starting training on world_size={world_size}, rank={rank}, local_rank={local_rank}")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    train_sampler = distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    test_sampler = distributed.DistributedSampler(testset, num_replicas=world_size, rank=rank)
    testloader = DataLoader(testset, batch_size=batch_size, sampler=test_sampler)

    model = get_model(model_name).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    total_iterations = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_iterations = len(trainloader)
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_iterations += 1
            
            current_average_loss = running_loss / total_iterations

            print(f"[Epoch {epoch + 1:03d}/{epochs}] Iteration {total_iterations:03d}/{epoch_iterations} | Current Average Loss: {current_average_loss:.4f} | world_size: {world_size} | rank: {rank} | local_rank: {local_rank}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
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

    print(f"Final best accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    setup()
    args = parse_args()
    print(f"Training {args.model} for {args.epochs} epochs with batch size {args.batch_size} and learning rate {args.lr}")
    train_and_evaluate(args.model, args.epochs, args.batch_size, args.lr)
    cleanup()
