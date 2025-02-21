# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

#Set up Transforms (train, val, and test)

#<<<YOUR CODE HERE>>>

# Define the transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop and resize images
    transforms.RandomHorizontalFlip(),  # Apply horizontal flipping
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

val_transforms = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




#Set up DataLoaders (train, val, and test)
batch_size = 10
num_workers = 4

#<<<YOUR CODE HERE>>>
data_dir = '/Users/raneem/Documents/GitHub/cd12528-small-data-project-starter/starter_code/part1-transfer/imagedata-50'
train_dir = data_dir + '/train'
val_dir = data_dir + '/val'
test_dir = data_dir + '/test'

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#hint, create a variable that contains the class_names. You can get them from the ImageFolder
class_names = train_dataset.classes



# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifer for our 3 categories

#<<<YOUR CODE HERE>>>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)

# Freeze the feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

# Get the number of input features for the classifier
num_features = model.classifier[6].in_features

# Replace the classifier with a new one for our 3 classes
model.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),  # 3 output classes
    nn.LogSoftmax(dim=1)  # Using LogSoftmax for classification
)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

#<<<YOUR CODE HERE>>>
# Number of epochs
num_epochs = 10

# Loss function (since we used LogSoftmax in the model, use NLLLoss)
criterion = nn.NLLLoss()

# Optimizer (Adam is good for transfer learning)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# Learning rate scheduler (Reduce LR when validation loss plateaus)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase, loader in zip(['train', 'val'], [train_loader, val_loader]):
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


def test_model(test_loader, model, class_names):
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}% ✅")

    # Show sample predictions
    visualize_predictions(test_loader, model, class_names, device)

def visualize_predictions(test_loader, model, class_names, device):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()

    for img, label, pred, ax in zip(images.cpu(), labels.cpu(), preds.cpu(), axes):
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Actual: {class_names[label]}\nPredicted: {class_names[pred]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# When you have all the parameters in place, uncomment these to use the functions imported above
def main():
  trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, train_loader, val_loader, num_epochs=num_epochs)
  test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
   main()
   print("done")