# Karan Shah 
# Transfer Learning applied to Plastics Classification

# Import statements
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory


# Loads the data
def loadData(root, batch_size, image_size): 
    data_transforms = {

    'train': transforms.Compose([ 
        # Flip across vertical axis
        transforms.RandomHorizontalFlip(),
         # Randomizing the cropping 
        transforms.Resize((image_size, image_size)),
        # Create tensor
        transforms.ToTensor(),
        # Normalize to 0.5 mean and 0.5 std dev across 3 channels
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
}

    data_dir = root
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}
    

    return dataloaders, image_datasets

# Displays the image
def imshow(inp, title=None):
    """Display image for Tensor."""
    # Moves the dimensions of the tensor
    inp = inp.permute((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp.numpy() + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    

# Initialize the model based on modelName parameter
def transferModels(modelName, weights='IMAGENET1K_V1'):
    if modelName == "resnet34":
        resnet34 = torchvision.models.resnet34(weights=weights)
        for param in resnet34.parameters():
            param.requires_grad = False
        return resnet34

    elif modelName == "resnet50":
        resnet50 = torchvision.models.resnet50(weights=weights)
        for param in resnet50.parameters():
            param.requires_grad = False
        return resnet50

    elif modelName == "densenet121":
        densenet121 = torchvision.models.densenet121(weights=weights)
        for param in densenet121.parameters():
            param.requires_grad = False
        return densenet121
    else:
        resnet18 = torchvision.models.resnet18(weights=weights)
        for param in resnet18.parameters():
            param.requires_grad = False
        return resnet18

# Trains the model 
def train_model(model, dataloaders, imageFolders, device, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                # Inputs: 4 x 3 x 128 x 128
                # Labels: 4
                for i, (inputs, labels) in enumerate(dataloaders[phase]):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Outputs is 4, 4
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                # Calculate epoch loss and accuracy
                epoch_loss = running_loss / len(imageFolders[phase])
                epoch_acc = running_corrects.double() / len(imageFolders[phase])

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Append the loss to the corresponding lists
                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    test_losses.append(epoch_loss)


                # deep copy the model and store best accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Validation Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, best_acc, train_losses, test_losses

# Plot graph of train and validation losses
def plotLosses(num_epochs, train_losses, test_losses, save_path=None):
    # Create a new figure
    plt.figure()
    # Plot the losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Validation')
    plt.title("Losses for Training and Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save path given then save the image
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved at {save_path}")
    else:
        plt.show()


# Visualize model performance on set of validation images
def visualize_model(model, dataloaders, imageFolders, device, num_images=6):
    # Set it to evaluation mode
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    # Forward pass of model on eval mode and plot images
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                class_names = imageFolders["test"].classes
                ax = plt.subplot(num_images//2, 2, images_so_far)
                plt.tight_layout()
                ax.axis('off')
                ax.set_title(f'pred: {class_names[preds[j]]},  act: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return
        model.train(mode=was_training)

# Obtain accuracy of best model on test set
def testPerformance(model, dataloader, device):
    model.eval()  # Set model to evaluate mode

    running_corrects = 0

    for inputs, labels in dataloader["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(dataloader["test"].dataset)

    return accuracy.item()




# Main function 
def main(argv):
     # Seeds used for repeatability
    random_seed = 42
    torch.backends.cudnn.enabled = False # type: ignore
    torch.manual_seed(random_seed)

    # Main training variables
    num_epochs = 50 # Number of Epochs
    lr = 0.0002 # Learning Rate
    momentum = 0.9 # Momentum
    batch_size = 4 # Batch size during training 
    image_size = 128 # Spatial size of training images. 
    gamma = 0.1
    device = "cpu"


    # Root directory for dataset 
    dataroot = "/Users/karanshah/Spring2023/CV/Karan/FinalProject/trainValTest"

    # Load the data
    dL, imageFolder = loadData(root=dataroot, batch_size=batch_size, image_size=image_size)

    # Load the models
    resnet18 = transferModels("resnet18", weights='IMAGENET1K_V1')
    resnet34 = transferModels("resnet34", weights='IMAGENET1K_V1' )
    resnet50 = transferModels("resnet50", weights='IMAGENET1K_V1')
    densenet121 = transferModels("densenet121", weights='IMAGENET1K_V1')


    ## RESNET18 ##

    # Change output layer
    numFtrs18 = resnet18.fc.in_features
    resnet18.fc = nn.Linear(numFtrs18, 4)
    resnet18 = resnet18.to(device)
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv_res18 = optim.SGD(resnet18.fc.parameters(), lr=lr, momentum=momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_conv_res18 = lr_scheduler.StepLR(optimizer_conv_res18, step_size=7, gamma=gamma)

    ## RESNET34 ## 

    # Change output layer
    numFtrs34 = resnet34.fc.in_features
    resnet34.fc = nn.Linear(numFtrs34, 4)
    resnet34 = resnet34.to(device)
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv_res34 = optim.SGD(resnet34.fc.parameters(), lr=lr, momentum=momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_conv_res34 = lr_scheduler.StepLR(optimizer_conv_res34, step_size=7, gamma=gamma)

    ## RESNET50 ##

    # Change output layer
    numFtrs50 = resnet50.fc.in_features
    resnet50.fc = nn.Linear(numFtrs50, 4)
    resnet50 = resnet50.to(device)
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv_res50 = optim.SGD(resnet50.fc.parameters(), lr=lr, momentum=momentum)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_conv_res50 = lr_scheduler.StepLR(optimizer_conv_res50, step_size=7, gamma=gamma)


    ## DENSENET121 ##

    # Change output layer 
    numFtrs121 = densenet121.classifier.in_features
    densenet121.classifier = nn.Linear(numFtrs121, 4)
    densenet121 = densenet121.to(device)
    # Observe that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv_dense121 = optim.SGD(densenet121.classifier.parameters(), lr=lr, momentum=momentum) 
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_conv_dense121 = lr_scheduler.StepLR(optimizer_conv_dense121, step_size=7, gamma=gamma)

    # Loss for all the models
    criterion = nn.CrossEntropyLoss()


    # lsOfClassNames = imageFolder["train"].classes
    # Make a grid from batch
    # out = torchvision.utils.make_grid(X)
    # imshow(out, title=[lsOfClassNames[name] for name in y])

    # Obain size of training and validation
    sizeOfTraining, sizeOfValidation = len(imageFolder["train"]), len(imageFolder["val"])

    print(f"Size of Training: {sizeOfTraining} ")
    print(f"Size of Validation: {sizeOfValidation} ")

    # Train the models and obtain the best accuracy per model, train losses, and test losses
    m_resnet18, resnet18Acc, resnet18TrainL, resnet18TestL = train_model(model=resnet18, dataloaders=dL, imageFolders=imageFolder, device=device, criterion=criterion, optimizer=optimizer_conv_res18, scheduler=exp_lr_scheduler_conv_res18, num_epochs=num_epochs)
    m_resnet34, resnet34Acc, resnet34TrainL, resnet34TestL = train_model(model=resnet34, dataloaders=dL, imageFolders=imageFolder, device=device, criterion=criterion, optimizer=optimizer_conv_res34, scheduler=exp_lr_scheduler_conv_res34, num_epochs=num_epochs)
    m_resnet50, resnet50Acc, resnet50TrainL, resnet50TestL = train_model(model=resnet50, dataloaders=dL, imageFolders=imageFolder, device=device, criterion=criterion, optimizer=optimizer_conv_res50, scheduler=exp_lr_scheduler_conv_res50, num_epochs=num_epochs)
    m_densenet121, densenet121Acc, densenet121TrainL, densenet121TestL = train_model(model=densenet121, dataloaders=dL, imageFolders=imageFolder, device=device, criterion=criterion, optimizer=optimizer_conv_dense121, scheduler=optimizer_conv_dense121, num_epochs=num_epochs)

    # Create directory and plot the losses
    os.makedirs("figures", exist_ok=True)
    plotLosses(num_epochs, resnet18TrainL, resnet18TestL, save_path="figures/resnet18.png")
    plotLosses(num_epochs, resnet34TrainL, resnet34TestL, save_path="figures/resnet34.png")
    plotLosses(num_epochs, resnet50TrainL, resnet50TestL, save_path="figures/resnet50.png")
    plotLosses(num_epochs, densenet121TrainL, densenet121TestL, save_path="figures/densenet121.png")

    # Obtain the best accuracy of all the models
    best_acc = max(resnet18Acc, resnet34Acc, resnet50Acc, densenet121Acc)

    # Store the model with best accuracy
    if best_acc == resnet18Acc:
        best_model = m_resnet18
        best_model_name = "Resnet18"
    elif best_acc == resnet34Acc:
        best_model = m_resnet34
        best_model_name = "Resnet34"
    elif best_acc == resnet50Acc:
        best_model = m_resnet50
        best_model_name = "Resnet50"
    else:
        best_model = m_densenet121
        best_model_name = "Densenet121"


    # Obtain the test accuracy
    testAccuracy = testPerformance(best_model, dL, device)
    print()
    print(f"Test Accuracy: {testAccuracy:.4f}")
    print(f"Best Model: {best_model_name}")
    print()

    # Create the target directory if it doesn't exist
    os.makedirs("newmodels", exist_ok=True)

    # Saves the models
    torch.save(m_resnet18, "newmodels/m_resnet18.pth")
    torch.save(m_resnet34, "newmodels/m_resnet34.pth")
    torch.save(m_resnet50, "newmodels/m_resnet50.pth")
    torch.save(m_densenet121, "newmodels/m_densenet121.pth")

    print("Saved PyTorch Models State to models folder")

    # Comment all lines above and uncomment to load the models after one succession of running the script obtaining the saved models
    # m_resnet18 = torch.load("models/m_resnet18.pth")
    # m_resnet34 = torch.load("models/m_resnet34.pth")
    # m_resnet50 = torch.load("models/m_resnet50.pth")
    # m_densenet121 = torch.load("models/m_densenet121.pth")

    # visualize_model(m_resnet18, dataloaders=dL, imageFolders=imageFolder, device=device, num_images=6)
    # visualize_model(m_resnet34, dataloaders=dL, imageFolders=imageFolder, device=device, num_images=6)
    # visualize_model(m_resnet50, dataloaders=dL, imageFolders=imageFolder, device=device, num_images=6)
    # visualize_model(m_densenet121, dataloaders=dL, imageFolders=imageFolder, device=device, num_images=6)

    return

if __name__ == "__main__":
    main(sys.argv)