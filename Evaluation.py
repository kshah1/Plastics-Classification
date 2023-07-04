# Karan Shah
# Evaluation of the models

# Import statements
import sys
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import precision_recall_curve

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


# main function (yes, it needs a comment too)
def main(argv):
    # Main Variables 
    batch_size = 4
    image_size = 128
    # Seeds used for repeatability
    random_seed = 42
    torch.backends.cudnn.enabled = False # type: ignore
    torch.manual_seed(random_seed)

    # Root directory for dataset
    dataroot = "/Users/karanshah/Spring2023/CV/Karan/FinalProject/trainValTest"
    # Load the data
    dL, imageFolder = loadData(root=dataroot, batch_size=batch_size, image_size=image_size)

    # Load the models
    m_resnet18 = torch.load("newmodels/m_resnet18.pth")
    m_resnet34 = torch.load("newmodels/m_resnet34.pth")
    m_resnet50 = torch.load("newmodels/m_resnet50.pth")
    m_densenet121 = torch.load("newmodels/m_densenet121.pth")

    # Build list of models
    models = [m_resnet18, m_resnet34, m_resnet50, m_densenet121]
    modelNames = ["resnet18", "resnet34", "resnet50", "densenet121"]

    # Set all models to evaluate
    m_resnet18.eval()
    m_resnet34.eval()
    m_resnet50.eval()
    m_densenet121.eval()

    # Get the class labels
    class_labels = imageFolder["val"].classes
    
    # Define lists to store true labels and predicted probabilities for each model
    trueLabels = []
    predProb = {model: {class_label: [] for class_label in class_labels
                        
                        }
                        
                        for model in models
                }
    
    # Iterate over the test dataset and make predictions
    with torch.no_grad():
        for images, labels in dL["val"]:
            # Store the true labels
            trueLabels.extend(labels.tolist())

             # Make predictions using each model
            outputs1 = m_resnet18(images)
            outputs2 = m_resnet34(images)
            outputs3 = m_resnet50(images)
            outputs4 = m_densenet121(images)

         # Get the predicted probabilities for each model and each class
            for i, class_label in enumerate(class_labels):
                probabilities1 = torch.softmax(outputs1[:, i], dim=0).tolist()
                probabilities2 = torch.softmax(outputs2[:, i], dim=0).tolist()
                probabilities3 = torch.softmax(outputs3[:, i], dim=0).tolist()
                probabilities4 = torch.softmax(outputs4[:, i], dim=0).tolist()
            
                predProb[m_resnet18][class_label].extend(probabilities1)
                predProb[m_resnet34][class_label].extend(probabilities2)
                predProb[m_resnet50][class_label].extend(probabilities3)
                predProb[m_densenet121][class_label].extend(probabilities4)

    # Compute precision and recall for each class and each model
    precision = {model: {class_label: [] for class_label in class_labels} for model in models}
    recall = {model: {class_label: [] for class_label in class_labels} for model in models}

    # Obtain the precision and recall for the set of probabilities and store
    for model in models:
        for class_label in class_labels:
            true_class_labels = np.array(trueLabels) == class_labels.index(class_label)
            probabilities = np.array(predProb[model][class_label])
            p, r, _ = precision_recall_curve(true_class_labels, probabilities)
            precision[model][class_label].extend(p)
            recall[model][class_label].extend(r)

    # Plot the precision-recall curves for each model and each class
    for class_label in class_labels:
        plt.figure()
        for modelName, model in zip(modelNames, models):
            p = precision[model][class_label]
            r = recall[model][class_label]
            plt.plot(r, p, label=modelName)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve - ' + class_label)
        plt.legend(loc='lower left')
        plt.savefig("figures/" + "PRCurve_" + class_label + ".png")
        print("Figure saved at ", "figures/" + "PRCurve_" + class_label + ".png")
        plt.show()
 

    return

if __name__ == "__main__":
    main(sys.argv)