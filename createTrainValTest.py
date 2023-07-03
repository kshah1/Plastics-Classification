# Karan Shah
# Utilization to create the appropriate train, val, test folder structure

# Import statements
import os
import shutil
import random
import sys
import numpy as np
from PIL import Image

# Create train, validation, and test data
def createTrainValTest(source_dir, target_dir, train_dir, val_dir, test_dir):
    if os.path.exists(target_dir):
        print(f"{target_dir} exists!")
        return
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Define class names and corresponding counts
    classNames = []
    imageCounts = []
    imgFiles = []
    
    # Iterate over the subdirectories in the source directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        
        if os.path.isdir(subdir_path):
             # Get the files in the subdirectory
            try:
                image_files = [f for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                image_files = np.array(image_files)
            except NotADirectoryError:
                print(f"Skipping {subdir_path} as it is not a directory")
                continue
            
            print(f"Number of image files in {subdir_path} : {len(image_files)}")

            classNames.append(subdir)
            imageCounts.append(len(image_files))
            imgFiles.append(image_files)

    
    # Define train, validation, and test ratios (e.g., 70%, 20%, 10%)
    train_ratio = 0.70
    val_ratio = 0.20
    test_ratio = 0.10

    # Initialize empty lists for train, validation, and test sets
    train_set = []
    val_set = []
    test_set = []

    # Create np arrays
    classNames = np.array(classNames)
    imageCounts = np.array(imageCounts)
    imgFiles = np.array(imgFiles)

    # Iterate over each class
    for idx, (class_name, count) in enumerate(zip(classNames, imageCounts)):
        # Create a list of class labels
        labels = [class_name] * count

        # Create a list of shuffled image indices
        indices = list(range(count))
        random.shuffle(indices)

        # Determine the number of images for train, validation, and test sets
        train_count = int(train_ratio * count)
        val_count = int(val_ratio * count)
        test_count = count - train_count - val_count

        # Slice the shuffled image indices to obtain train, validation, and test sets
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]


        # Create the directories if it doesn't exist
        trainDirPerClass = os.path.join(target_dir, train_dir, class_name)
        valDirPerClass = os.path.join(target_dir, val_dir, class_name)
        testDirPerClass = os.path.join(target_dir, test_dir, class_name)

        os.makedirs(trainDirPerClass, exist_ok=True)
        os.makedirs(valDirPerClass, exist_ok=True)
        os.makedirs(testDirPerClass, exist_ok=True) 
        
        # Copy over the paths from source to target 
        for img_path in imgFiles[idx][train_indices]:
            src_path = os.path.join(source_dir, class_name, img_path)
            dst_path = os.path.join(trainDirPerClass, img_path)
            shutil.copyfile(src_path, dst_path)

        for img_path in imgFiles[idx][val_indices]:
            src_path = os.path.join(source_dir, class_name, img_path)
            dst_path = os.path.join(valDirPerClass, img_path)
            shutil.copyfile(src_path, dst_path)

        for img_path in imgFiles[idx][test_indices]:
            src_path = os.path.join(source_dir, class_name, img_path)
            dst_path = os.path.join(testDirPerClass, img_path)
            shutil.copyfile(src_path, dst_path)

    print("Finished.")

# main function (yes, it needs a comment too)
def main(argv):
    source_directory = argv[1] # /Users/karanshah/Spring2023/CV/Karan/FinalProject/Plastics\ Classification

    target_directory = argv[2] # /Users/karanshah/Spring2023/CV/Karan/FinalProject/trainValTest

    # Create target directory with train, val, test
    createTrainValTest(source_directory, target_directory, "train", "val", "test")

    return


if __name__ == "__main__":
    main(sys.argv)