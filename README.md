# CS6423_knowledge_distillation_project
Group project for CS6423: Scalable Computing for Data Analysis; project centered on Knowledge Distillation and pruning applied to a trained CT Lung Nodule Detection model.

# RadImage Models

For downloading the pretrained weights follow the start of `radimage_model_download.ipynb`

For downloading the dataset follow `radimage_data_download.ipynb`

# Dataset and Trained Models

These are both located on the HuggingFace hub. I have also included the code on how these were originally downloaded and trained for reference.

[Dataset Link](https://huggingface.co/datasets/fiabar/cs6423_datasets)
[Models Link](https://huggingface.co/fiabar/cs6423_model_repo)

# Modules

The `modules/` directory contains several helper classes for preprocessing, training and evaluating models. For example usage of these see the `examples/` directory.

# Running On GPU Server

First, to clone this repo onto the server, create a new notebook with the following cell and run it:
```
!git clone https://github.com/eoconn25/CS6423_knowledge_distillation_project
```

NOTE: If you find you are getting import errors or missing paths you may have to add the following cell (if it doesn't exist already) to the top of each notebook to ensure that all downloads and data loading is done correctly (relative to current working directory not kernel root). Basically you want every file to run from the location of the file. If you are getting errors you can always modify the paths to point to the correct location.

```
import os

target_folder = "https://github.com/eoconn25/CS6423_knowledge_distillation_project" 
path = os.path.join(os.getcwd(), target_folder)

if os.path.exists(path):
    os.chdir(path)

# should match the folder you cloned into
print(f"Current working directory: {os.getcwd()}")
```

## Instructions

1. Start with `radimage_data_download.ipynb`
2. Then run `radimage_model_download.ipynb`
3. Then you are ready for training/evaluation !!
