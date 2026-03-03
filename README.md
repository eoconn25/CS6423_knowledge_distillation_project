# CS6423_knowledge_distillation_project
Group project for CS6423: Scalable Computing for Data Analysis; project centered on Knowledge Distillation and pruning applied to a trained CT Lung Nodule Detection model.

# RadImage Models

For downloading the pretrained weights follow the start of `radimage_model_download.ipynb`

For downloading the dataset follow `radimage_data_download.ipynb`

# Modules

The `modules/` directory contains several helper classes for preprocessing, training and evaluating models. For example usage of these see the `examples/` directory.

# Running On GPU Server

First, to clone this repo onto the server, create a new notebook with the following cell and run it:
```
!git clone https://github.com/eoconn25/CS6423_knowledge_distillation_project
```

NOTE: You may have to add the following cell (if it doesn't exist already) to the top of each notebook to ensure that all downloads and data loading is done correctly (relative to current working directory not kernel root)

```
import os

# Get the absolute path of the current notebook
module_path = os.path.abspath(os.curdir)
os.chdir(module_path)

print(f"Current working directory: {os.getcwd()}")  # should be the same folder you want everything to be downloaded to
```

1. Start with `radimage_data_download.ipynb`
2. Then run `radimage_model_download.ipynb`
3. Then you are ready for training/evaluation !!
