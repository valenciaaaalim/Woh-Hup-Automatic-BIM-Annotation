
# FOR SPATIAL DESIGN STUDIO 
1. Create your environment and interpreter
2. Run ``` pip install -r requirements.txt ```

## Information about this repo
1. ```ExtractGraph.py``` converts the ```.txt``` file from ```1.Input``` into ```.graphml``` in ```2.OutputGraph```.
2. ```ServerMLviaGAT.py``` and ```ServerMLviaGCN.py``` are scripts that train a GAT and GCN model respectively by taking in graph models from ```3.InputML/Train data``` (but there is currently no Train data folder, it is pointed to 3.Server Sample Projects and 2.Sample Projects by floor instead), and ```3.InputML/4.Test data``` and ```3.InputML/5.Validation data```. It will output the evaluation metrics of its own model in ```4.OutputML_GCN``` and ```5.OutputML_GAT``` respectively.
3. ```visualize_graph.py``` supposed to show you what the graph looks like but I dont see anything so far

## How to run and what to run
1. Clear all data in 1.,2.,3.,4.,5. Your directory should look like:
```
ROOT
└── Automated-BIM-Model-Annotation-via-Graph-Neural-Networks-Model-Architecture
    └── 1.Input # Raw extracted BIM data (text format)
    └── 2.OutputGraph # Processed graph outputs from extraction 
    └── 3.InputML # Graph datasets for model training/testing 
        ├── Train data
        ├── Test data
        └── Validation data
    └── 4.OutputML_GCN/ # Output results from GCN model
    └── 5.OutputML_GAT/ # Output results from GAT model
    └── ExtractGraph.py
    └── ServerMLviaGAT.py
    └── ServerMLviaGCN.py
    └── spatial_queries.py
    └── visualize_graph.py
└── README.md
└── requirements.txt
```

1. Upload your text files into ```1.Input``` folder
2. Change the directory of your input folder in ```ExtractGraph.py``` to the ```1.Input``` folder.
3. Create and run a new python script to split the files in ```1.Input``` into ```3.InputML/Train data```, ```3.InputML/Test data```, and ```3.InputML/Validation data```.
4. Change the directory of your input folder in ```ServerMLviaGAT.py```, and ```ServerMLviaGCN.py``` to the TRAIN, TEST and VALIDATION folders respectively in ```3.InputML```. Output directories should still be the same.
2. ```cd Automated-BIM-Model-Annotation-via-Graph-Neural-Networks-Model-Architecture ```
2. ```python ExtractGraphy.py```
3. ```python ServerMLviaGAT.py```
4. ```python ServerMLviaGCN.py```

# Cloned repo instructions below

# Reminder: Organize hard-coded file paths before running to ensure all necessary files are generated 
# and to review the final results.

# Automated BIM Model Annotation via Graph Neural Networks (GNN) - Python

This component of the project applies Graph Neural Networks (GNN) to train  Building Information Modeling (BIM) data. It processes input files to construct graph representations of BIM elements, which are then used to train and infer the GNN models.

## Description

The pipeline processes input files to generate node and edge data suitable for GNNs. 

## Graph Generation and Preprocessing

Graphs are generated from input files using the `parse_graphml_to_pyg` function, converting them into PyTorch Geometric Data format. This ensures proper structuring of node features and edge connections for GNN training.

## Model Architecture

Our GNN model, implemented in the `GCNModel` class, uses two layers of GCNConv, designed for processing graph-structured data. The model takes node features as input and produces annotations for BIM elements as output.

## Training Procedure

Training involves the following steps:

1. Splitting the data into training, validation, and testing sets.
2. Initializing the GCN model with specified input and output dimensions.
3. Training the model across several epochs and evaluating against a validation set.
4. Employing early stopping to mitigate overfitting.
5. Conducting final evaluations on a test set.

## Installation

Please ensure the following dependencies are installed:

- PyTorch
- PyTorch Geometric
- NetworkX
- Scikit-learn
- Pandas
- Matplotlib (for visualization)

Install the necessary Python packages using pip:

```sh
pip install torch torch-geometric networkx scikit-learn pandas matplotlib

