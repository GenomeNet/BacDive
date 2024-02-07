# BacDive phenotype prediction

To train a deep learning model for phenotype prediction, we take samples from fasta files and assign 
labels to the sequence based on a table with lable annotations.  

<img src="images/workflow.png" alt="Workflow for BacDive training" width="600" height="auto">


# Model architectures

Summary of a model architecture for spore training:

<img src="images/spore_model.png" alt="Model architecture for spore training" width="200" height="auto">


We use very similar architectures for cell size, pathogenicity and oxygen requirements training.
Depending on the task, we have to adjust the loss function and activation of the output layer.

Models can also have multiple output layers, i.e. the model tries to predict several features at once.
For the morphology prediction, the model predicts cell shape, flagellum, gram staining and motility. 

<img src="images/morph_model.png" alt="Model architecture for morphology training (multi task training)" width="400" height="auto">

