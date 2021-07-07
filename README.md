# MetaHG
Distilling Meta Knowledge on Heterogeneous Graph for Illicit Drug Trafficker Detection on Social Media
====
This repository includes the source code for the paper above.

## Requirements

This code package was developed and tested with python 3.7.6. Please make sure all dependencies specified in the ```requirements.txt``` file are satisfied before running the model. This can be achieved by
```
pip install -r requirements.txt
```

## Usage

This source code contains two parts, meta learning source code and representation learning source code.

We can run the code with GSL (graph structure learning), SSL (self-supervised Learning), RGCN (relation graph convolutional network) and Meta-Learning with the scripts ```main.py```.

We can run the knowledge distillation based meta-learning code with the scripts ```kd_meta.py```.


## Data
Due to privacy issues, all of the data we collected and utilized for this project cannot be public accessed for the time being. 


