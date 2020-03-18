# MUPS
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8)](https://github.com/y2l/meta-transfer-learning/tree/master/pytorch)

Code for MUPS model introduced in "Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification"


### Setting up environment:

   The model is implemented with Pytorch, we recommend python 3.5 and PyTorch 0.4.0 with Anaconda.
   
   Create a new environment and install python packages in it:
   
       conda create --name mups python=3.5
       conda activate mups
       conda install pytorch=0.4.0
       conda install scipy scikit-learn
   
   Install other dependencies:
   
       pip install tqdm tensorboardX
       
   Clone the repository:
   
       git clone https://github.com/tiehangd/MUPS
       
### Dataset preparation:

   Download BCI-IV 2a dataset from http://bnci-horizon-2020.eu/database/data-sets, Four class motor imagery (001-2014)
   
   Place the 18 files inside ./data folder
   
   Data preprocess, run from command line
   
       python ./dataloader/data_preprocessing.py
   
   This produces data for our cross subject task, which is stored in ./data/cross_sub
   
### Running the model:
   1) Pretraining of feature extractor
   
          python pre_train.py
     
   2) Meta adaptation 
   
          python meta_adapt.py
      
### Acknowledgements:

    Implementation of MUPS model utilized code from the following repositories:
    
    1) https://github.com/yaoyao-liu/meta-transfer-learning
    2) https://github.com/aliasvishnu/EEGNet
      
      
      
      
      
     
   
   
   


