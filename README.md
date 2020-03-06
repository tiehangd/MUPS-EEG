# MUPS

Code for MUPS model introduced in "Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification"

1) Setting up environment:
   The model is implemented with Pytorch, we recommend python 3.5 and PyTorch 0.4.0 with Anaconda.
   
   Create a new environment and install PyTorch on it:
   
       conda create --name mups python=3.5
       conda activate mups
       conda install pytorch=0.4.0
   
   Install necessary python packages:
   
       pip install tqdm tensorboardX
       
   Clone the repository:
   
       git clone https://github.com/tiehangd/MUPS
       
2) Dataset preparation:
   Download BCI-IV 2a dataset from http://bnci-horizon-2020.eu/database/data-sets, Four class motor imagery (001-2014)
   
   Place the 18 files inside ./data folder
   
   Data preprocess, run from command line
   
       python ./dataloader/data_preprocessing.py
   
   This produces data for our cross subject task, which is stored in ./data/cross_sub
   
3) Running the model:
   1) Pretraining of feature extractor
   
          python pre_train.py
     
   2) Meta adaptation 
   
          python meta_adapt.py
      
4) Acknowledgements:
    Our project utilized code from the following repositories:
    
    1) https://github.com/yaoyao-liu/meta-transfer-learning
    2) https://github.com/aliasvishnu/EEGNet
      
      
      
      
      
     
   
   
   


