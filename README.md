# MUPS

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
   
   This produces data for our cross subject classification task.
   
   
   
   


