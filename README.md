# MUPS-EEG
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8)](https://github.com/y2l/meta-transfer-learning/tree/master/pytorch)

Code for MUPS-EEG model introduced in ["Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification"](https://arxiv.org/pdf/2003.06113.pdf).


## Setting up environment

   The model is implemented with Pytorch, we recommend python 3.5 and PyTorch 0.4.0 with Anaconda.
   
   Create a new environment and install python packages in it:
   
       conda create --name mups python=3.5
       conda activate mups
       conda install pytorch=0.4.0 -c pytorch
       conda install scipy scikit-learn
   
   Install other dependencies:
   
       pip install tqdm tensorboardX
       
   Clone the repository:
   
       git clone https://github.com/tiehangd/MUPS-EEG
       
## Dataset preparation

   Download BCI-IV 2a dataset from http://bnci-horizon-2020.eu/database/data-sets, Four class motor imagery (001-2014)
   
   Place the 18 files inside ./data folder
   
   Data preprocess, run from command line
   
       python ./dataloader/data_preprocessing.py
   
   This produces data for our cross subject task, which is stored in ./data/cross_sub
   
## Running the model
   1) Pretraining of feature extractor
   
          python pre_train.py
     
   2) Meta adaptation 
   
          python meta_adapt.py
          
          
          
## Citation

Please cite our paper if it is helpful to your work:

```
@article{Duan2020,
  title={Ultra Efficient Transfer Learning with Meta Update for Cross Subject EEG Classification},
  author={Tiehang Duan and Mihir Chauhan and Mohammad Abuzar Shaikh and Jun Chu and Sargur N. Srihari},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.06113}
}
```


## Acknowledgements

  Implementation of MUPS model utilized code from the following repositories:
    
    1) https://github.com/yaoyao-liu/meta-transfer-learning
    2) https://github.com/aliasvishnu/EEGNet
      
      
      
      
      
     
   
   
   


