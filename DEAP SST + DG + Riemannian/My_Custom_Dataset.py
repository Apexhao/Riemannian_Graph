 # -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:41:33 2023

@author: ha242089
"""


import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io



class DEAP(Dataset):
     def __init__(self, Index_subject, transform = None, pre_transform = None):
        #super().__init__(Index_subject, transform, pre_transform)
        
        self.Index_subject = Index_subject
        
        self.num_trials = 600  # Augment * number_videos = 15*40 = 600
        self.num_nodes = 32    # self.num_nodes = 32      
        self.num_features = 41
                                        
        ## the following code is for x(data) spectrum extraction from .mat, matlab EEG image data computed by brainstorm and spectrum is calculated from Pwelch methods        
        filename_1 = 'C:/Users/ha242089/Dropbox/A. Journal/2024 GNN/Data/DEAP/EEG/PSD_Aug/S_' + str(Index_subject+1) + '.mat'
        
        mat_content = scipy.io.loadmat(filename_1)
        temp_data = mat_content['filtered_data']    ## self.num_trials x self.num_nodes x 41      
        temp_label = mat_content['Label']    ## self.num_trials x 4    Valance - Arousal - Like - Dominance                                       
        
        #data = np.expand_dims(temp_data,1)
        data = temp_data                     
        label = temp_label[:,0]  
               
        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        
        self.data = data
        self.label = label
        
     def __info__(self):
           
             return self.num_trials, self.num_nodes, self.num_features
        
     def __len__(self):
        
          return len(self.label)

     def __getitem__(self,index):
      
          return self.data[index,:,:], self.label[index]
        
      