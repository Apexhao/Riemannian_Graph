# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 13:55:11 2023

@author: ha242089

Main run 
"""

import torch
import numpy as np
import sys
import scipy.io
from Experiments_setup_10CV import seed_fix, train_intra, test_intra
from My_Custom_Dataset import DEAP
from My_Custom_GNN import DGSST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import mctorch.nn as mnn
import mctorch.optim as moptim

sys.path.append('C:/Users/ha242089/Dropbox/A. Journal/2024 GNN/Intra Subjects/DEAP SST + DG + Riemannian/')

## Enable GPU if possible 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
criterion = nn.BCELoss()
lr_E =  0.001
lr_R =  0.01
L2_lambda = 0.01
training_epoch = 100
batch_size_train = 30
batch_size_test = 60

n_subject = 32
n_trial = 600
n_fold = 10  
Augment = 15

seed_fix(3407)

Predict_fold_subject = np.zeros((n_trial,n_subject))
Accuracy_fold_subject = np.zeros((n_fold,n_subject))

       
for Index_subject in range(n_subject):
    
    print('--------------------------------')  
    dataset = DEAP(Index_subject)        
    kfold = KFold(n_splits = n_fold, shuffle = False)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
            print(f'FOLD {fold}')
            print('--------------------------------')
            net = DGSST(net_type = 'R').to(device) # my_GCN_spectral().to(device)    
            
            
            # A. only use Euclidean optimizer for net parameters
            # optimizer_E = optim.Adam(params = net.parameters(),
            #                         lr = lr_E, weight_decay = L2_lambda)  
            
            # B. Initial and define Euclidean optimizer for net parameters <-> Riemannian optimizer for functional connectivity, also this [list(net.parameters())[0]]
            optimizer_E = optim.SGD(params = list(net.layer1.parameters()) + list(net.BN1.parameters()) + list(net.fc1.parameters()) + list(net.fc2.parameters()),
                                    lr = lr_E, weight_decay = L2_lambda)  
            optimizer_R = moptim.rAdagrad(params = [net.Connectivity],  
                                          lr = lr_R, weight_decay = L2_lambda)
            
                                                                       
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # Define dataloaders for train and test
            
            trainloader = DataLoader(dataset, batch_size = batch_size_train, sampler = train_subsampler)                        
            testloader = DataLoader(dataset, batch_size = batch_size_test, sampler = test_subsampler) 
            
            for epoch in range(training_epoch):
                
                net = train_intra(trainloader, net, criterion, epoch, optimizer_E, None)
                Predictor, accuracy = test_intra(testloader, net, criterion, epoch)
                
            Accuracy_fold_subject[fold,Index_subject] = accuracy                                       
            Predict_fold_subject[fold*batch_size_test : fold*batch_size_test + batch_size_test, Index_subject] = Predictor.reshape((-1,))
            
    # Evaluation for this subject
    mean = np.mean(Accuracy_fold_subject[:,Index_subject],axis = 0)
    print('subject %d, accuracy %d %%' % (Index_subject+1,100.0 * mean))   
    print('--------------------------------')  
    
Accuracy_subject_sequence = np.mean(Accuracy_fold_subject,axis = 0)
average = np.mean(Accuracy_subject_sequence)
std = np.std(Accuracy_subject_sequence)  

print('Accuracy is %f %%, Standard Deviation is %f %%, ' % (100.0 * average,100*std))   
Dict = {'Predictor': Predict_fold_subject, 'Accuracy': Accuracy_fold_subject }


#scipy.io.savemat('Pred_DEAP_EEG_Aug_arousal_DGCNN_Cheb_1.mat',Dict)  


scipy.io.savemat('Pred_DEAP_EEG_Aug_valance_DGCNN_Cheb_1.mat',Dict)  



                