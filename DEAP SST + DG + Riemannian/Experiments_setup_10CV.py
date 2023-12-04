# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:39:46 2021

@author: ha242089
"""

import os
import random
import numpy as np
import torch
#from sklearn.model_selection import KFold
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader
#from My_Custom_Dataset import DEAP
# import sys
# import scipy.io


## Enable GPU if possible 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
#device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')

# Set the random number seed in all modules to guarantee the same result when running again.
def seed_fix(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train_intra(trainloader, net, criterion, epoch, optimizer_E, optimizer_R = None):
    
    running_loss = 0
    net.train()
    
    ACC = []
    for batch_idx, (batch_data,batch_label) in enumerate(trainloader, 0):
        
        Inputs = batch_data.to(device)            
        Targets = batch_label.unsqueeze(1).to(device)

        # zero the parameter gradients
        optimizer_E.zero_grad()
        #optimizer_R.zero_grad()
        
        #optimizer_R = None
    
        # forward + backward + optimize
        Outputs = net(Inputs)
        loss = criterion(Outputs, Targets)                                               
        loss.backward()
                          
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=False)
        optimizer_E.step()       
        #optimizer_R.step()
        
        running_loss += loss.data.item()
        
        
        Predictor_batch = np.round(Outputs.data.cpu().numpy())
                                   
        #print(Predictor_batch)                            
        #print(Targets.data.cpu().numpy())
        
        comparsion = Predictor_batch == Targets.data.cpu().numpy()
        accuracy = comparsion.sum()/np.size(comparsion,0)  
        ACC.append(accuracy)
        
    if (epoch+1) % 10 == 0:                       
        #print('Epoch %d : training loss %f' % (epoch+1,running_loss))  
        print('Epoch %d : train loss %f, train accuracy %f' % (epoch+1,running_loss, np.mean(ACC)))  
        
        
    return net
        
def test_intra(testloader, net, criterion, epoch):
    
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
               
         for batch_idx, (batch_data,batch_label) in enumerate(testloader, 0):
                                                  
                Inputs = batch_data.to(device)            
                Targets = batch_label.unsqueeze(1).to(device)
                # only forward and generate outputs
                Outputs = net(Inputs)
                loss = criterion(Outputs, Targets) 

                #print(loss)
                running_loss += loss.data.item()                                                 
                
                Predictor_batch = np.round(Outputs.data.cpu().numpy())
                                           
                #print(Predictor_batch)                            
                #print(Targets.data.cpu().numpy())
                
                comparsion = Predictor_batch == Targets.data.cpu().numpy()
                accuracy = comparsion.sum()/np.size(comparsion,0)                       
         
         if (epoch+1) % 10 == 0:                       
                print('Epoch %d : test loss %f, test accuracy %f' % (epoch+1,running_loss, accuracy))  
                
         return Predictor_batch,accuracy
    
        

      
# def Intra_train_test(batch_size_train, batch_size_test, lr, training_epoch):    
      
#     n_subject = 32 
#     n_trial = 40
#     n_fold = 10   
#     Augment = 1
    
    
#     Predict_fold_subject = np.zeros((n_trial,n_subject))
#     #F1_subject = np.zeros((n_fold,n_subject))
#     Accuracy_fold_subject = np.zeros((n_fold,n_subject))
           
#     for Index_subject in range(n_subject):
        
#         print('--------------------------------')  
#         dataset = my_Graph_DEAP(Index_subject)        
#         kfold = KFold(n_splits = n_fold, shuffle=False)
                
#         for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            
#                 print(f'FOLD {fold}')
#                 print('--------------------------------')
#                 net = my_GCN_spectral().to(device) # my_GCN_spectral().to(device)    #my_GCN_message_passing().to(device)   
#                 net.train()
                
#                 criterion = nn.BCELoss()
#                 optimizer = optim.Adam(net.parameters(),lr)            
                                                                
#                 # Sample elements randomly from a given list of ids, no replacement.
#                 train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#                 test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
                
#                 # Define data loaders for training, fin tuning and testing data in one fold
                
#                 trainloader = DataLoader(dataset, batch_size = batch_size_train, sampler = train_subsampler)                        
#                 testloader = DataLoader(dataset, batch_size = batch_size_test, sampler = test_subsampler)   
              
#                 for epoch in range(training_epoch):
                                   
#                     running_loss = 0.0
                    
#                     for batch_idx, batch_data in enumerate(trainloader, 0):
                        
#                         Inputs = batch_data.x.to(device);
#                         Targets = batch_data.y.unsqueeze(1).to(device);
#                         edge_index = batch_data.edge_index.to(device);                   
#                         batch_inf =  batch_data.batch.to(device);
#                         # zero the parameter gradients
#                         optimizer.zero_grad()
                    
#                         # forward + backward + optimize
#                         Outputs = net(Inputs,edge_index,batch_inf)
#                         loss = criterion(Outputs, Targets)                                               
#                         loss.backward()
                                          
#                         #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=False)
#                         optimizer.step()
                        
#                         running_loss += loss.data.item()
                    
                        
#                     if epoch % 25 == 0:                       
#                         print('Epoch %d : training loss %f' % (epoch,running_loss/4))   
                                                                           
#                 # Evaluation for this fold  
#                 print('All Training epochs have been finished. Starting testing')  
                
#                 net.eval()
#                 with torch.no_grad():
                    
#                      running_loss = 0.0
                    
#                      for batch_idx, batch_data in enumerate(testloader, 0):
                                                  
#                             Inputs = batch_data.x.to(device);
#                             Targets = batch_data.y.unsqueeze(1).to(device);
#                             edge_index = batch_data.edge_index.to(device);   
#                             batch_inf =  batch_data.batch.to(device);
                            
#                             # only forward and generate outputs
#                             Outputs = net(Inputs,edge_index,batch_inf)
#                             loss = criterion(Outputs, Targets)                       
#                             running_loss += loss.data.item()                                                       
#                             print('test loss %f' % (running_loss))
                            
#                             Predictor_batch = np.round(Outputs.data.cpu().numpy())
                                                       
#                             #print(Predictor_batch)                            
#                             #print(Targets.data.cpu().numpy())
                            
#                             comparsion = Predictor_batch == Targets.data.cpu().numpy()
#                             accuracy = comparsion.sum()/np.size(comparsion,0)                       
#                             Accuracy_fold_subject[fold,Index_subject] = accuracy
                                                       
#                             Predict_fold_subject[fold*batch_size_test : fold*batch_size_test + batch_size_test, Index_subject] = Predictor_batch.reshape((-1,))
#                             #print('Testing loss %f' % (running_loss))   
#         # Evaluation for this subject
#         mean = np.mean(Accuracy_fold_subject[:,Index_subject],axis = 0)
#         print('subject %d, accuracy %d %%' % (Index_subject+1,100.0 * mean))   
#         print('--------------------------------')  

            
#     return Predict_fold_subject, Accuracy_fold_subject


# batch_size_train = 9
# batch_size_test = 4
# lr =  0.01
# training_epoch = 200
            
# Predict_fold_subject, Accuracy_fold_subject = Intra_train_test(batch_size_train, batch_size_test, lr, training_epoch)   

# Accuracy_subject_sequence = np.mean(Accuracy_fold_subject,axis = 0)

# average = np.mean(Accuracy_subject_sequence)
# std = np.std(Accuracy_subject_sequence)  

# print('Accuracy is %f %%, Standard Deviation is %f %%, ' % (100.0 * average,100*std))   
        
# Dict = {'Predictor': Predict_fold_subject, 'Accuracy': Accuracy_fold_subject }

#scipy.io.savemat('Pred_DEAP_arousal_structure_Cheb.mat',Dict)  
#scipy.io.savemat('Pred_DEAP_arousal_random_Cheb.mat',Dict)  
#scipy.io.savemat('Pred_DEAP_arousal_identity_MP.mat',Dict)    
#scipy.io.savemat('Pred_DEAP_valance_coherence.mat',Dict)     
#scipy.io.savemat('Pred_DEAP_valance_gaussian.mat',Dict)        

             
             
             
             
             
             
             
             
             
             
             
             
             