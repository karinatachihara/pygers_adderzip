#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:19:17 2022

@author: Work
"""

import csv
import numpy as np

path  = '/Volumes/norman/karina/adderzip_fMRI/adderzip'

for eachParti in range(5,25):
    
    partiData = np.zeros((12,2))
    
    partiData[:,0] = np.arange(1,13)
    
    if eachParti <10: 
        sub = '00' + str(eachParti)
    else:
        sub = '0' + str(eachParti)
    
    for eachRun in range (1,13):
    
        if eachRun == 1:
            task_name = 'exposure'
        if (eachRun == 2) or (eachRun == 3) or (eachRun == 6) or (eachRun == 7):
            task_name = 'forcedChoice'
        if (eachRun == 4) or (eachRun == 5) or (eachRun == 8) or (eachRun == 9):
            task_name = 'imagine'
        if eachRun >= 10:
            task_name = 'localizer'
        
        file_name = path + '/data/behavioral/data_10_15_21/'+task_name+'Data/'+str(eachRun)+'/logs/'+task_name+'_log_'+sub+'.txt'
    
        #get file object reference to the file
        file = open(file_name, "r")
        
        #read content of file to string
        data = file.read()
        
        #get number of occurrences of the substring in the string
        n_trig = data.count("Trigger received - post trials")
        
        #print('Number of triggers :', occurrences)
        
        partiData[eachRun-1,1] = n_trig

    np.savetxt(path+'/data/behavioral/data_10_15_21/n_trunc_end/n_trunc_end_'+sub+'.csv', partiData, fmt='%f', delimiter=",", header = 'run,triggerCount')
