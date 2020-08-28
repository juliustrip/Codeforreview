#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:53:44 2019

@author:  Dian, Aubert-Kato

This is a smart shoes program.
Start from reading the calibrated data
Filtration, normalizing, window cutting, feature extraction, ML program
"""

import scipy
from scipy import signal
import scipy.signal as sig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import heapq
import time
import seaborn as sns
from multiprocessing import Pool
import psutil
import sys
import os
import glob


    
'''Identify functions'''
def instepfeature(window,availableSensors):
    "The average difference in one step for all steps in a window"
    "...."
    def stepcounting(x):
        "sumyl dosen't mean any about LEFT, it is just a sum of halfwindow, it could be L or R"
        A = x.max(axis = 1)
        A[A<1] = 0       
        count = 1
        start = 0
        step = []
        for i in range(len(A)-1):
            if A[i] == 0:
                if A[i+1] == 0:
                    continue
                else:
                    count = count + 1
                    start = i
            else:
                if A[i+1] != 0:
                    continue
                else:
                    stop = i
                    t = stop - start
                    if A[start] == 0:
                            if t > 10:
                                step.append(x[start:stop+1])    
        return step
    
    def avepeakdiff(step):
        if step != []:
            peakdiff = []
            for each in step:
                maxinstep = each.max(axis=1)                
                peaks, properties = scipy.signal.find_peaks(maxinstep,distance = 10, width = 5)
                peaknum = len(peaks)
                if peaknum > 1:
                    peakdiff.append(maxinstep[peaks[peaknum-1]]-maxinstep[peaks[0]])
                else:
                    peakdiff.append(0)
        else:
            peakdiff = 0
        return np.mean(peakdiff)

    avL = availableSensors[[i in list(range(7)) for i in availableSensors]]
    avR = availableSensors[[i in list(range(7,14)) for i in availableSensors]]
    yl = np.array(window[:,avL])
    yr = np.array(window[:,avR])
    Lstep = stepcounting(yl)
    Rstep = stepcounting(yr)
    if (avepeakdiff(Lstep)+avepeakdiff(Rstep))/2  :
        inst = (avepeakdiff(Lstep)+avepeakdiff(Rstep))/2  
    else:
        inst = 0
    return inst   

def inerpeakinfo(window):
    "Peak intervals"
    def inersensorpeakinfo(single):
        isf = [] 
        peaks, properties = scipy.signal.find_peaks(single, distance = 30,prominence=2*np.std(single),width=0,rel_height=0.7)
        "Total # of peaks"
        ft_pk_no = len(peaks)
        if ft_pk_no <= 0:
            return [0,0,0,0,0,0,0]
        
        "Average/std of magnitudes"
        ft_mg_ave = np.average(properties['prominences'])
        ft_mg_std = np.std(properties['prominences'])
        "Average/std of width"
        ft_wd_ave = np.average(properties['widths'])
        ft_wd_std = np.std(properties['widths']) 
        ft_mg_ave = np.average(properties['prominences'])

        if ft_pk_no > 1:
            "Average distans between peaks"
            ft_pk2pk_dst = np.average(np.diff(peaks))
            ft_pk2pk_std = np.std(np.diff(peaks))
            isf = [ft_pk_no,ft_pk2pk_dst,ft_pk2pk_std,ft_mg_ave,ft_mg_std,ft_wd_ave,ft_wd_std]
        elif ft_pk_no == 1:
            isf = [ft_pk_no,0,0,ft_mg_ave,ft_mg_std,ft_wd_ave,ft_wd_std]
        else:
            isf = [0,0,0,0,0,0,0]
        return isf
    inerpeakinfo = []
    for column in window.T:
        inerpeakinfo.append(inersensorpeakinfo(column))
    return inerpeakinfo

def sensorgeneral (window):
    std = []
    ave = []
    mid = []
    maximum = []
    for columns in window.T:
        y = list(filter(lambda a: a != 0.0, columns)) #Ramove zeros 
        if len(y) == 0:
            y = [0]
#        y = columns
        ave.append(np.mean(y))
        std.append(np.std(y))  
        mid.append(np.median(y))  
        maximum.append(np.max(y))
    return [ave,std,mid,maximum]

'''The difference Anterior & Posterior mean of max in one step'''
def anterposterNAKDR(window,availableSensors):
    avL = availableSensors[[i in [0,2,3,4] for i in availableSensors]]
    avR = availableSensors[[i in [7,9,10,11] for i in availableSensors]]
    if (avL.size == 0 or not 6 in availableSensors) and (avR.size ==0 or not 13 in availableSensors):
        return ()
    if (avL.size > 0 and 6 in availableSensors):
        anterL = window[:,avL].max(axis=1) 
        posterL = window[:,6]
        diffl = np.mean(anterL)-np.mean(posterL)
        correlationL = np.corrcoef(anterL,posterL)[0][1]
    else:
        diffl = 0
        correlationL = 0
        
    if (avR.size >0 and 13 in availableSensors):
        anterR = window[:,avR].max(axis=1) 
        posterR = window[:,13]
        diffr = np.mean(anterR)-np.mean(posterR)
        correlationR = np.corrcoef(anterR,posterR)[0][1]
    else:
        diffr = diffl
        correlationR = 0
    
    avediff = (diffl+diffr)/2 if diffl > 0 else diffr
    return (avediff,correlationL,correlationR)

'''The difference latterior & middle mean of max in one step'''
def lattomidNAKDR(window,availableSensors):    
    avL = availableSensors[[i in [4,5] for i in availableSensors]]
    avR = availableSensors[[i in [11,12] for i in availableSensors]]
    if (avL.size == 0 or not 2 in availableSensors) and (avR.size ==0 or not 9 in availableSensors):
        return ()
    if (avL.size > 0 and 2 in availableSensors):
        latteriorL = window[:,avL].max(axis=1)  
        midL = window[:,2]
        diffl = np.mean(latteriorL)-np.mean(midL)
        correlationL = np.corrcoef(latteriorL,midL)[0][1]
    else:
        diffl = 0
        correlationL = 0
        
    if (avR.size >0 and 9 in availableSensors):
        anterR = window[:,avR].max(axis=1) 
        posterR = window[:,9]
        diffr = np.mean(anterR)-np.mean(posterR)
        correlationR = np.corrcoef(anterR,posterR)[0][1]
    else:
        diffr = diffl
        correlationR = 0
    
    avediff = (diffl+diffr)/2 if diffl > 0 else diffr
    return (avediff,correlationL,correlationR)
    
def overlappingrate (window, availableSensors, threshold = 5): 
    copyWin = window.copy()
    copyWin[copyWin<threshold] = 0 
    count = 0
    avL = availableSensors[[i in list(range(7)) for i in availableSensors]]
    avR = availableSensors[[i in list(range(7,14)) for i in availableSensors]]
    for each in copyWin:
        if any(each[avL]) != 0 and any(each[avR]) != 0:
            count += 1
    return count/len(window)

def fft(window):
    t = window.sum(axis = 1)
    fft_abs_amp = np.abs(np.fft.fft(t))*2/len(t)
    freq_spectrum = fft_abs_amp[1:int(np.floor(len(t) * 1.0 / 2)) + 1]
    skewness = scipy.stats.skew(freq_spectrum[0:150])
#    entropy = scipy.stats.entropy(freq_spectrum)
    s=0
    for i in range(25,int(np.floor(len(t) * 1.0 / 2))):
        s+=i*freq_spectrum[i]
    return (np.mean(freq_spectrum[30:150]),np.std(freq_spectrum[30:150]), (s/np.sum(freq_spectrum))/15, np.sum(freq_spectrum ** 2) / len(freq_spectrum), skewness) 

def fft_RE(window,availableSensors):
    t = window[:,availableSensors].sum(axis = 1)
    TENhz = int(10*len(t)/100)#Identify 10Hz index for all window size
    TWOhz = int(2*len(t)/100)    #Identify 2Hz index for all window size
    ONESIXSEVENhz = int(5/3*len(t)/100)    #Identify 1.67Hz index for all window size
    fft_abs_amp = np.abs(np.fft.fft(t))*2/len(t)    #Calculate the amptitude
    freq_spectrum = fft_abs_amp[1:int(np.floor(len(t) * 1.0 / 2)) + 1]    
    skewness = scipy.stats.skew(freq_spectrum[0:TENhz])    #skewness from 0-10Hz
    s=0
    for i in range(ONESIXSEVENhz,len(freq_spectrum)):    # This is the step to modify the range of frequency of MeanFrequency. int(5/3*len(t)/100) means 1.67Hz
        s+=i*freq_spectrum[i]
    MeanFreq = (s/np.sum(freq_spectrum[ONESIXSEVENhz:len(freq_spectrum)])/(len(t)/100))   
    power = np.sum(freq_spectrum ** 2)/len(freq_spectrum)
    return (np.mean(freq_spectrum[TWOhz:TENhz]),np.std(freq_spectrum[TWOhz:TENhz]), MeanFreq, power, skewness)

'''Functions to read files'''
def read_files(path, extention, labels = ['upstairs', 'downstairs', 'jog', 'nonlocal', 'run', 'sit', 'stand', 'walkF', 'walkN', 'upslop', 'cycling']):
    files = []
    for i in range(len(labels)):
        filename = path + labels[i] + extention
        with open(filename, 'rb') as f:
            files.append(pickle.load(f))
    return files

def read_csvs(path):
    fullData = []
    for fname in glob.glob(path+os.path.sep+'*.csv'):
        with open(fname, 'r') as f:
            fullData.append(pd.read_csv(f))
    return fullData

'''Filtration'''
def filtdata(calibdata_list, order_filter=2, critical_frequency=0.2):
    b, a = sig.butter(order_filter, critical_frequency)
    filtdata_list = []

    for i in range(len(calibdata_list)):
        filtdata_list.append(np.zeros((len(calibdata_list[i]), 14)))
        caliarray = np.array(calibdata_list[i][['L1','L2','L3','L4','L5','L6','L7','R1','R2','R3','R4','R5','R6','R7']])
        for j in range(0,14):
            filtdata_list[i][:,j]= sig.filtfilt(b,a,caliarray[:,j])

    for i in range(len(filtdata_list)):
        # remove the negative value
        filtdata_list[i][filtdata_list[i]<0] = 0
 
        
    return filtdata_list

'''Window cutting'''
def cutWindow(data, wl, step):

#    data_list_reshaped = []
    data_window_lists = []

    j = 0
    
    while j < len(data) - wl:
        window = data[j : j + wl]
        data_window_lists.append(window)
        j += step
    
    return data_window_lists

'''Hardcoded list of features; should be moved to a config file'''
def init_feature_names():
    columnNames = ['aveL7', 'aveL6','aveL5', 'aveL4','aveL3', 'aveL2','aveL1',
                                               'aveR7', 'aveR6','aveR5', 'aveR4','aveR3', 'aveR2','aveR1',
                                               'stdL7', 'stdL6','stdL5', 'stdL4','stdL3', 'stdL2','stdL1',
                                               'stdR7', 'stdR6','stdR5', 'stdR4','stdR3', 'stdR2','stdR1',
                                               'midL7', 'midL6','midL5', 'midL4','midL3', 'midL2','midL1',
                                               'midR7', 'midR6','midR5', 'midR4','midR3', 'midR2','midR1',
                                               'maxL7', 'maxL6','maxL5', 'maxL4','maxL3', 'maxL2','maxL1',
                                               'maxR7', 'maxR6','maxR5', 'maxR4','maxR3', 'maxR2','maxR1',
                                       'peaknumL7','peakdisaveL7','peakdisstdL7','peakmgaveL7','peakmgstdL7','peakwthaveL7','peakwthstdL7',
                                       'peaknumL6','peakdisaveL6','peakdisstdL6','peakmgaveL6','peakmgstdL6','peakwthaveL6','peakwthstdL6',
                                       'peaknumL5','peakdisaveL5','peakdisstdL5','peakmgaveL5','peakmgstdL5','peakwthaveL5','peakwthstdL5',
                                       'peaknumL4','peakdisaveL4','peakdisstdL4','peakmgaveL4','peakmgstdL4','peakwthaveL4','peakwthstdL4',
                                       'peaknumL3','peakdisaveL3','peakdisstdL3','peakmgaveL3','peakmgstdL3','peakwthaveL3','peakwthstdL3',
                                       'peaknumL2','peakdisaveL2','peakdisstdL2','peakmgaveL2','peakmgstdL2','peakwthaveL2','peakwthstdL2',
                                       'peaknumL1','peakdisaveL1','peakdisstdL1','peakmgaveL1','peakmgstdL1','peakwthaveL1','peakwthstdL1',
                                       'peaknumR7','peakdisaveR7','peakdisstdR7','peakmgaveR7','peakmgstdR7','peakwthaveR7','peakwthstdR7',
                                       'peaknumR6','peakdisaveR6','peakdisstdR6','peakmgaveR6','peakmgstdR6','peakwthaveR6','peakwthstdR6',
                                       'peaknumR5','peakdisaveR5','peakdisstdR5','peakmgaveR5','peakmgstdR5','peakwthaveR5','peakwthstdR5',
                                       'peaknumR4','peakdisaveR4','peakdisstdR4','peakmgaveR4','peakmgstdR4','peakwthaveR4','peakwthstdR4',
                                       'peaknumR3','peakdisaveR3','peakdisstdR3','peakmgaveR3','peakmgstdR3','peakwthaveR3','peakwthstdR3',
                                       'peaknumR2','peakdisaveR2','peakdisstdR2','peakmgaveR2','peakmgstdR2','peakwthaveR2','peakwthstdR2',
                                       'peaknumR1','peakdisaveR1','peakdisstdR1','peakmgaveR1','peakmgstdR1','peakwthaveR1','peakwthstdR1']
    to_recomp = ['APdiff','APcoraL','APcoraR','LMdiff','LMcoraL','LMcoraR','fftmean','fftstd','fftweight','fftennergy','fftskewness','overlap','inerstepinterval']
    return columnNames,to_recomp

'''Make a feature list'''
def comp_base_features(windowtotal):
   
    feature_all = list()
    for each in windowtotal:
        featureforone = []
        window = each
        temp = sensorgeneral(window)
        for i in temp:
            for j in i:
                featureforone.append(j)
        temp = inerpeakinfo(window)
        for i in temp:
            for j in i:
                featureforone.append(j)
    
    
        feature_all.append(featureforone)
   
    return feature_all

def recomp(windowtotal,availableSensors,availableFeatureNames):
    df_featureplus = pd.DataFrame()
    if 'APdiff' not in availableFeatureNames:
        featuretotAP = []
        for window in windowtotal:
            featureforone = []
            temp = anterposterNAKDR(window,availableSensors)
            for i in temp:
                featureforone.append(i)
            featuretotAP.append(featureforone)
        if len(featureforone)!= 0:
            df_AP = pd.DataFrame(featuretotAP,columns = ['APdiff','APcoraL','APcoraR'])
            df_featureplus = pd.concat([df_featureplus, df_AP], axis=1, sort=False)
        
    if 'LMdiff' not in availableFeatureNames:    
        featuretotLM = []
        for window in windowtotal:
            featureforone = []
            temp = lattomidNAKDR(window,availableSensors)
            for i in temp:
                featureforone.append(i)
            featuretotLM.append(featureforone)
        if len(featureforone)!= 0:
            df_LM = pd.DataFrame(featuretotLM,columns = ['LMdiff','LMcoraL','LMcoraR'])
            df_featureplus = pd.concat([df_featureplus,df_LM], axis=1, sort=False)        
        
    featuretotFFT = []    
    for window in windowtotal:
        featureforone = []
        temp = fft_RE(window,availableSensors)
        for i in temp:
            featureforone.append(i)
        featuretotFFT.append(featureforone)
    df_FFT = pd.DataFrame(featuretotFFT,columns = ['fftmean','fftstd','fftweight','fftennergy','fftskewness'])
    df_featureplus = pd.concat([df_featureplus,df_FFT], axis=1, sort=False)

    featuregatephase = []
    for window in windowtotal:
        featuregatephase.append([overlappingrate(window,availableSensors),instepfeature(window,availableSensors)])
    df_gatephase = pd.DataFrame(featuregatephase,columns=['overlap','inerstepinterval'])
    df_featureplus = pd.concat([df_featureplus,df_gatephase], axis = 1, sort=False)
        
    return df_featureplus

'''Get list of available features'''
def get_available_features(availableSensors, columnNames, nSensors = 14, totsinglesensordep = 4, totsinglesensorcol = 7):
    '''
    !!NOTICE!!
    when input the intrest configurations follow this:
    L1 -- 6       R1 -- 13
    L2 -- 5       R2 -- 12
    L3 -- 4       R3 -- 11
    L4 -- 3       R4 -- 10
    L5 -- 2       R5 -- 9
    L6 -- 1       R6 -- 8
    L7 -- 0       R7 -- 7
    
    '''
    requirements = []
    for _ in range(totsinglesensordep):
        requirements += [[i] for i in range(nSensors)]
    for i in range(nSensors):
        requirements += [[i] for _ in range(totsinglesensorcol)]
    
    availableFeatureNames =[]
    for i in range(len(requirements)):
        if all(sensor in availableSensors for sensor in requirements[i]):
            availableFeatureNames.append(columnNames[i])
    return availableFeatureNames

#================================#
#===== High level functions =====#
#================================#

def base_train_evaluate(dfsample, training_subjects_assignments,nRepeats = 100, interval = 5, nTrees = 100):
    n_ft = dfsample.shape[1] #Get the size of sample list: The result here is feature number + label number
    score_rf = []

    score_all = []
    pred_all = []
    test_all = []
    
    for training_subject in training_subjects_assigments:
        
        X_train = np.array([], dtype=np.int64).reshape(0,n_ft-2) 
        X_test = X_train.copy() #Empty, same size, for test sample feature vector
        y_train = [] #To fill with training labels
        y_test = [] #To fill with test labels
        
        
        trn = dfsample.loc[dfsample['Subject'].isin (training_subject)] #Select training and testing samples according to subjects
        other_indexes = [i for i in unique_labels(dfsample['Subject']) if i not in training_subject]
        tst = dfsample.loc[dfsample['Subject'].isin (other_indexes)]
            
        X_train = np.concatenate((X_train, trn.iloc[:,0:-2].values), axis=0)
        X_test = np.concatenate((X_test, tst.iloc[:,0:-2].values), axis=0)
        y_train = np.concatenate((y_train, trn.iloc[:,-1].values), axis=0)
        y_test = np.concatenate((y_test, tst.iloc[:,-1].values), axis=0)
      
        repeats = nRepeats 
        seed = 0
        
        featureimportance = np.zeros(len(dfsample.keys())-2)
        
        for index in range(repeats):
            seed += interval
            rf = RandomForestClassifier(n_estimators=nTrees, random_state=seed, verbose=0,
                                        min_samples_split=2, class_weight="balanced", n_jobs = 1)    
            rf.fit(X_train, y_train)
        
            featureimportance += rf.feature_importances_
        #        print(rf.feature_importances_)
            y_pred = rf.predict(X_test)
            score_rf.append(accuracy_score(y_test, y_pred))

            pred_all.append(y_pred)
            test_all.append(y_test)
        featureimportance /= repeats
            
    
    return score_rf,pred_all,test_all,featureimportance

def setup_evaluation(path,windowLength = 1500):
    #Read files
    files = read_csvs(path)

    #Filter data
    filt_list = filtdata(files)

    #Cut windows
    windowtotal = []    
    df_label = pd.DataFrame()
    
    for i in range(len(filt_list)): 
        temp = cutWindow(filt_list[i],windowLength,int(windowLength/2))
        for t in temp:
            windowtotal.append(t)
            df_label = pd.concat([df_label,files[i][:1][['Subject','Activity']]],ignore_index = True) 


    print(df_label)
    featureNames, to_recomp = init_feature_names()

    feature_all = comp_base_features(windowtotal)
    
    df_feature = pd.DataFrame(feature_all,columns=featureNames)

    #df_label =  pd.DataFrame(label_all,columns = ('Subject','Activity')) #Label columns
    return windowtotal, df_feature, df_label

def remove_features(df_sample, featureimportance, differencePerRun = 20):
    
    #differencePerRun: number of features removed per run
    lfeatureimportance = featureimportance.tolist()
        
    max_num_index = map(lfeatureimportance.index, 
                        heapq.nlargest(len(lfeatureimportance)-differencePerRun, lfeatureimportance))
        
    ft_selected = df_sample.iloc[:,list(max_num_index)]
    print(ft_selected.columns) #Get the names of 40 most important features
    sample_ft_selected = pd.concat([ft_selected, df_sample[['Subject','Activity']]], axis=1, sort=False)
    return sample_ft_selected

def plot_confusion_matrix(y_true, y_pred, classes,
                      normalize=False,
                      title=None,
                      cmap=plt.cm.Blues):

##    This function prints and plots the confusion matrix.
##    Normalization can be applied by setting `normalize=True`.
    #sns.set(rc={'figure.figsize':(11.7,8.27)})
    font = {'weight':'normal', 'size': 18,}
    
  
    ## Adapted from the original code of sklearn and Dian
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes)
           #ylabel='True label',
           #xlabel='Predicted label')
    plt.xlabel('Predicted label',fontsize = 18)
    plt.ylabel('True label',fontsize = 18)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im,ax=ax)
   
   

    ax.set_aspect ('equal')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor", fontsize = 18)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor", fontsize = 18)

    
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",fontdict = font)
            
    fig.tight_layout()
    return ax

def maxfeature_run(availableSensors,training_subjects_assigments, windowtotal,df_feature,df_label, output_folder = '', nTrees = 100):
    featureNames = list(df_feature.columns)
    configName = " ".join(str(x) for x in availableSensors)
    available_features =  get_available_features(availableSensors,featureNames)
    #print(available_features)                                                                                                                                                                              
    df_temp = recomp(windowtotal,availableSensors,available_features)
    #print(df_temp)                                                                                                                                                                                         
    sample = pd.concat([df_feature[available_features], df_temp, df_label], axis=1, sort=False)                                           
    sample = sample.fillna(0)                                 
    totalfeatures = len(sample.columns) - len(df_label.columns)
    scores,preds,reals,featureimportance = base_train_evaluate(sample,training_subjects_assigments, nRepeats = 20, nTrees = nTrees)
    df_score = pd.DataFrame(scores,columns = [totalfeatures])
    df_score.to_csv(output_folder+configName+'_'+str(totalfeatures)+'.csv')
        #print(featureimportance)                                                                                                                                                                           
        #print(sample.columns.to_numpy()[:-len(df_label.columns)])                                                                                                                                          
    df_featureimportance = pd.DataFrame([featureimportance],columns = sample.columns.to_numpy()[:-len(df_label.columns)])
    df_featureimportance.to_csv(output_folder+configName+'_'+str(totalfeatures)+'features.csv')
    overall_true_categories = []
    overall_predictions = []

    cmnmlzd = []
    for i in range(len(reals)):
        overall_true_categories.extend(list(reals[i]))
        overall_predictions.extend(list(preds[i]))
        cm = confusion_matrix(reals[i], preds[i])
        cmnmlzd.append(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    #print(availableSensors,(totalfeatures-nFeatures),"acc:",accuracy_score(overall_true_categories, overall_predictions)) #average accurate                             
    meancm = sum(cmnmlzd)/len(cmnmlzd)#average accurate
    with sns.axes_style("whitegrid",{'axes.grid': False}):
        fig = plot_confusion_matrix(overall_true_categories,overall_predictions,classes=['upstairs', 'downstairs', 'housework', 'run', 'sit', 'stand', 'walk', 'upslope', 'cycling','walk'], normalize=True,title='Normalized confusion matrix')
        plt.savefig(fname = output_folder+'heatmap_'+configName+'_'+str(totalfeatures)+'.pdf',format="pdf")
        plt.close()
    np.savetxt(output_folder+'overall_true_pred_'+configName+'_'+str(totalfeatures)+'.dat',meancm,delimiter=',')
        #with open('overall_true_pred_'+configName+'_'+str(totalfeatures-nFeatures)+'.dat','w') as savefile:                                                                                                
        #    savefile.write(str(overall_true_categories))                                                                                                                                                   
        #    savefile.write("\n")                                                                                                                                                                           
        #    savefile.write(str(overall_predictions))                                                                                                                                                                      

def full_run(availableSensors,training_subjects_assigments, windowtotal,df_feature,df_label, output_folder = '', nTrees = 100):
    featureNames = list(df_feature.columns)
    configName = " ".join(str(x) for x in availableSensors)
    available_features =  get_available_features(availableSensors,featureNames)
    #print(available_features)
    df_temp = recomp(windowtotal,availableSensors,available_features)
    #print(df_temp)
    sample = pd.concat([df_feature[available_features], df_temp, df_label], axis=1, sort=False)
    sample = sample.fillna(0) 
    totalfeatures = len(sample.columns) - len(df_label.columns)
    for nFeatures in range(totalfeatures):
        scores,preds,reals,featureimportance = base_train_evaluate(sample,training_subjects_assigments, nRepeats = 20, nTrees = nTrees)
        df_score = pd.DataFrame(scores,columns = [totalfeatures-nFeatures])
        df_score.to_csv(output_folder+configName+'_'+str(totalfeatures-nFeatures)+'.csv')
        #print(featureimportance)
        #print(sample.columns.to_numpy()[:-len(df_label.columns)])
        df_featureimportance = pd.DataFrame([featureimportance],columns = sample.columns.to_numpy()[:-len(df_label.columns)])
        df_featureimportance.to_csv(output_folder+configName+'_'+str(totalfeatures-nFeatures)+'features.csv')
        overall_true_categories = []
        overall_predictions = []
        
        cmnmlzd = []
        for i in range(len(reals)):
            overall_true_categories.extend(list(reals[i]))
            overall_predictions.extend(list(preds[i]))
            cm = confusion_matrix(reals[i], preds[i])
            cmnmlzd.append(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
            
        print(availableSensors,(totalfeatures-nFeatures),"acc:",accuracy_score(overall_true_categories, overall_predictions)) 
        meancm = sum(cmnmlzd)/len(cmnmlzd)
        with sns.axes_style("whitegrid",{'axes.grid': False}):
            fig = plot_confusion_matrix(overall_true_categories,overall_predictions,classes=['upstairs', 'downstairs', 'housework', 'run', 'sit', 'stand', 'walk', 'upslope', 'cycling','walk'], normalize=True,title='Normalized confusion matrix')
            plt.savefig(fname = output_folder+'heatmap_'+configName+'_'+str(totalfeatures-nFeatures)+'.pdf',format="pdf")
            plt.close()
        np.savetxt(output_folder+'overall_true_pred_'+configName+'_'+str(totalfeatures-nFeatures)+'.dat',meancm,delimiter=',')
        #with open('overall_true_pred_'+configName+'_'+str(totalfeatures-nFeatures)+'.dat','w') as savefile:
        #    savefile.write(str(overall_true_categories))
        #    savefile.write("\n")
        #    savefile.write(str(overall_predictions))
        if nFeatures < totalfeatures - 1:
            sample = remove_features(sample,featureimportance,differencePerRun = 1)

if __name__ == '__main__':
    start_time = time.time()

    output_folder = 'output/'
    nTrees = 100
    windowLength = [100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
    
    interestConfig67 = [np.array([i,i+7]) for i in range(7)] #all single sensors (pairs)
    interestConfig67 += [np.array([6,6-i,13,13-i]) for i in range(1,6)] #all pair of sensors including the heel
    interestConfig67 += [np.array([2,4,9,11]),np.array([0,2,7,9]),np.array([0,4,7,11]),np.array([0,3,7,10])] #horizontal pairs
    interestConfig67 += [np.array([0,1,7,8]),np.array([0,5,7,12]),np.array([0,6,7,13])]
    interestConfig67 += [np.array([0,2,6,7,9,13]),np.array([0,4,6,7,11,13]),np.array([2,4,6,9,11,13]),np.array([2,3,4,9,10,11])] # 3 sensors
    interestConfig67 += [np.array([0,1,6,7,8,13]),np.array([0,1,3,7,8,10]),np.array([0,5,6,7,12,13]),np.array([0,1,2,7,8,9])] # 3 sensors
    interestConfig67 += [np.array([0,4,5,7,11,12]),np.array([2,4,5,9,11,12]),np.array([0,2,5,7,9,12]),np.array([1,5,6,8,12,13]), np.array([0,1,5,7,8,12])] # 3 sensors
    interestConfig67 += [np.array([2,3,4,6,9,10,11,13]),np.array([0,2,3,6,7,9,10,13]),np.array([0,3,4,6,7,10,11,13]),np.array([0,2,4,6,7,9,11,13]),np.array([0,1,3,6,7,8,10,13])] # 4 sensors
    interestConfig67 += [np.array([0,1,2,3,7,8,9,10]),np.array([0,1,3,4,7,8,10,11]),np.array([1,2,3,4,8,9,10,11]),np.array([2,3,4,5,9,10,11,12])] # 4 sensors
    interestConfig67 += [np.array([0,2,3,5,7,9,10,12]),np.array([0,3,4,5,7,10,11,12]),np.array([0,2,4,5,7,9,11,12]),np.array([0,1,5,6,7,8,12,13])] # 4 sensors
    interestConfig67 += [np.array([1,2,5,6,8,9,12,13]),np.array([1,4,5,6,8,11,12,13]),np.array([3,4,5,6,10,11,12,13]),np.array([0,4,5,6,7,11,12,13]), np.array([1,2,3,6,8,9,10,13])] # 4 sensors
    interestConfig67 += [np.array([0,2,3,4,6,7,9,10,11,13]),np.array([0,2,4,5,6,7,9,11,12,13]),np.array([0,1,4,5,6,7,8,11,12,13]),np.array([0,1,2,5,6,7,8,9,12,13])] # 5 sensors
    interestConfig67 += [np.array([1,3,4,5,6,8,10,11,12,13]),np.array([2,3,4,5,6,9,10,11,12,13]),np.array([0,2,3,4,5,7,9,10,11,12]),np.array([1,2,3,4,5,8,9,10,11,12]),np.array([0,1,2,3,4,7,8,9,10,11])] # 5 sensors
    interestConfig67 += [np.delete(np.array([i for i in range(14)]),[j,j+7]) for j in range(7)]
    interestConfig67 += [np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])] # sensors    
    
    interestConfig = interestConfig67
    
    def changeConfigsto127():
        from itertools import combinations 
        interestConfig127 = []
        for combi in [combinations([0, 1, 2, 3, 4, 5, 6], i) for i in range(1,8)]:
            for Config in combi:
                interestConfig127 += [np.concatenate((np.array(Config),np.array(Config)+7),axis=None)]   
        return interestConfig127
    
    def changeConfigsto25():
        interestConfig25 = [np.array([6,13]), np.array([0,7]), np.array([4,6,11,13]), np.array([0,6,7,13]), np.array([2,4,9,11]), np.array([0,3,7,10]),
                            np.array([0,4,6,7,11,13]), np.array([0,5,6,7,12,13]), np.array([2,3,4,9,10,11]), np.array([0,1,3,7,8,10]),
                            np.array([3,4,5,6,10,11,12,13]), np.array([0,4,5,6,7,11,12,13]), np.array([2,3,4,6,9,10,11,13]), np.array([1,2,3,4,8,9,10,11]),
                            np.array([0,1,4,5,6,7,8,11,12,13]), np.array([0,2,3,4,6,7,9,10,11,13]), np.array([0,2,4,5,6,7,9,11,12,13]), np.array([2,3,4,5,6,9,10,11,12,13]),
                            np.array([0,1,2,3,4,7,8,9,10,11]), np.array([0,2,3,4,5,7,9,10,11,12]),
                            np.array([0,1,3,4,5,6,7,8,10,11,12,13]), np.array([0,2,3,4,5,6,7,9,10,11,12,13]), np.array([0,1,2,4,5,6,7,8,9,11,12,13]), np.array([0,1,2,3,4,5,7,8,9,10,11,12]),
                            np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])]
        return interestConfig25
        
    
    availableThreads = psutil.cpu_count()


    basePath = sys.argv[1] if len(sys.argv) > 1 else None
    if basePath == None:
        print("Error: basePath not set. Please set path.")
        exit()
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    if len(sys.argv) > 3:
        windowLength = [int(sys.argv[3])]
    if len(sys.argv) > 4:
        if int(sys.argv[4]) == 127:
            interestConfig = changeConfigsto127()
        elif int(sys.argv[4]) == 67:
            pass
        elif int(sys.argv[4]) == 25:
            interestConfig = changeConfigsto25()
        else:
            print("Error: Configuration number is not available, please replace Configuration_number with 127 for result of all sensor configurations or 67 for original analysing results or 25 for configurations in the paper")
            exit()
    
    totalFeatures = [setup_evaluation(basePath,windowLength=winLength) for winLength in windowLength]
    #windowtotal, df_feature, df_label = setup_evaluation(basePath,windowLength=windowLength)
    featureNames = list(totalFeatures[0][2].columns)
    
    
    print("Tried windowsize:", windowLength)
    
    print("Tried configurations:", interestConfig)
    print("Configuration number:", len(interestConfig))
    
    #training_subjects_assigments = [np.random.choice([4,5,7,8,9,10,11,13,15,18,30],6,replace=False) for _ in range(5)]
    
    training_subjects_assigments = [[4,5,7,8,9,10],[7,8,11,13,15,30],[4,7,9,10,15,18],[4,5,11,13,18,30],[7,8,10,15,18,30]]
    print("Training assignments:",training_subjects_assigments)
    with open(output_folder+'assignments.txt','w') as f:
        f.write(str(training_subjects_assigments))
        f.flush()

    

    for elem in totalFeatures:
    
        elem[2].loc[elem[2].Activity == 'walkF','Activity'] = 'walk'
        elem[2].loc[elem[2].Activity == 'walkN','Activity'] = 'walk'
        elem[2].loc[elem[2].Activity == 'jog','Activity'] = 'run'
        elem[2].loc[elem[2].Activity == 'nonlocal','Activity'] = 'housework'
        elem[2].loc[elem[2].Activity == 'sit','Activity'] = 'sitting'
        
        

    labels = ['upstairs', 'downstairs', 'housework', 'run', 'sitting', 'standing', 'upslope', 'cycling','walk'] 
    def temp_run(availableSensors):
        full_run(availableSensors,training_subjects_assigments, totalFeatures[0][0],totalFeatures[0][1],totalFeatures[0][2],output_folder = output_folder,nTrees = nTrees)
    def temp_run_fullfeatures(elem):
        maxfeature_run(interestConfig[-1],training_subjects_assigments, elem[0],elem[1],elem[2],output_folder = output_folder+str(len(elem[0][0]))+"_",nTrees = nTrees)
    with Pool(availableThreads) as pool:
        if len(windowLength) > 1:
            pool.map(temp_run_fullfeatures,totalFeatures)
        else:
            pool.map(temp_run,interestConfig)
        #

    print('Strart time',start_time,'end time',time.time())
