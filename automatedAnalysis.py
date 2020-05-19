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
    "First count step then calculate each step"
    def stepcounting(x):
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
    "If in one step there are more than two peaks, the difference between the last one and the first one"
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

        "Total number of peaks"
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
"General features"
def sensorgeneral (window):
    std = []
    ave = []
    mid = []
    maximum = []
    for columns in window.T:
        y = list(filter(lambda a: a != 0.0, columns)) #Ramove zeros while calculating features
        if len(y) == 0:
            y = [0]
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
        #Situation that featrues cannot be computed because available sensors are not enough to find anterior-posterior relation
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
        #Situation that featrues cannot be computed because available sensors are not enough to find lateral-medial foot relation
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

"Double floating rate" 
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

"FFT features"
def fft_RE(window,availableSensors):
    t = window[:,availableSensors].sum(axis = 1)
    TENhz = int(10*len(t)/100)#Identify 10Hz index for all window size
    TWOhz = int(2*len(t)/100)    #Identify 2Hz index for all window size
    ONESIXSEVENhz = int(5/3*len(t)/100)    #Identify 1.67Hz index for all window size
    fft_abs_amp = np.abs(np.fft.fft(t))*2/len(t)    #Calculate the amptitude
    freq_spectrum = fft_abs_amp[1:int(np.floor(len(t) * 1.0 / 2)) + 1]    #Only half of the data is meaningful for our case we can only get the spectum up to 50Hz
    skewness = scipy.stats.skew(freq_spectrum[0:TENhz])    #skewness from 0-10Hz
    s=0
    for i in range(ONESIXSEVENhz,len(freq_spectrum)):    # This is the step to modify the range of frequency of MeanFrequency. int(5/3*len(t)/100) means 1.67Hz
        s+=i*freq_spectrum[i]

    MeanFreq = (s/np.sum(freq_spectrum[ONESIXSEVENhz:len(freq_spectrum)])/(len(t)/100))  #weighted mean frequency  
    power = np.sum(freq_spectrum ** 2)/len(freq_spectrum)#Power density.
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
        filtdata_list[i][filtdata_list[i]<0] = 0
 
        
    return filtdata_list

'''Window cutting'''
def cutWindow(data, wl, step):

    data_window_lists = []

    j = 0
    
    while j < len(data) - wl:
        window = data[j : j + wl]
        data_window_lists.append(window)
        j += step
    
    return data_window_lists

'''Hardcoded list of features'''
def init_feature_names():
    columnNames = ['mean of pressures left central forefoot', 'mean of pressures left central midfoot','mean of pressures left medial forefoot', 'mean of pressures left big toe','mean of pressures left lateral forefoot', 'mean of pressures left lateral midfoot','mean of pressures left heel',
                                               'mean of pressures right central forefoot', 'mean of pressures right central midfoot','mean of pressures right medial forefoot', 'mean of pressures right big toe','mean of pressures right lateral forefoot', 'mean of pressures right lateral midfoot','mean of pressures right heel',
                                               'SD of pressures left central forefoot', 'SD of pressures left central midfoot','SD of pressures left medial forefoot', 'SD of pressures left big toe','SD of pressures left lateral forefoot', 'SD of pressures left lateral midfoot','SD of pressures left heel',
                                               'SD of pressures right central forefoot', 'SD of pressures right central midfoot','SD of pressures right medial forefoot', 'SD of pressures right big toe','SD of pressures right lateral forefoot', 'SD of pressures right lateral midfoot','SD of pressures right heel',
                                               'median of pressures left central forefoot', 'median of pressures left central midfoot','median of pressures left medial forefoot', 'median of pressures left big toe','median of pressures left lateral forefoot', 'median of pressures left lateral midfoot','median of pressures left heel',
                                               'median of pressures right central forefoot', 'median of pressures right central midfoot','median of pressures right medial forefoot', 'median of pressures right big toe','median of pressures right lateral forefoot', 'median of pressures right lateral midfoot','median of pressures right heel',
                                               'maximal pressures left central forefoot', 'maximal pressures left central midfoot','maximal pressures left medial forefoot', 'maximal pressures left big toe','maximal pressures left lateral forefoot', 'maximal pressures left lateral midfoot','maximal pressures left heel',
                                               'maximal pressures right central forefoot', 'maximal pressures right central midfoot','maximal pressures right medial forefoot', 'maximal pressures right big toe','maximal pressures right lateral forefoot', 'maximal pressures right lateral midfoot','maximal pressures right heel',
                                       'number of peaks left central forefoot','average peak interval left central forefoot','SD of peak interval left central forefoot','average peak magnitude left central forefoot','SD of peak magnitude left central forefoot','average peak width left central forefoot','SD of peak width left central forefoot',
                                       'number of peaks left central midfoot','average peak interval left central midfoot','SD of peak interval left central midfoot','average peak magnitude left central midfoot','SD of peak magnitude left central midfoot','average peak width left central midfoot','SD of peak width left central midfoot',
                                       'number of peaks left medial forefoot','average peak interval left medial forefoot','SD of peak interval left medial forefoot','average peak magnitude left medial forefoot','SD of peak magnitude left medial forefoot','average peak width left medial forefoot','SD of peak width left medial forefoot',
                                       'number of peaks left big toe','average peak interval left big toe','SD of peak interval left big toe','average peak magnitude left big toe','SD of peak magnitude left big toe','average peak width left big toe','SD of peak width left big toe',
                                       'number of peaks left lateral forefoot','average peak interval left lateral forefoot','SD of peak interval left lateral forefoot','average peak magnitude left lateral forefoot','SD of peak magnitude left lateral forefoot','average peak width left lateral forefoot','SD of peak width left lateral forefoot',
                                       'number of peaks left lateral midfoot','average peak interval left lateral midfoot','SD of peak interval left lateral midfoot','average peak magnitude left lateral midfoot','SD of peak magnitude left lateral midfoot','average peak width left lateral midfoot','SD of peak width left lateral midfoot',
                                       'number of peaks left heel','average peak interval left heel','SD of peak interval left heel','average peak magnitude left heel','SD of peak magnitude left heel','average peak width left heel','SD of peak width left heel',
                                       'number of peaks right central forefoot','average peak interval right central forefoot','SD of peak interval right central forefoot','average peak magnitude right central forefoot','SD of peak magnitude right central forefoot','average peak width right central forefoot','SD of peak width right central forefoot',
                                       'number of peaks right central midfoot','average peak interval right central midfoot','SD of peak interval right central midfoot','average peak magnitude right central midfoot','SD of peak magnitude right central midfoot','average peak width right central midfoot','SD of peak width right central midfoot',
                                       'number of peaks right medial forefoot','average peak interval right medial forefoot','SD of peak interval right medial forefoot','average peak magnitude right medial forefoot','SD of peak magnitude right medial forefoot','average peak width right medial forefoot','SD of peak width right medial forefoot',
                                       'number of peaks right big toe','average peak interval right big toe','SD of peak interval right big toe','average peak magnitude right big toe','SD of peak magnitude right big toe','average peak width right big toe','SD of peak width right big toe',
                                       'number of peaks right lateral forefoot','average peak interval right lateral forefoot','SD of peak interval right lateral forefoot','average peak magnitude right lateral forefoot','SD of peak magnitude right lateral forefoot','average peak width right lateral forefoot','SD of peak width right lateral forefoot',
                                       'number of peaks right lateral midfoot','average peak interval right lateral midfoot','SD of peak interval right lateral midfoot','average peak magnitude right lateral midfoot','SD of peak magnitude right lateral midfoot','average peak width right lateral midfoot','SD of peak width right lateral midfoot',
                                       'number of peaks right heel','average peak interval right heel','SD of peak interval right heel','average peak magnitude right heel','SD of peak magnitude right heel','average peak width right heel','SD of peak width right heel']
    to_recomp = ['anterior-posterior mean difference','anterior-posterior correlation coefficient left foot','anterior-posterior correlation coefficient right foot','median-lateral mean difference','median-lateral correlation coefficient left foot','median-lateral correlation coefficient right foot','mean of AC component FFT','SD of AC component FFT','weighted frequency average','FFT energy','FFT skewness','double float phase duration','pressure difference between foot landing and lifting']
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

"Calculate Anterior-posterior, lateral-medial foot relation features and fft features while using different sensor configurations"
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
            df_AP = pd.DataFrame(featuretotAP,columns = ['anterior-posterior mean difference','anterior-posterior correlation coefficient left foot','anterior-posterior correlation coefficient right foot'])
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
            df_LM = pd.DataFrame(featuretotLM,columns = ['median-lateral mean difference','median-lateral correlation coefficient left foot','median-lateral correlation coefficient right foot'])
            df_featureplus = pd.concat([df_featureplus,df_LM], axis=1, sort=False)        
        
    featuretotFFT = []    
    for window in windowtotal:
        featureforone = []
        temp = fft_RE(window,availableSensors)
        for i in temp:
            featureforone.append(i)
        featuretotFFT.append(featureforone)
    df_FFT = pd.DataFrame(featuretotFFT,columns = ['mean of AC component FFT','SD of AC component FFT','weighted frequency average','FFT energy','FFT skewness'])
    df_featureplus = pd.concat([df_featureplus,df_FFT], axis=1, sort=False)

    featuregatephase = []
    for window in windowtotal:
        featuregatephase.append([overlappingrate(window,availableSensors),instepfeature(window,availableSensors)])
    df_gatephase = pd.DataFrame(featuregatephase,columns=['double float phase duration','pressure difference between foot landing and lifting'])
    df_featureplus = pd.concat([df_featureplus,df_gatephase], axis = 1, sort=False)
        
    return df_featureplus

'''Get list of available features'''
def get_available_features(availableSensors, columnNames, nSensors = 14, totsinglesensordep = 4, totsinglesensorcol = 7):
    '''
    colomns in raw data is reversed
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
    #all other features must be recalculated
    
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
        
        X_train = np.array([], dtype=np.int64).reshape(0,n_ft-2) #An empty array shaped according to feature numbers, will be filled in with training sample feature vector
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
      
        repeats = nRepeats #It is how many time fitting repeats
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
            df_label = pd.concat([df_label,files[i][:1][['Subject','Activity']]],ignore_index = True) #all the content of the file should be the same subject/activity combo


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
    font = {'weight':'normal', 'size': 18,}
    

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
    
     #show all ticks.
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

#    print(availableSensors,(totalfeatures-nFeatures),"acc:",accuracy_score(overall_true_categories, overall_predictions)) #Add up predicted activities and true activities in 100 run.
    meancm = sum(cmnmlzd)/len(cmnmlzd)#Calculate confusion matrix after 100 runs
#    with sns.axes_style("whitegrid",{'axes.grid': False}):
#        fig = plot_confusion_matrix(overall_true_categories,overall_predictions,classes=['upstairs', 'downstairs', 'housework', 'run', 'sit', 'stand', 'walk', 'upslope', 'cycling','walk'], normalize=True,title='Normalized confusion matrix')
#        plt.savefig(fname = output_folder+'heatmap_'+configName+'_'+str(totalfeatures)+'.pdf',format="pdf")
#        plt.close()
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
#        with sns.axes_style("whitegrid",{'axes.grid': False}):
#            fig = plot_confusion_matrix(overall_true_categories,overall_predictions,classes=['upstairs', 'downstairs', 'housework', 'run', 'sit', 'stand', 'walk', 'upslope', 'cycling','walk'], normalize=True,title='Normalized confusion matrix')
#            plt.savefig(fname = output_folder+'heatmap_'+configName+'_'+str(totalfeatures-nFeatures)+'.pdf',format="pdf")
#            plt.close()
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
    windowLength = [2000] 
    windowLength_multi = [100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]
    


    availableThreads = psutil.cpu_count()


    basePath = sys.argv[1] if len(sys.argv) > 1 else None
    if basePath == None:
        print("Error: basePath not set. Please set path.")
        exit()
    output_folder = sys.argv[2]
    output_folder_window = sys.argv[3]

    totalFeatures = [setup_evaluation(basePath,windowLength=winLength) for winLength in windowLength]
    #windowtotal, df_feature, df_label = setup_evaluation(basePath,windowLength=windowLength)
    featureNames = list(totalFeatures[0][2].columns)

    interestConfig =  [np.array([i]) for i in range(14)] #all single sensors
    interestConfig += [np.array([i,i+7]) for i in range(7)] #all single sensors (pairs)
    interestConfig += [np.array([6,6-i,13,13-i]) for i in range(1,6)] #all pair of sensors including the heel
    interestConfig += [np.array([2,4,9,11]),np.array([0,2,7,9]),np.array([0,4,7,11]),np.array([0,3,7,10])] #horizontal pairs
    interestConfig += [np.array([0,1,7,8]),np.array([0,5,7,12]),np.array([0,6,7,13])]
    interestConfig += [np.array([0,2,6,7,9,13]),np.array([0,4,6,7,11,13]),np.array([2,4,6,9,11,13]),np.array([2,3,4,9,10,11])] # 3 sensors
    interestConfig += [np.array([0,1,6,7,8,13]),np.array([0,1,3,7,8,10]),np.array([0,5,6,7,12,13]),np.array([0,1,2,7,8,9])] # 3 sensors
    interestConfig += [np.array([0,4,5,7,11,12]),np.array([2,4,5,9,11,12]),np.array([0,2,5,7,9,12]),np.array([1,5,6,8,12,13]), np.array([0,1,5,7,8,12])] # 3 sensors
    interestConfig += [np.array([2,3,4,6,9,10,11,13]),np.array([0,2,3,6,7,9,10,13]),np.array([0,3,4,6,7,10,11,13]),np.array([0,2,4,6,7,9,11,13]),np.array([0,1,3,6,7,8,10,13])] # 4 sensors
    interestConfig += [np.array([0,1,2,3,7,8,9,10]),np.array([0,1,3,4,7,8,10,11]),np.array([1,2,3,4,8,9,10,11]),np.array([2,3,4,5,9,10,11,12])] # 4 sensors
    interestConfig += [np.array([0,2,3,5,7,9,10,12]),np.array([0,3,4,5,7,10,11,12]),np.array([0,2,4,5,7,9,11,12]),np.array([0,1,5,6,7,8,12,13])] # 4 sensors
    interestConfig += [np.array([1,2,5,6,8,9,12,13]),np.array([1,4,5,6,8,11,12,13]),np.array([3,4,5,6,10,11,12,13]),np.array([0,4,5,6,7,11,12,13]), np.array([1,2,3,6,8,9,10,13])] # 4 sensors
    interestConfig += [np.array([0,2,3,4,6,7,9,10,11,13]),np.array([0,2,4,5,6,7,9,11,12,13]),np.array([0,1,4,5,6,7,8,11,12,13]),np.array([0,1,2,5,6,7,8,9,12,13])] # 5 sensors
    interestConfig += [np.array([1,3,4,5,6,8,10,11,12,13]),np.array([2,3,4,5,6,9,10,11,12,13]),np.array([0,2,3,4,5,7,9,10,11,12]),np.array([1,2,3,4,5,8,9,10,11,12]),np.array([0,1,2,3,4,7,8,9,10,11])] # 5 sensors
    interestConfig += [np.delete(np.array([i for i in range(14)]),[j,j+7]) for j in range(7)]
    interestConfig += [np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])] # sensors
    #interestConfig = [np.array([0,6,7,13]),np.array([0,1,3,6,7,8,10,13])]
    print("Tried configurations:", interestConfig)
    
    
    #training_subjects_assigments = [np.random.choice([4,5,7,8,9,10,11,13,15,18,30],6,replace=False) for _ in range(5)]
    #training_subjects_assigments = [[4,5,7,8,9,10]]
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
    
    labels = ['upstairs', 'downstairs', 'housework', 'run', 'sitting', 'standing', 'upslope', 'cycling','walk'] #Renewed Label list
    def temp_run(availableSensors):
        full_run(availableSensors,training_subjects_assigments, totalFeatures[0][0],totalFeatures[0][1],totalFeatures[0][2],output_folder = output_folder,nTrees = nTrees)
    def temp_run_fullfeatures(elem):
        maxfeature_run(interestConfig[-1],training_subjects_assigments, elem[0],elem[1],elem[2],output_folder = output_folder_window + str(len(elem[0][0]))+"_",nTrees = nTrees)
    
    with Pool(availableThreads) as pool:
        pool.map(temp_run,interestConfig)
    
    
    totalFeatures_multi_win = [setup_evaluation(basePath,windowLength=winLength) for winLength in windowLength_multi]
    for elem in totalFeatures_multi_win:

        elem[2].loc[elem[2].Activity == 'walkF','Activity'] = 'walk'
        elem[2].loc[elem[2].Activity == 'walkN','Activity'] = 'walk'
        elem[2].loc[elem[2].Activity == 'jog','Activity'] = 'run'
        elem[2].loc[elem[2].Activity == 'nonlocal','Activity'] = 'housework'
        elem[2].loc[elem[2].Activity == 'sit','Activity'] = 'sitting'

        labels = ['upstairs', 'downstairs', 'housework', 'run', 'sitting', 'standing', 'upslope', 'cycling','walk'] #Renewed Label list

        temp_run_fullfeatures(elem)
        
    


    print('Strart time',start_time,'end time',time.time())
