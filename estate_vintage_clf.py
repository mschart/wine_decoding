from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,KBinsDiscretizer
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from datetime import datetime
import numpy as np
import pandas as pd
import os 
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from random import shuffle
from copy import deepcopy
import random
import seaborn as sns
from scipy import sparse
from scipy.stats import ranksums, ttest_1samp, ttest_ind, percentileofscore
import string
import matplotlib.image as mpimg

'''
Script to decode estate and vintage from chemical features of wine
and creating figures that show results.

This script contains the analyses, saving intermediate
results in Source Data that are plotted in figures [function] 
(unless raw data is required, as indicated):

Figure 2, 5(c,d) [plot_violin]
Figure 3, S9 [plot_chunks2] 
Figure S3 [plot_best_concat]
Figure S4 [plot_vintage_decoding_per_wine]
Figure S5 [PCA_features] (needs raw data)
Figure S6 [plot_weights]
Figure S7, S8, S11 [plot_all_survival]
Figure S12 [plot_weights_m_abs]
Figure S13 [plot_chunks_m_dists]

See README.md for a list of all figures. Function can be run in an interactive python session after reading the script as
run 'wine_repo/estate_vintage.py'
'''


def get_data(chem_type, vintage = False, exclude = True, get_map=False):

    '''
    load chemical data, transform into x, y format for decoder
    
    chem_type: string, in ['oak', 'esters', 
                           'offFla', 'concat',
                           'm_oak',  'm_esters', 
                           'm_offFla', 'm_concat']
    vintage: if True, create y class vector according to
             vintage, else estate
    '''
    

    chem = np.load('/home/mic/wine/data/%s.npy' %chem_type,
                    allow_pickle=True).flat[0]


    if exclude: # exclude categories with less than 5 samples
        if vintage:
            u = Counter([x.split('_')[-1] for x in chem.keys()])    
            
        else:
            u = Counter([x[0] for x in chem.keys()])             
    
        r = [p for p in u if u[p] > 4]
        if vintage:
            idx = -1
        else:
            idx = 0    
        
        chem2 = {w:chem[w] for w in chem if w.split('_')[idx] in r}
        chem = chem2
        
    print(f'after exclusion there is {len(chem)} wines' if exclude else 
          f'{len(chem)} wines, no exclusion')
    if vintage:
        u = Counter([x.split('_')[-1] for x in chem.keys()])    
        print(len(u), 'vintages',u)
    else:
        u = Counter([x[0] for x in chem.keys()])        
        print(len(u), 'estates',u)
                
        
    cc = list(u.keys())
    chats = dict(zip(cc,range(len(cc))))

    x = []
    y = []

    wines = []

    for wine in chem:
    
        wines.append(wine)
        x.append(chem[wine])
        if vintage:
            y.append(chats[wine.split('_')[-1]])
        else:
            y.append(chats[wine[0]])


    x = np.array(x)
    y = np.array(y)
    
    if get_map:
        return chats
    else:    
        return x, y


def get_best_data(vintage = False, best = 4, rand=False, wines=None):

    ''' pick best segments of each spectrum; then concatenate
    '''
    
    n_chunks = 50
    #best = 0.1 # best 10 percent of surviving sections
    Ds = []
    
    for chem_type in ['oak'] :#['esters','oak','offFla']
    
        chem = np.load('/home/mic/wine/data/%s.npy' %chem_type,
                    allow_pickle=True).flat[0]
         
        m2 = np.array_split(range(
                len(chem[list(chem.keys())[0]])),n_chunks)

        if vintage:
            leasts3 = np.load(f'/home/mic/wine/res'
                              f'/leasts_vintage_{chem_type}.npy',
                              allow_pickle=True).flat[0]      
        else: 
            leasts3 = np.load(f'/home/mic/wine/res/leasts_{chem_type}.npy',
                        allow_pickle=True).flat[0]  
                        
        # plot most important sections    
        last = list(set(range(n_chunks)) - set(list(leasts3.keys()))) 
        lasts = list(leasts3.keys()) + last
        
        if rand == True:
            lasts = random.sample(lasts, best)
        else:    
            lasts = lasts[-best:]  

        d = {}
        for i in chem:
            d[i] = np.array(chem[i])[np.concatenate(
                       [m2[ii] for ii in lasts])]  

        print(chem_type, len(d[i]))
        Ds.append(d)
        
    chem = {}
    for i in Ds[0]:
        chem[i] = np.concatenate([d[i] for d in Ds])   


    if vintage:
        u = Counter([x.split('_')[-1] for x in chem.keys()])    
        print('vintage',u)
    else:
        u = Counter([x[0] for x in chem.keys()])        
        print('estate',u)
    
    
    cc = list(u.keys())

    chats = dict(zip(cc,range(len(cc))))

    x = []
    y = []

    wines = []

    for wine in chem:
        wines.append(wine)
        x.append(chem[wine])
        if vintage:
            y.append(chats[wine.split('_')[-1]])
        else:
            y.append(chats[wine[0]])


    x = np.array(x)
    y = np.array(y)
    
    if wines == None:
        return x, y
    else:
        return x, y, wines    


def perf(y_true,y_predict):
    '''
    Helper function to evaluate classifier performance;
    check if y_true==y_predict; input is single sample, one-hot encoded
    '''
    if len(y_true)!=len(y_predict):
        return "lengths don't match"

    if np.argmax(y_true) == np.argmax(y_predict):   
        return 1
    else:
        return 0


def custom_split(y, reps):
    '''
    Folding for cross validation:
    pick one wine per category at random as test set
    train on rest
    ''' 

    classes = list(Counter(y))  
    splits = []
    for i in range(reps):  
        test_ind = []

        for cl in classes:
            a = random.sample(list(np.where(y==cl)[0]),1)[0]
            test_ind.append(a)  

        train_ind = list(set(range(len(y))) - set(test_ind))
        splits.append([train_ind,test_ind])    
    return splits  


def NN(x,y,decoder='LDA', wines=None, CC = 1.0,
       return_weights=False, shuf = False):
    
    '''
    function to decode estate or vintage (y) from the chemical
    spectrum (x); x,y = get_data(chem_type)
    '''

    print('input dimension:', np.shape(x)) 
    print('chance at ', 1/len(Counter(y)))

    startTime = datetime.now()

    print('x.shape:', x.shape, 'y.shape:', y.shape)

     
    if shuf:
        print('labels are SHUFFLED')
        np.random.shuffle(y)
        
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    train_r2 = []
    test_r2 = []
    acs = [] 
    
    y_p = []
    y_t = []    
    ws = []

    splits = custom_split(y, 100) 
    if decoder == 'NN':
        y = np_utils.to_categorical(y)
        rm = optimizers.RMSprop(learning_rate=0.0001)
        callback = EarlyStopping(monitor='loss', patience=7, 
                                 min_delta=0.01)
                                       
    k=1        
    for train_index, test_index in splits:

  
        sc = StandardScaler()
        train_X = sc.fit_transform(x[train_index])
        test_X = sc.fit_transform(x[test_index])

        train_y = y[train_index]
        test_y = y[test_index]    


        if k==1: 

            print('train/test samples:', len(train_y), len(test_y))
            print('')   

        if decoder == 'LR':
            #CC = 1  #0.00001
            clf = LogisticRegression(C = CC,random_state=0, n_jobs = -1)
            clf.fit(train_X, train_y)
            
            y_pred_test = clf.predict(test_X) 
            y_pred_train = clf.predict(train_X)  
   
        elif decoder == 'LDA':    
        
            clf = LinearDiscriminantAnalysis()
            clf.fit(train_X, train_y)
            
            y_pred_test = clf.predict(test_X) 
            y_pred_train = clf.predict(train_X)  
               
        else:
            return 'what model??'    
 
        res_test = np.mean(test_y == y_pred_test) 
        res_train = np.mean(train_y == y_pred_train)    

        if wines != None:
            y_p.append(y_pred_test)
            y_t.append(y[test_index])
            ws.append(np.array(wines)[test_index])


        ac_test = round(np.mean(res_test),4)
        ac_train = round(np.mean(res_train),4)             
        acs.append([ac_train,ac_test])     
               
        k+=1 
  
    r_train = round(np.mean(np.array(acs)[:,0]),3)
    r_test = round(np.mean(np.array(acs)[:,1]),3)
       
    print('Mean train accuracy:', r_train)
    print('Mean test accuracy:', r_test)
    print(f'time to compute:', datetime.now() - startTime) 
    
    if return_weights:
        if decoder == 'LR':
            clf = LogisticRegression(C = CC,random_state=0, n_jobs = -1)
        else:
            clf = LinearDiscriminantAnalysis()
            
        clf.fit(x, y)

        return clf.coef_ 
    
    if wines != None:
        return y_p, y_t, ws
    else:    
        return np.array(acs)
            

'''
##########
### batch processing
##########
'''


def get_weights(manual=True, decoder='LDA'):
  
    C = {}
    for vintage in [True, False]:
        for chem_type in (['m_concat']
                          if manual else ['esters','oak','offFla']):
            x,y = get_data(chem_type,vintage=vintage)
            if vintage:
                t = f'vintage from {chem_type}'
            else:
                t = f'estate from {chem_type}'

            # keep weights from first split only
            C[t] = NN(x,y,return_weights=True,decoder=decoder)    
    
    if manual:
        s = f'/home/mic/wine/res/LDA_weights_manual_{decoder}.npy'
    else:
        s = '/home/mic/wine/res/LDA_weightsS11.npy'    
    np.save(s,C,allow_pickle=True)        
           


def concat_best_data(vintage=False):
    '''
    get performance for concatenating best sections of 3 GC
    '''

    R = {}
    for decoder in ['LDA','LR']:
        res = []
        for frac in [1,2,3,4]:
            x,y = get_best_data(best=frac, vintage=vintage) 
            res.append(NN(x,y,decoder))
        R[decoder] = res
    if vintage:
        np.save('wine/res/best_vintage.npy',R)    
    else:
        np.save('wine/res/best.npy',R)
    return R


def concat_rand_data(vintage=True):
    '''
    get performance for concatenating best sections of 3 GC
    '''

    R = {}
    for decoder in ['LDA']:
        res = {}
        for frac in [5,10,15,20]:
            r = []
            for i in range(25):                
                x,y = get_best_data(best=frac, vintage=vintage,rand=True) 
                r.append(NN(x,y,decoder))
            res[frac] = r    
        R[decoder] = res
    if vintage:
        np.save('wine/res/rand_vintage.npy',R)    
    else:
        np.save('wine/res/rand.npy',R)
    return R


def batch_job(manual =True):
    '''
    use LDA/LR for vintage/estate decoding
    change vintage/estate in x,y function
    '''
    
    m_types = ['m_esters', 'm_oak', 'm_offFla', 'm_concat', 'm_all']
    gc_types = ['concat','esters','oak','offFla']    
    
    for dr in [False]:#True, 
        for vint in [True, False]:
            R = {}
            for decoder in ['LDA','LR']:
                res = {}
                for chem_type in (m_types if manual else gc_types):
                    print('')
                    print(chem_type)
                    x,y = get_data(chem_type, vintage =vint) 
                    res[chem_type] = NN(x,y,decoder=decoder,dim_red = dr)
                R[decoder] = res   
                     
            np.save(f'/home/mic/wine/res/'
                    f'{"m" if manual else "R"}_dr_{dr}'
                    f'_vintage_{vint}.npy', R)         


def chunk_survival(chem_type, loosers = False, vintage = False):
    '''
    What is the 10 % of the data that is most informative for estate
    decoding? Survival algorithm:    
    
    1. divide concatenated spectrum into 100 chunks;
    2. for all chunks, compute accuracy for data w/o that chunk 
       (5 times 10 fold cross validation)
    3. remove the data chunk whose removal
       resulted in highest performance; 
    4. for the remaining chunks, remove the next one,
       chosen again by going through all and discarding
       the one whose removal had the highest decoding performance
    5. iterate 4 until only 10% of chunks are left, 
       which are thus the most important ones of the data for decoding  
       
    If losers = True, the data chunk is removed, whose removal resulted in the lowest   
    performance
    '''
    
    x0,y = get_data(chem_type, vintage = vintage)#'esters'

    n_chunks = x0.shape[-1] if chem_type[:2] == 'm_' else 50.0
    print(n_chunks, 'chunks')
    
    n_deletions = int(n_chunks - 1)    

    
    l0 = len(x0[0])    

    m2 = np.array_split(range(l0),n_chunks)    
    d = dict(zip(range(int(n_chunks)),m2))
        
    leasts = {}    
    for ii in range(n_deletions): # (80)   

        Res = {}       
        for i in d:

            # remove i'th chunk from data    
            d2 = deepcopy(d)                        
            del d2[i]            
            x = x0[:,np.concatenate(list(d2.values()))]
            
            print(len(x), len(x[0]), len(list(d2.values())))

            Res[i] = np.mean(NN(x,y,'LDA')[:,1]) 
        
        if loosers:
            # find chunk whose removal caused min decoding
            discard = min(Res.keys(), key=(lambda k: Res[k]))            
        else:
            # find chunk whose removal caused max decoding
            discard = max(Res.keys(), key=(lambda k: Res[k]))
        
        leasts[discard] = Res[discard]
        print(leasts)
        del d[discard]    
        print('######################')
        print(f'{ii} of {n_deletions}')
        print('######################')        

    print(leasts)
    if loosers:
        np.save(f'/home/mic/wine/res/leasts_vintage_{vintage}'
                f'_loosers_{chem_type}.npy', leasts)
    else:
        if vintage:
            p = f'/home/mic/wine/res/leasts_vintage_{chem_type}.npy'
        else:
            p = f'/home/mic/wine/res/leasts_{chem_type}.npy'
        np.save(p, leasts)        
   

def chunks2(chem_type, vintage = False, shuf=False, n_chunks=50,
            cla = 'LDA'):
    '''
    evaluate decoding performance
    for sections of the spectrum
    just linearly
    '''
    
    x0,y = get_data(chem_type,vintage=vintage)
    n_chunks = x0.shape[-1] if chem_type[:2] == 'm_' else 50.0
    print(n_chunks, 'chunks') 
    
    l0 = len(x0[0])    

    m2 = np.array_split(range(l0),n_chunks)    
    d = dict(zip(range(int(n_chunks)),m2))
        
    leasts = {}    
     
    Res = {}
    for i in d:
        
        if chem_type == 'm_concat' and cla == 'LDA':
            if i == 29:
                continue
        
        # keep i'th chunk of data only            
        x = x0[:,d[i]]
        
        print(len(x), i, len(list(d.values())))

        Res[i] = np.mean(NN(x,y,cla, shuf=shuf)[:,1]) 
              
    return Res


def batch_chunk2():
    for n_chunks in [5.0,10.0,20.0,50.0]:
        R = {}
        for chem_type in ['esters','oak','offFla']:
            R[chem_type] = chunks2(chem_type,n_chunks)
            np.save(f'/home/mic/wine/res/'
                    f'R_chunks2_vintage_{n_chunks}.npy', R)
      
            
def batch_chunk2_m_parallel(shuf, vintage, cla='LDA'):
 
    nrand = 150
    R = {}
    chem_type = 'm_concat'

    r = []
    for i in range(nrand):
        r.append(chunks2(chem_type, cla=cla, vintage=vintage,
                         shuf=shuf))
                           
    R[chem_type] = r

    p = f'/home/mic/wine/res/c2_m_v{vintage}_s{shuf}_{cla}.npy'
    np.save(p, R)           
            


def several_chem_types(chem_types):
    #chem_types = ['full_A', 'full_B', 'full_C']
    info = []
    for chem_type in chem_types:
        for algo in ['LDA', 'LR']:
            for vintage in [False, True]:
                print(chem_type, algo, 'vintage:',vintage)
                x,y = get_data(chem_type, vintage = vintage)
                r = NN(x,y,decoder=algo)
                info.append([chem_type, algo, 
                    'vintage' if vintage else 'estate', r])

    print(info)
    return info

'''
#####
# plotting
#####
'''


def add_panel_letter(k,ax =None):

    '''
    k is the number of the subplot
    '''
    L = string.ascii_lowercase[k-1]
    if ax is None:
        ax = plt.gca()
    ax.text(-0.1, 1.15, L, transform=ax.transAxes,
      fontsize=16,  va='top', ha='right')#fontweight='bold',

    
    
def plot_best_concat():
    
    '''
    Figure S3
    
    plot of test accuracy for concatenating the 
    best segments from the survival algo of 
    all 3 GCs

    Plot of vintage decoding from random concatenation
    '''    
    
    fig = plt.figure(figsize=(7,7))    
    
    
    cls = {'estate':1/7., 'vintage':1/12.}    
  
    k = 1
    for variable in cls:
        if k==1:
            ax = plt.subplot(2,2,k)
        else:
            ax = plt.subplot(2,2,k, sharex = ax, sharey = ax)     
    
        add_panel_letter(k)
        if variable == 'vintage':
            R = np.load(f'/home/mic/wine/res/best_vintage.npy',
                        allow_pickle=True).flat[0]   
        else:         
            R = np.load(f'/home/mic/wine/res/best.npy',
                        allow_pickle=True).flat[0] 
                        

        columns=['decoder', '#bins', 'train','test']        
        bests = [1,2,3,4]
        r = []   
        
        for decoder in R:
            for best in range(len(R[decoder])):
            
                accs = R[decoder][best][:,1]
                rands = [cls[variable]]*len(accs)
                print(variable,decoder,best+1,"d'", 
                      np.round(cohend(accs,rands),2),
                      'mean', np.round(np.mean(accs),2))        
            
                for tr in range(len(R[decoder][best])):
                    r.append([decoder, best + 1,
                              R[decoder][best][tr][0]*100,
                              R[decoder][best][tr][1]*100]) 
                    
                    
        df  = pd.DataFrame(data=r,columns=columns)    
        plt.axhline(y=0.08, linestyle = '--', c='k', 
                    label = f'chance level')  
            
        sns.violinplot(x="#bins", y="test", hue="decoder",
                       data=df, inner=None, color=".8", split=True)
                       
        sns.pointplot(x="#bins", y="test", hue="decoder",
                       data=df, #split=True,
                       color='k', dodge=.432, join=False, 
                       ci=None, markers = '_')                           
      
      
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3],
        columnspacing=1,
        loc="best", ncol=1, frameon=True).set_draggable(1)  

        plt.xlabel('number of best 2% bins \n concatenated from 3 GCs')    
        plt.title(f'{variable.capitalize()}'
                   ' decoding \n using best concatenated GC')    
        plt.ylabel('accuracy [%]')

        k+=1
     
    ax = plt.subplot(2,2,3)
    add_panel_letter(3)
    
    R = np.load(f'/home/mic/wine/res/rand_vintage.npy',
                allow_pickle=True).flat[0]                       

    columns=['decoder', '#bins', 'test']        
    bests = [1,2,3,4]
    r = []   
    
    for decoder in R:
        for best in R[decoder]:
        
            accs = np.array(R[decoder][best])
            accs = np.reshape(accs,[25*50,2])[:,1]
            rands = [1/12.]*len(accs)
            print(decoder,best,"d'", 
                  np.round(cohend(accs,rands),2),
                  'mean', np.round(np.mean(accs),2))        
        
            for tr in accs:
                r.append([decoder, best, tr*100])
                 
                
    df  = pd.DataFrame(data=r,columns=columns)    
    plt.axhline(y=0.08, linestyle = '--', c='k', 
                label = f'chance level')  
        
    sns.violinplot(x="#bins", y="test",
                   data=df, inner=None, color=".8", split=True)
                   
    sns.pointplot(x="#bins", y="test",
                   data=df, #split=True,
                   color='k', dodge=.432, join=False, 
                   ci=None, markers = '_')                           
  
  
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3],
    columnspacing=1,
    loc="best", ncol=1, frameon=True).set_draggable(1)  
 
  
    plt.xlabel('number of 2% bins from oak')    
    plt.title(f'Vintage'
               ' decoding \n concatenating random segments')    
    plt.ylabel('accuracy [%]')
    plt.tight_layout()


def cohend(d1, d2):
    '''calculate Cohen's d for independent samples
    '''
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s



def plot_violin(dr = False):
    
    '''
    figure 2 and figure 5c, d
    
    plotting results from batch_job() as violins
    
    variable  # estate or vintage
    train  # display training accuracy
    dr  # use PCA dim reduction
    '''    
    
    nd = {'estate': False,'vintage': True}    
    cls = {'estate':100/7., 'vintage':100/12.}  # chance levels
        
    dfs = []
    for manual in [False, True]:
        k = 1 
        fig = plt.figure(figsize=(6,3))
        for variable in nd:
            if k==1:
                ax = plt.subplot(1,2,k)
            else:
                ax = plt.subplot(1,2,k, sharey = ax)       
        
            qq = (f'/home/mic/wine/res/{"m" if manual else "R"}'
                  f'_dr_{dr}_vintage_{nd[variable]}.npy')
                  
            R = np.load(qq,
                        allow_pickle=True).flat[0]  


            columns=['decoder', 'chem type', 'accuracy [%]']        
            
            # bring into df shape for plotting
            r = []
            for decoder in R:
                for chem_type in R[decoder]:
                    if chem_type == 'm_all':
                        continue
                    accs = R[decoder][chem_type][:,1]
                    
                    _,p = ttest_1samp(accs,cls[variable])
                   
                    
                    #p = p # * 4 * 2  #Bonferroni correction
                    
                    print(variable, decoder,chem_type,"p", 
                          np.round(p,4),
                          'mean', np.round(np.mean(accs),2))
                
                    for tr in R[decoder][chem_type][:,1]: 
                        r.append([decoder, 
                                  'concat' if chem_type == 'chem_concat' 
                                  else chem_type, tr*100]) 

                       
            df  = pd.DataFrame(data=r,columns=columns)
            print (manual, variable)
            #print(df)
            dfs.append(df)
            plt.axhline(y=cls[variable], linestyle = '--', c='gray', 
                        label = f'chance level')  
                
            sns.violinplot(x="chem type", y="accuracy [%]", hue="decoder",
                           data=df, inner = None, 
                           color=".8", split=True,meanline=True, bw=.1) 
                                        
                           
            sns.pointplot(x="chem type", y="accuracy [%]", hue="decoder",
                           data=df, #split=True,
                           color='k', dodge=.432, join=False, 
                           errorbar=None, markers = '_', 
                           scale = 1.5)
              
            if manual:                                     
                plt.xticks(rotation=45)                              
            add_panel_letter(k)
            plt.title(f'{variable} decoding')
            
                 
            handles, labels = ax.get_legend_handles_labels()
            if k == 2:
                ax.legend(handles[:3], labels[:3],
                  columnspacing=1,
                  loc="center left", ncol=1, frameon=False, 
                  fontsize='10').set_draggable(1) 
            else: 
                ax.get_legend().remove()
                 
            k+=1       
                
        fig.tight_layout()


def plot_chunks2(n_chunks, vintage=False):

    '''
    plot decoding per chunk
    place example spectrum on top
    
    n_chunks = 50.0 for figure 3    
    
    For figure S9:
    
    for n_chunks in [5.0,10.0,20.0,50.0]: 
        for vintage in [True,False]:
    

    ''' 
    if vintage:
        R = np.load(f'/home/mic/wine/res/R_chunks2_vintage_{n_chunks}.npy',
                    allow_pickle=True).flat[0]     
        variable = 'vintage'            
    else:
        R = np.load(f'/home/mic/wine/res/R_chunks2_{n_chunks}.npy',
                    allow_pickle=True).flat[0]  
        variable = 'estate'   
             
    fig = plt.figure(figsize=(5,4))
    
    k = 1
    for chem_type in R:
        
        if k == 1: 
            ax1 = plt.subplot(len(R), 1, k)
        else:
            ax1 = plt.subplot(len(R), 1, k,sharey=ax1)     
        
        
        # decoding per chunk
        pos = range(len(R[chem_type]))
        ys = list(R[chem_type].values())
        plt.bar(pos,np.array(ys)*100, color = 'r', 
                alpha = 0.5, align='edge')
        if k == len(R):
            ax1.set_xlabel('data section number \n'
                           f' [one bar is {100/len(R[chem_type])}%'
                           '  of the spectrum]')
        ax1.axhline(y= 8 if vintage else 14, linestyle = '--',
                    label = 'chance', color = 'grey')
                    
        ax1.legend(fontsize=8, frameon=False)                               
        ax1.set_ylabel(f'accuracy [%] \n LDA, {variable} ', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        plt.title(chem_type)

        # load example spectrum                
        ax2 = ax1.twinx()               
        x0,_ = get_data(chem_type)
        xs = np.linspace(0,len(R[chem_type]),len(x0[0]))
        ys = x0[0]
        plt.plot(xs,ys,c=u'#1f77b4', alpha = 0.8)  
        ax2.tick_params(axis='y', labelcolor=u'#1f77b4')  
        ax2.set_ylabel(' GC response \n [a.u.]', color=u'#1f77b4')
        k+=1
        
    plt.tight_layout()    
           

def plot_chunks_m_dists(cla='LDA', sigl=0.01): 

    '''
    figure S13
    
    plot decoding accuracy distributions per compound; 
    one plot for vintage, one for estate
    '''
    # get compound names
    
    rs = '/home/mic/wine/data/32_compound_names.npy'
    labs = list(np.load(rs, allow_pickle=True))

    if cla == 'LDA':
        del labs[29]  # deleting comp that is zero too often for LDA
    
    fig, axs = plt.subplots(nrows=2, figsize=(12,8), sharex=True, 
                            sharey=True)
                            
    i = 0
    for target in ['estate', 'vintage']:
        columns = ['compound', 'accuracy [%]', 'shuf', 'it']
        r = [] 
        ys = []
        for shuf in [False, True]:
                
            p = ('/home/mic/wine/res/'
                 f'c2_m_v{True if target == "vintage" else False}'
                 f'_s{shuf}_{cla}.npy')
               
            q = np.load(p,allow_pickle=True).flat[0]['m_concat']
            comps = list(q[0].keys())      
            y = np.array([list(x.values()) for x in q]).T
            ys.append(y)
            k = 0
            
            # minimal random perturbation for plotting compound 29
            w = np.random.rand(150)/100
            for comp in comps:
                for it in range(len(y[k])):
                    r.append([comp, y[k][it]*100 + w[it], 
                              'shuf' if shuf else 'norm', it]) 
                          
                k += 1
        
        # get ttest p-values per compound
        sc,ps = ttest_ind(ys[0].T, ys[1].T, alternative = 'greater')
        print(target)
        print('ttest scores', sc)
        print('ttest ps', ps)
        print('')
        
        # indicate insignificant differences of distributions
        for j in range(len(ps)):
            if ps[j] > sigl:          
                axs[i].axvline(x=j, c='r', linestyle='--',
                               linewidth = 0.5)          
                    
        df = pd.DataFrame(data=r,columns=columns)

        v = sns.violinplot(x="compound", y="accuracy [%]", hue="shuf",
                       data=df, inner = None, ax = axs[i], 
                       palette='Greys', split=True,meanline=True, bw=.1) 
    
        jjj = 0         
        for jj in axs[i].collections:
                   
            c = 'k' if jjj%2 == 0 else 'r'                   
            jj.set_edgecolor(c)
            jj.set_facecolor(c)
            

            jjj += 1
            
        axs[i].set_title(target)
        axs[i].set_xticklabels(labs, rotation = 90)
        
        handles = [mpatches.Patch(color='k', label='true labels'),
                   mpatches.Patch(color='r', label='shuffled labels')] 
        
        if i > 0:
            axs[i].legend(handles=[Line2D([0], [0], 
                          color='r', lw=1, linestyle='--', 
                          label='no sig. difference')], 
                          frameon=False, loc='upper center').set_draggable(True)
        else:
            axs[i].legend(handles=handles, ncols=2, 
                          loc='upper center').set_draggable(True)    
   
        i += 1

    fig.tight_layout()

    
def plot_survival(chem_type,vintage=False,ax=None,ax2=None, 
                  savef=False, losers=False):
    '''
    survival of the fittest data segment
    '''

    
    if losers:
        p = (f'/home/mic/wine/res/losers/'
             f'leasts_vint_{vintage}_loosers_{chem_type}.npy')
    else:
        if vintage:
            p = f'/home/mic/wine/res/leasts_vintage_{chem_type}.npy'
        else:
            p = f'/home/mic/wine/res/leasts_{chem_type}.npy'

    leasts3 = np.load(p, allow_pickle=True).flat[0]  
    n_chunks = len(leasts3) + 1                  
     
    #plt.ion()       
    
    if ax == None:   
        f, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]},
                               figsize=(5,4))

    ax.plot(range(len(leasts3)),np.array(list(leasts3.values()))*100,
            label=chem_type)
    ax.legend()
    print('max score:', max(list(leasts3.values())))
    
    
    def my_formatter(q, pos=None):
        if chem_type == 'manual':
            return (n_chunks - q)
        else:
            return (n_chunks - q)*100/ n_chunks   
        
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(my_formatter))        
    #ax.set_title(f'"survival of the fittest" GC sections; {chem_type}')
    if chem_type == 'manual':
        ax.set_xlabel(f"Best #features for decoding")
    else:
        ax.set_xlabel(f"{['Worst' if losers else 'Best'][0]} fraction "
                   "of spectrum for decoding [%]")
    if vintage: 
        ax.set_ylabel('LDA accuracy [%] \n for vintage')    
    else:
        ax.set_ylabel('LDA accuracy [%] \n for estate')

    # plot most important sections    
    last = list(set(range(n_chunks)) - set(list(leasts3.keys())))
    lasts = list(leasts3.keys())[-(int(n_chunks* 0.1) - 1):] + last
 
    x0,y = get_data(chem_type)
    
    l0 = len(x0[0])    

    m2 = np.array_split(range(l0),n_chunks)    
    d = dict(zip(range(int(n_chunks)),m2))   

    cmap = matplotlib.cm.get_cmap('Reds')
    cn = np.arange(len(lasts))/max(np.arange(len(lasts))) 

    # map x axis rentention time to data section numbers
    xs = np.linspace(0,int(n_chunks),len(x0[0]))*100/ n_chunks 
 
    p = 0      
    for i in lasts:
        # get boundaries 
        ax2.axvspan(xs[d[i][0]],xs[d[i][-1]], 
                    alpha=0.5, color=cmap(cn[p]),
                    label = i *2)
        p += 1
        
    # illustrate most important parts of GC
    ys = x0[0]
    ax2.plot(xs,ys,c=u'#1f77b4', alpha = 0.8) 
    ax2.tick_params(axis='y', labelcolor=u'#1f77b4')  
    ax2.set_ylabel(' GC response \n [a.u.]', color=u'#1f77b4')
    ax2.set_xlabel('retention time in [%]')
    #ax2.legend(fontsize=6, ncol =1,loc=0).set_draggable(1)
    plt.tight_layout()    
    return ax,ax2


def plot_all_survival(losers=False):

    '''
    figure S7 (S8 with losers=True)
    
    Illustrate survival algorithm results
    '''

    fig = plt.figure(figsize=(9, 10))
    outer = gridspec.GridSpec(3, 2,
                            top=0.975,
                            bottom=0.07,
                            left=0.15,
                            right=0.93,
                            hspace=0.3,
                            wspace=0.6)                             

    k = 0
    for chem in ['oak','esters','offFla']:
        for vintage in [True, False]:
            
            inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[k],
                    hspace=0.6,
                    wspace=0.6)

            ax = plt.Subplot(fig, inner[0])
            ax2 = plt.Subplot(fig, inner[1])
                        
            plot_survival(chem,vintage=vintage,ax=ax,ax2=ax2,
                          losers=losers)
 
            fig.add_subplot(ax)                              
            fig.add_subplot(ax2)
            plt.tight_layout()
            k+=1
  
  
def new_code(L):

    # remapping the wine estate letters
    
    m = dict(zip(['V','A','S','F','T','G','B'],
                 ['A','B','C','D','E','F','G']))
    return m[L]  
    
    
def plot_vintage_decoding_per_wine():

    '''
    Figure S4
    
    LDA vintage decoding per wine, ordered by accuracy
    '''


    # 20 distinct colors
    cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
            '#ffffff', '#000000']

    fig = plt.figure(figsize=(9,2))
    R = np.load(f'/home/mic/wine/res/best_20percent_oak_vintage.npy',
                allow_pickle=True).flat[0]  
  
    ws = R['ws']
    y_p = R['y_p']
    
    wines = Counter([item for sublist in ws for item in sublist])
    wines = list(wines.keys())
  
    D = {}
    for wine in wines:
        D[wine] = []
  
    t = 0
    for tr in y_p:
        k = 0
        for w in tr:        
            wine = ws[t][k]
            if w == k:
                D[wine].append(1)
            else:
                D[wine].append(0)
            k += 1    
        t += 1
    
    D_mean = {}
    for i in D:
        D_mean[i] = np.mean(D[i])
    
    D2 = dict(sorted(D_mean.items(), 
                     key=lambda item: item[1]))                             
        
    # color list by vintage
    
    vints = list(Counter([x[-4:] for x in D2.keys()]).keys())
    vc = dict(zip(vints,cols[:len(vints)]))
    
    cs = [vc[x[-4:]] for x in D2.keys()]
    
    plt.bar(range(len(D2)),np.array(list(D2.values()))*100, color = cs)
    plt.show()
    ax = plt.gca()
    ax.set_xticks(range(len(D2)))        
    
    ws = ['_'.join([new_code(wine.split('_')[0]),wine.split('_')[1]]) for
          wine in D2.keys()]

    ax.set_xticklabels(D2.keys(),rotation=90, fontsize=8)
    
    k = 0
    for ticklabel in plt.gca().get_xticklabels():
        ticklabel.set_color(cs[k])        
        k += 1
       
    plt.title('vintage decoding from best 20% of oak; LDA')
    plt.ylabel('mean test accuracy')
    plt.xlabel('wine')  
    plt.tight_layout()     
    return D2                
  
  
def plot_weights():

    '''
    Figure S6
    
    Illustrate example decoder weights for eststate, 
    vintage and Parker decoding.
    '''
    
    D0 = np.load('/home/mic/wine/res/LDA_weightsS11.npy',
                allow_pickle=True).flat[0]
    D1 = np.load('/home/mic/wine/res/Ridge_weightsS11.npy',
                allow_pickle=True).flat[0]
             
    D = {}
    
    for i in D0:
        D[i] = D0[i][0]  # pick first estate or vintage
    for i in D1:
        D[i] = D1[i][0]  # [0] just for format
           
                
    plt.figure(figsize=(7,7))
    
    k = 1
    for i in D:
        ax = plt.subplot(3,3,k)
        plt.plot(D[i])
        plt.ylabel('weights [a.u.]')
        plt.xlabel('retention time [a.u.]')
        plt.title(i+'\n')
        k +=1        
        
    plt.tight_layout()
    


def PCA_features(ev=False):

    '''
    if ev, scree plot, i.e. eigenvalue per component
    '''
    
    fig, axs = plt.subplots(ncols=2,figsize=(6,3))
    
    for k in range(2):
        for chem_type in ['m_concat', 'concat', 'oak', 'offFla', 'esters']:

            x,y =  get_data(chem_type)
            pca = PCA()    

            pca.fit(x)
            
            if k == 1:
                axs[k].plot(pca.explained_variance_,label=chem_type)
                axs[k].set_ylabel('eigenvalue')   
            else:
                axs[k].plot(np.cumsum(pca.explained_variance_ratio_)*100,
                         label=chem_type)
                axs[k].set_ylabel('cummulative \n var explained [%]')
        
            axs[k].set_xlabel(f'PCs')
            axs[k].legend(frameon=False)
    
    fig.tight_layout()        
  
  
def plot_weights_m_abs():

    '''
    Figure S12
    
    get estate and vintage weights plot for m_concat
    '''

    # get compound names
    rs = '/home/mic/wine/data/32_compound_names.npy'
    labs = list(np.load(rs, allow_pickle=True))    

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,6), sharex=True)
    
    r = 0
    for decoder in ['LR']:
        c = 0
        for vintage in [True, False]:

            # get weights
            x,y = get_data('m_concat',vintage=vintage)
            w = NN(x,y,return_weights=True, decoder=decoder)

            ax[c].bar(range(w.shape[1]),np.mean(abs(w),axis=0))
            ax[c].set_xlabel('compounds')
            ax[c].set_xticks(range(w.shape[1]))
            ax[c].set_xticklabels(labs, rotation=90)
            ax[c].set_ylabel('absolute weights \n class average')
            ax[c].set_title(f'{decoder},' 
                    'f{"vintage" if vintage else "estate"}')   
            
            c+=1

    fig.tight_layout() 


