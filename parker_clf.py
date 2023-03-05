from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV

import numpy as np
from datetime import datetime
import pandas as pd
from random import random, shuffle
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
from scipy.stats import pearsonr,ttest_1samp,percentileofscore
import warnings
warnings.filterwarnings('ignore')
import string   

'''
Script for Parker decoding and plotting of results. Figure [function]:

Figure 4, Table S1, Figure 5e,f [plot_violin_p]
Figure S10 [plot_chunk2]

For a full list of figures, see README.md.

Function can be run in an interactive python session after reading the script as
run 'wine_repo/parker_clf.py'
'''


def get_parker_data(chem_type):

    '''
    get parker scores and chemical spectra
    '''
    
    # load Parker ratings
    d = np.load('/home/mic/wine/data/parker_ratings.npy',
                allow_pickle=True).flat[0]
    
    # load chemical spectra
    chem = np.load('/home/mic/wine/data/%s.npy' %chem_type,
                   allow_pickle=True).flat[0]
   
    x = []
    y = []

    wines = []
    for wine in chem:
        wines.append(wine)        
        x.append(chem[wine])
        y.append(d[wine])

    x = np.array(x)
    y = np.array(y)      
       
                  
    return x,y,wines


def get_best_data(best = 4):

    ''' pick best segments of each spectrum; then concatenate
    
    best = 4, take the 4 data chunks with best decoding performance
    '''
    
    n_chunks = 50
    #best = 0.1 # best 10 percent of surviving sections
    Ds = []
    
    for chem_type in ['oak'] :#['esters','oak','offFla']
    
        chem = np.load('/home/mic/wine/data/%s.npy' %chem_type,
                    allow_pickle=True).flat[0]
         
        m2 = np.array_split(range(len(chem[list(chem.keys())[0]])),n_chunks)     

        leasts3 = np.load(f'/home/mic/wine/res/leasts_parker_{chem_type}.npy',
                        allow_pickle=True).flat[0]      
          
        # plot most important sections    
        last = list(set(range(n_chunks)) - set(list(leasts3.keys()))) 
        lasts = list(leasts3.keys()) + last
        lasts = lasts[-best:]  

        d = {}
        for i in chem:
            d[i] = np.array(chem[i])[np.concatenate([m2[ii] for ii in lasts])]  

        print(chem_type, len(d[i]))
        Ds.append(d)
        
    chem = {}
    for i in Ds[0]:
        chem[i] = np.concatenate([d[i] for d in Ds])   

    d, _ = excel_to_dict(chem_type)
    x = []
    y = []

    wines = []
    for wine in d:
        wines.append(wine)        
        x.append(chem[wine])
        y.append(d[wine])
         
    
    x = np.array(x)
    y = np.array(y)   
    return x,y,wines


def decodeP(chem_type, decoder = 'Ridge',return_weights=False, 
            shuf = False, alphaL=0.08, alphaR = 3.5 * 10 **12,
            grids=True):

    '''
    grid search on alphaR now included
    '''
    
    x,y0,wines = get_parker_data(chem_type)
    
    # oak: alphaR = 3.5 * 10 **12
    
    # grid search space for alphaR
    ars = []
    ars.append(np.linspace(3*(10**5),3.5*(10**5),1000))
    ars.append(np.linspace(3*(10**12),3.5*(10**12),1000))
    
    
#    ar0 = np.linspace(10**5,10**6,50000)
#    ar1 = np.linspace(10**12,10**13,50000)
    ar = np.concatenate(ars)   

   
    if not grids:
        # exploration of grid search range on full data
        reg = RidgeCV(alphas=ar)
        reg.fit(x,y0)
        alphaR = reg.alpha_
        print('optimised alphaR:', alphaR)

    startTime = datetime.now()
    
    # get data
    y = copy.deepcopy(y0)

    print('x.shape:', x.shape, 'y.shape:', y.shape) 

    input_dim = len(x[0])    
     
    # shuffle control
    
    if shuf:
        print('SHUFFLE')    
        o = np.arange(len(x))
        shuffle(o)
        y = y[o]
        
    # cross validation 
    train_r2 = []
    test_r2 = []
     
    y_p = []
    y_t = []
       
    y_p_train = []
    y_t_train = []       
       
    folds = len(x) # 20
    kf = KFold(n_splits=folds, shuffle=True)    

    ws = []
    ws_train = []
    k=0    
    
    
    for train_index, test_index in kf.split(x):  
        
        train_X = x[train_index]
        test_X = x[test_index]          
            
        
        train_y = y[train_index]  
        test_y = y[test_index]      
        
        if decoder == 'Lasso':
            clf = Lasso(alpha=alphaL)
            clf.fit(train_X, train_y)
            
            y_pred_test = clf.predict(test_X) 
            y_pred_train = clf.predict(train_X)
 
            
        if decoder == 'Ridge':
        
            if grids:
                # do hyperparameter gridsearch for each train split
                reg = RidgeCV(alphas=ar)
                reg.fit(train_X,train_y)
                alphaR = reg.alpha_
                #print(f'alpha fitted for split {k}')
                 
            clf = Ridge(alpha=alphaR) # no scale:  3500000000000.
            clf.fit(train_X, train_y)
            
            y_pred_test = clf.predict(test_X) 
            y_pred_train = clf.predict(train_X)                


        if k==0: 
            print('train/test samples:', len(train_y), len(test_y))

        if decoder == 'NN':
       
            model = Sequential()   
            model.add(Dense(len(y[0]), input_dim=input_dim,
                            activation='linear')) 

            model.compile(optimizer=rm,loss='mse') 

            if k==0: 
                print(model.summary())
                print('')
            
            model.fit(train_X, train_y,
                      batch_size=4,
                      epochs=EPOCHS,
                      verbose=0,
                      callbacks=[callback])
        
            # test performance for this fold
            y_pred_test = model.predict_on_batch(test_X) 
            y_pred_train = model.predict_on_batch(train_X)
       

        y_p.append(y_pred_test)
        y_t.append(y[test_index])
        ws.append(np.array(wines)[test_index])
        
        y_p_train.append(y_pred_train)
        y_t_train.append(y[train_index])
        ws_train.append(np.array(wines)[train_index])        

        k +=1       

    # values
    y_p_f = [item for sublist in y_p for item in sublist]
    y_t_f = [item[0] for sublist in y_t for item in sublist]
    ws_f = [item for sublist in ws for item in sublist]

    y_p_f_train = [item for sublist in y_p_train for item in sublist]
    y_t_f_train = [item[0] for sublist in y_t_train for item in sublist]
    ws_f_train = [item for sublist in ws_train for item in sublist]
    
    print(f'time to compute:', (datetime.now() - startTime)/60)        
    print('train: ',np.round(r2_score(y_t_f_train,y_p_f_train),2),
          'test: ',np.round(r2_score(y_t_f,y_p_f),2))
      
    if return_weights:
        reg = RidgeCV(alphas=ar)
        reg.fit(x,y0)
        alphaR = reg.alpha_            
        clf = Ridge(alpha=alphaR) 
        clf.fit(x,y0)
        print('returning weights; train r**2 = ',clf.score(x,y0)) 
        return clf.coef_
                       
    print(r2_score(y_t_f,y_p_f))      
 
    return y_t_f, y_p_f  
      

'''    
#########
## batch processing
#########    
'''


def get_weights():
    '''
    For fig S6 get weights
    '''
    
    C = {}
    
    for chem_type in ['esters','oak','offFla']:
        x,y,w = get_parker_data(chem_type)
    
        C[f'parker from {chem_type}'] = decodeP(x,y,w,return_weights=True)
    
    np.save('/home/mic/wine/res/Ridge_weightsS11.npy',C,allow_pickle=True)    


def chunks2(chem_type, n_chunks):
    '''
    evaluate decoding performance
    for sections of the spectrum
    just linearly
    '''       
    x0,y,w = get_parker_data(chem_type)
    
    l0 = len(x0[0])    

    m2 = np.array_split(range(l0),n_chunks)    
    d = dict(zip(range(int(n_chunks)),m2))        
     
    R = []
    for i in d:

        # keep i'th chunk of data only            
        x = x0[:,d[i]]

        R.append(decodeP(x,y,w))
              
    return R


def batch_chunk2(chem_type='oak'):

    '''
    evaluate decoding performance
    for sections of the spectrum
    just linearly
    '''   
    res = {}
    for n_chunks in [1,2,3,4,5,6,7,8,9,10,15]:
        res[n_chunks] = chunks2(chem_type,n_chunks)
    np.save('/home/mic/wine/res/parker_chunks2.npy',res, allow_pickle=True)    


def concat_best_data():
    '''
    get performance for concatenating best sections of 3 GC
    '''

    R = {}
    for decoder in ['Ridge','Lasso']:
        res = {}
        for frac in [5,6,7,8]:
            x,y,wines = get_best_data(best=frac)            
            r = []
            for i in range(50):
                r.append(decodeP(x,y,wines,decoder))
            res[frac] = r
        R[decoder] = res

    np.save('wine/res/best_parker.npy',R)    

    return R


def batch_job():
    '''
    use Ridge for parker decoding
    for 4 different chemical input types
    adapted to leave-one-out
    '''

    R = {}
    for stype in ['norm','shuf']:
        res = {}
        if stype == 'shuf':
            shuf = True
        else:
            shuf = False
                
        for chem_type in ['m_esters','m_offFla']:
            print(chem_type)
            r = []
            if shuf:
                for i in range(400):
                    r.append(decodeP(chem_type,shuf=shuf))   
            else:   
                r.append(decodeP(chem_type,shuf=shuf))
                
            res[chem_type] = r
        R[stype] = res    

    np.save(f'/home/mic/wine/res/parker_ridge_m2.npy',R) 
       
    
def chunk_survival_p(chem_type):
    '''
    What is the 10 % of the data that is most informative for domain
    decoding? Survival algorithm:    
    
    1. divide concatenated spectrum into 100 chunks;
    2. for all chunks, compute accuracy for data w/o that chunk 
       (5 times 10 fold cross validation)
    3. remove the data chunk whose removal
       resulted in highest performance; 
    4. for the remaining chunks, remove the next one,
       chosen again by going through all and discarding
       the one whose removal had the highest decoding performance
    5. iterate 4 until only 10 % of chunks are left, 
       which are thus the most important ones of the data for decoding            
    '''

    n_chunks = 50.0 # 100.0
    n_deletions = int(n_chunks - 1)
    x0,y, wines = get_parker_data(chem_type)
    
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
            
            res = []
            for j in range(50):
                train_, test_ = decodeP(x,y,wines,'Lasso')
                res.append(test_)
            Res[i] = np.mean(res) 
                                
        # find chunk whose removal caused max decoding
        discard = max(Res.keys(), key=(lambda k: Res[k]))
        leasts[discard] = Res[discard]
        print(leasts)
        del d[discard]    
        print('######################')
        print(f'{ii} of {n_deletions}')
        print('######################')        

    print(leasts)
    np.save(f'/home/mic/wine/res/leasts_parker_{chem_type}.npy', leasts)   
    
    

              
'''
#######################
# plotting
#######################
'''


def add_panel_letter(k, ax=None):

    '''
    k is the number of the subplot
    '''
    L = string.ascii_lowercase[k-1]
    if ax is None:
        ax = plt.gca()
    ax.text(-0.1, 1.15, L, transform=ax.transAxes,
      fontsize=16,  va='top', ha='right')
      
      
def plot_chunk2():
    '''
    Figure S10
    
    plot decoding r2 per data chunk
    '''

    res = np.load('/home/mic/wine/res/parker_chunks2.npy',
                  allow_pickle=True).flat[0]
    x,y,w = get_parker_data('oak')              
               
    l0 = len(x[0])    
    
    fig = plt.figure(figsize=(5,5))
    
    k = 1
    for n_chunks in res:
        if k == 1:
            ax = plt.subplot(len(res),1,k)
        else:
            ax = plt.subplot(len(res),1,k,sharey=ax)# sharex=ax,
            
        m2 = np.array_split(range(l0),n_chunks)    
        d = dict(zip(range(int(n_chunks)),m2))      
        ms = [int(np.mean(d[i])) for i in d]
        width = len(d[list(d.keys())[0]])
        plt.bar(ms,res[n_chunks],width = width, color = 'r', 
                alpha = 0.5, align='center')
        k+=1
        plt.ylim([-0.001, 0.11])
        plt.ylabel('acc.')
        if k == 12:
            plt.xlabel('oak retention time [a.u.]')
        if k != 12:
            ax.set_xticklabels([])    
        ax2 = ax.twinx()
        ax2.set_ylabel(n_chunks)
        ax2.set_yticklabels([])
        print(n_chunks, max(res[n_chunks]))

    plt.subplots_adjust(top=0.97,
                        bottom=0.11,
                        left=0.11,
                        right=0.9,
                        hspace=0.35,
                        wspace=0.2) 


def plot_violin_p(metric = 'm',high='oak'):
    
    '''
    Figure 4, Table S1, Figure 5e,f
    
    plotting results from batch_job() as violins;
    
    Panel a showing scatter of predicted verus true ratings
    Panel b showing decoding results next to null distribution
    
    variable  # domain or vintage
    train  # display training accuracy
    '''
    
    R0 = np.load('/home/mic/wine/res/parker_ridge_raw.npy',  # was ridge2
                allow_pickle=True).flat[0]
    
    
    if high[0] == 'm':
        R = {}
        key_order = ['m_esters', 'm_oak', 'm_offFla', 'm_concat']   
        R['norm'] = {k : R0['norm'][k] for k in key_order}
        R['shuf'] = {k : R0['shuf'][k] for k in key_order}        
    else:
        R = {}
        key_order = ['esters', 'oak', 'offFla', 'concat']   
        R['norm'] = {k : R0['norm'][k] for k in key_order}
        R['shuf'] = {k : R0['shuf'][k] for k in key_order}    

    fig, axs = plt.subplots(ncols=2, figsize=(6,3))  

    columns=['stype', 'GC type', 'R²', 'r', 'm', 'b','pm']        
    
    r = []
    for stype in R:
        for chem_type in R[stype]:
            for tr in R[stype][chem_type]:
                y_t_f, y_p_f0 = tr  
                y_p_f = [x[0] for x in y_p_f0] 
                
                m, b = np.polyfit(y_t_f, y_p_f, 1)
         
                # Alex' measure
                m_t = np.mean(y_t_f)
                pm = np.mean((y_t_f>m_t) == (y_p_f>m_t))
                
                
                r2 = np.round(r2_score(y_t_f,y_p_f),2)
                pr, _ = np.round(pearsonr(y_t_f,y_p_f),2)
                m = np.round(m,2)                                
 
                r.append([stype, chem_type, 
                         r2, pr, m, b, pm])

    df  = pd.DataFrame(data=r,columns=columns) 
    
    P = {}
    for chem_type in Counter(df['GC type']):
        D = df[df['GC type']==chem_type]
        D_s = D[D['stype']=='shuf']
        D_n = D[D['stype']=='norm']
        
        P[chem_type] = {}
        for q in ['R²', 'r', 'm','pm']:
            
            # one-sided t-test                        
            #_,p = ttest_1samp(D_s[q].values,D_n[q].values[0]) 
            #p = 1 - 0.01 * percentileofscore(D_s[q].values,D_n[q].values[0])
            p =  np.mean(np.array(list(D_s[q].values) + 
                                  [D_n[q].values[0]]) 
                                  >= D_n[q].values[0])
                                  
            # Bonferroni correction                       
            p = p * len(Counter(df['GC type']))
            
            if p>1:
                p = 1       
                       
            print(f'chem type: {chem_type}; metric: {q}; '
                  f'score: {D_n[q].values[0]}; '
                  f"p-value: {'{:.1e}'.format(p)}")
    
            P[chem_type][q] = [D_n[q].values[0],np.round(p,3)]
        

    if metric=='pm':
        lab = 'chance level'
        Y = 0.5
        
    if metric=='R²':
        lab = 'baseline'
        Y = 0
        
    if metric=='m':
        lab = 'baseline'
        Y = 0        
        
    if metric=='r':
        lab = 'baseline'
        Y = 0        
        
      
    sns.violinplot(x="GC type", y=metric, hue="stype", #y="R²"
                   data=df, inner=None,palette=['k','.8'], 
                   color=".8", split=True, ax=axs[1])

    sns.pointplot(x="GC type", y=metric, hue="stype",
                   data=df, #split=True,
                   palette=['k','gray'], dodge=.432, join=False, 
                   ci=None, markers = '_',scale = 2.5, ax=axs[1])  
                   
    add_panel_letter(2, ax=axs[1])           
    # annotate plot with p values (choose metric)    
    u = 0
    
    for chem_type in P:
        #t = int(np.floor(np.log10(np.abs(P[chem_type]['m'][1]))))
        
        axs[1].annotate(f"p = {np.round(P[chem_type]['m'][1],2)}",
                        (u - 0.3, 0.19 if high[0] == 'm' else 0.5), 
                        fontsize=7)       
        u += 1       

              
    axs[1].axhline(y=Y,c='k',linestyle='--',label=lab)
    if metric == 'm':   
        axs[1].set_ylabel('slope')  
                                                        
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles[:2], labels[:2],
      columnspacing=1,
      loc="best", ncol=1, frameon=False).set_draggable(1)   
 
    
    '''
    panel to highlight fit example
    '''              

    # shuffle       
    k =0
    for tr in R['shuf'][high]:
        if k == 0:
            la = 'shuf'
        else:
            la = '_nolegend_'
    
        y_t_f = np.array(tr[0])
        y_p_f = [t[0] for t in tr[1]]
                    
        m, b = np.polyfit(y_t_f, y_p_f, 1)            
        axs[0].plot(y_t_f, m*y_t_f + b, color='.8', 
                 linewidth=0.1, alpha = 0.5,label=la)
                                            
        k+=1      
        
    # normal
    y_t_f = np.array(R['norm'][high][0][0]) 
    y_p_f = [t[0] for t in R['norm'][high][0][1]] 
                   
    axs[0].scatter(y_t_f, y_p_f, s=2,
                color='k')
                
    m, b = np.polyfit(y_t_f, y_p_f, 1)            
    axs[0].plot(y_t_f, m*y_t_f + b, color='k', 
             linewidth=1, label='norm')             
    
    #plt.title('Parker score from oak')
    add_panel_letter(1, ax = axs[0])
    axs[0].set_xlabel('true Parker score')
    axs[0].set_ylabel('predicted parker score')
    axs[0].set_title(high)
    leg = axs[0].legend(frameon=False)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
   
    if high[0] == 'm':     

        axs[1].tick_params(axis='x', labelrotation = 45)
    fig.tight_layout()
    fig.tight_layout()
    return P






    
