import string
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso, RidgeCV
import numpy as np
from datetime import datetime
import pandas as pd
from random import shuffle
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy.stats import pearsonr
import warnings
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

'''
Script for Parker decoding and plotting of results. Figure [function]:

Figure 4, Table S1, Figure 5e,f [plot_violin_p]
Figure S10 [plot_chunk2]

For a full list of figures, see README.md.

Function can be run in an interactive python session
after reading the script as

run 'wine_repo/parker_clf.py'

Switch on plt.ion() to see figures.
'''


# set data path
# Source_Data/data: raw data (chemical spectra, varietals, Parker rating)
# Source_Data/res: intermediate results for plotting
pth_dat = Path.cwd() / 'Source_Data'


def get_parker_data(chem_type):
    '''
    get parker scores and chemical spectra
    '''

    # load Parker ratings
    d = np.load(pth_dat / 'data/parker_ratings.npy',
                allow_pickle=True).flat[0]

    # load chemical spectra
    chem = np.load(pth_dat / f'data/{chem_type}.npy',
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

    return x, y, wines


def decodeP(chem_type, decoder='Ridge', return_weights=False,
            shuf=False, alphaL=0.08, alphaR=3.5 * 10 ** 12,
            grids=True):
    '''
    grid search on alphaR now included
    '''

    x, y0, wines = get_parker_data(chem_type)

    # grid search space for alphaR
    ars = []
    ars.append(np.linspace(3 * (10**5), 3.5 * (10**5), 1000))
    ars.append(np.linspace(3 * (10**12), 3.5 * (10**12), 1000))
    ar = np.concatenate(ars)

    if not grids:
        # exploration of grid search range on full data
        reg = RidgeCV(alphas=ar)
        reg.fit(x, y0)
        alphaR = reg.alpha_
        print('optimised alphaR:', alphaR)

    startTime = datetime.now()

    # get data
    y = copy.deepcopy(y0)

    print('x.shape:', x.shape, 'y.shape:', y.shape)

    # shuffle control
    if shuf:
        print('SHUFFLE')
        o = np.arange(len(x))
        shuffle(o)
        y = y[o]

    y_p = []
    y_t = []

    y_p_train = []
    y_t_train = []

    folds = len(x)  # 20
    kf = KFold(n_splits=folds, shuffle=True)

    ws = []
    ws_train = []
    k = 0

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
                reg.fit(train_X, train_y)
                alphaR = reg.alpha_

            clf = Ridge(alpha=alphaR)  # no scale:  3500000000000.
            clf.fit(train_X, train_y)

            y_pred_test = clf.predict(test_X)
            y_pred_train = clf.predict(train_X)

        if k == 0:
            print('train/test samples:', len(train_y), len(test_y))

        y_p.append(y_pred_test)
        y_t.append(y[test_index])
        ws.append(np.array(wines)[test_index])

        y_p_train.append(y_pred_train)
        y_t_train.append(y[train_index])
        ws_train.append(np.array(wines)[train_index])

        k += 1

    # values
    y_p_f = [item for sublist in y_p for item in sublist]
    y_t_f = [item[0] for sublist in y_t for item in sublist]

    y_p_f_train = [item for sublist in y_p_train for item in sublist]
    y_t_f_train = [item[0] for sublist in y_t_train for item in sublist]

    print('time to compute:', (datetime.now() - startTime) / 60)
    print('train: ', np.round(r2_score(y_t_f_train, y_p_f_train), 2),
          'test: ', np.round(r2_score(y_t_f, y_p_f), 2))

    if return_weights:
        reg = RidgeCV(alphas=ar)
        reg.fit(x, y0)
        alphaR = reg.alpha_
        clf = Ridge(alpha=alphaR)
        clf.fit(x, y0)
        print('returning weights; train r**2 = ', clf.score(x, y0))
        return clf.coef_

    print(r2_score(y_t_f, y_p_f))

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

    for chem_type in ['esters', 'oak', 'offFla']:
        x, y, w = get_parker_data(chem_type)

        C[f'parker from {chem_type}'] = decodeP(x, y, w, return_weights=True)

    np.save(pth_dat / 'res/Ridge_weightsS11.npy', C, allow_pickle=True)


def chunks2(chem_type, n_chunks):
    '''
    evaluate decoding performance
    for sections of the spectrum
    just linearly
    '''
    x0, y, w = get_parker_data(chem_type)

    l0 = len(x0[0])

    m2 = np.array_split(range(l0), n_chunks)
    d = dict(zip(range(int(n_chunks)), m2))

    R = []
    for i in d:

        # keep i'th chunk of data only
        x = x0[:, d[i]]

        R.append(decodeP(x, y, w))

    return R


def batch_chunk2(chem_type='oak'):
    '''
    evaluate decoding performance
    for sections of the spectrum
    just linearly
    '''
    res = {}
    for n_chunks in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]:
        res[n_chunks] = chunks2(chem_type, n_chunks)
    np.save(pth_dat / 'res/parker_chunks2.npy', res, allow_pickle=True)


def batch_job(compounds=False):
    '''
    use Ridge for parker decoding
    for 4 different chemical input types
    adapted to leave-one-out
    '''
 
    if compounds:
        chem_types = ['m_oak', 'm_esters', 'm_offFla', 'm_concat']
    else:
    
        chem_types = ['oak', 'esters', 'offFla', 'concat']
                  

    R = {}
    for stype in ['norm', 'shuf']:
        res = {}
        if stype == 'shuf':
            shuf = True
        else:
            shuf = False

        for chem_type in chem_types:
            print(chem_type)
            r = []
            if shuf:
                for i in range(400):
                    r.append(decodeP(chem_type, shuf=shuf))
            else:
                r.append(decodeP(chem_type, shuf=shuf))

            res[chem_type] = r
        R[stype] = res
    
    if compounds:
        np.save(pth_dat / 'res/parker_ridge_m2.npy', R)
    else:
        np.save(pth_dat / 'res/parker_ridge_raw.npy', R)


'''
#######################
# plotting
#######################
'''


def add_panel_letter(k, ax=None):
    '''
    k is the number of the subplot
    '''
    L = string.ascii_lowercase[k - 1]
    if ax is None:
        ax = plt.gca()
    ax.text(-0.1, 1.15, L, transform=ax.transAxes,
            fontsize=16, va='top', ha='right')


def plot_chunk2():
    '''
    Figure S10

    plot decoding r2 per data chunk
    '''

    res = np.load(pth_dat / 'res/parker_chunks2.npy',
                  allow_pickle=True).flat[0]
    x, y, w = get_parker_data('oak')

    l0 = len(x[0])

    fig = plt.figure(figsize=(5, 5))

    k = 1
    for n_chunks in res:
        if k == 1:
            ax = plt.subplot(len(res), 1, k)
        else:
            ax = plt.subplot(len(res), 1, k, sharey=ax)  # sharex=ax,

        m2 = np.array_split(range(l0), n_chunks)
        d = dict(zip(range(int(n_chunks)), m2))
        ms = [int(np.mean(d[i])) for i in d]
        width = len(d[list(d.keys())[0]])
        plt.bar(ms, res[n_chunks], width=width, color='r',
                alpha=0.5, align='center')
        k += 1
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


def plot_violin_p(metric='m', high='oak'):
    '''
    Figure 4, Table S1 (print out), 
    Figure 5e,f (set high='m_oak')

    plotting results from batch_job() as violins;

    Panel a showing scatter of predicted verus true ratings
    Panel b showing decoding results next to null distribution

    variable  # domain or vintage
    train  # display training accuracy
    '''

    if high[0] == 'm':
        R0 = np.load(pth_dat / 'res/parker_ridge_m2.npy',
                 allow_pickle=True).flat[0]
    
        R = {}
        key_order = ['m_esters', 'm_oak', 'm_offFla', 'm_concat']
        R['norm'] = {k: R0['norm'][k] for k in key_order}
        R['shuf'] = {k: R0['shuf'][k] for k in key_order}
    else:
        R0 = np.load(pth_dat / 'res/parker_ridge_raw.npy',
                 allow_pickle=True).flat[0]    
        
        R = {}
        key_order = ['esters', 'oak', 'offFla', 'concat']
        R['norm'] = {k: R0['norm'][k] for k in key_order}
        R['shuf'] = {k: R0['shuf'][k] for k in key_order}

    fig, axs = plt.subplots(ncols=2, figsize=(6, 3))

    columns = ['stype', 'GC type', 'R²', 'r', 'm', 'b', 'pm']

    r = []
    for stype in R:
        for chem_type in R[stype]:
            for tr in R[stype][chem_type]:
                y_t_f, y_p_f0 = tr
                y_p_f = [x[0] for x in y_p_f0]

                m, b = np.polyfit(y_t_f, y_p_f, 1)

                # Alex' measure
                m_t = np.mean(y_t_f)
                pm = np.mean((y_t_f > m_t) == (y_p_f > m_t))

                r2 = np.round(r2_score(y_t_f, y_p_f), 2)
                pr, _ = np.round(pearsonr(y_t_f, y_p_f), 2)
                m = np.round(m, 2)

                r.append([stype, chem_type,
                         r2, pr, m, b, pm])

    df = pd.DataFrame(data=r, columns=columns)

    P = {}
    for chem_type in Counter(df['GC type']):
        D = df[df['GC type'] == chem_type]
        D_s = D[D['stype'] == 'shuf']
        D_n = D[D['stype'] == 'norm']

        P[chem_type] = {}
        for q in ['R²', 'r', 'm', 'pm']:

            # one-sided t-test
            p = np.mean(np.array(list(D_s[q].values) +
                                 [D_n[q].values[0]])
                        >= D_n[q].values[0])

            # Bonferroni correction
            p = p * len(Counter(df['GC type']))

            print(f'chem type: {chem_type}; metric: {q}; '
                  f'score: {D_n[q].values[0]}; '
                  f"p-value: {'{:.1e}'.format(p)}")

            P[chem_type][q] = [D_n[q].values[0], np.round(p, 3)]

    if metric == 'pm':
        lab = 'chance level'
        Y = 0.5

    if metric == 'R²':
        lab = 'baseline'
        Y = 0

    if metric == 'm':
        lab = 'baseline'
        Y = 0

    if metric == 'r':
        lab = 'baseline'
        Y = 0

    sns.violinplot(x="GC type", y=metric, hue="stype",  # y="R²"
                   data=df, inner=None, palette=['k', '.8'],
                   color=".8", split=True, ax=axs[1])

    sns.pointplot(x="GC type", y=metric, hue="stype",
                  data=df,  # split=True,
                  palette=['k', 'gray'], dodge=.432, join=False,
                  ci=None, markers='_', scale=2.5, ax=axs[1])

    add_panel_letter(2, ax=axs[1])
    # annotate plot with p values (choose metric)
    u = 0

    for chem_type in P:

        axs[1].annotate(f"p = {np.round(P[chem_type]['m'][1],2)}",
                        (u - 0.3, 0.19 if high[0] == 'm' else 0.5),
                        fontsize=7)
        u += 1

    axs[1].axhline(y=Y, c='k', linestyle='--', label=lab)
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
    k = 0
    for tr in R['shuf'][high]:
        if k == 0:
            la = 'shuf'
        else:
            la = '_nolegend_'

        y_t_f = np.array(tr[0])
        y_p_f = [t[0] for t in tr[1]]

        m, b = np.polyfit(y_t_f, y_p_f, 1)
        axs[0].plot(y_t_f, m * y_t_f + b, color='.8',
                    linewidth=0.1, alpha=0.5, label=la)

        k += 1

    # normal
    y_t_f = np.array(R['norm'][high][0][0])
    y_p_f = [t[0] for t in R['norm'][high][0][1]]

    axs[0].scatter(y_t_f, y_p_f, s=2,
                   color='k')

    m, b = np.polyfit(y_t_f, y_p_f, 1)
    axs[0].plot(y_t_f, m * y_t_f + b, color='k',
                linewidth=1, label='norm')

    add_panel_letter(1, ax=axs[0])
    axs[0].set_xlabel('true Parker score')
    axs[0].set_ylabel('predicted parker score')
    axs[0].set_title(high)

    custom_lines = [Line2D([0], [0], color='k', lw=2),
                    Line2D([0], [0], color='Gray', lw=2)] 
                       
    axs[0].legend(custom_lines, ['norm', 'shuf'],frameon=False)

    if high[0] == 'm':
        axs[1].tick_params(axis='x', labelrotation=45)
        
    fig.tight_layout()
    fig.tight_layout()
    return P
