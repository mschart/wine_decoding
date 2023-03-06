from pathlib import Path
import dataframe_image as dfi
import distinctipy
import string
import matplotlib.image as mpimg
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11})

'''
Script to analyse and plot dimensionality reduction results,
raw spectra and varietal info (grpae type ratios),
shown in the following figures [function], using Source Data
files:

Figure 1 [plot_tile_main]
Figure 5a,b [plot_tile_chem32]

Figure S2 [plot_tile_supp]
Table S2 [varietals_table]

See README.md for a the full list of figures.

Function can be run in an interactive python session
after reading the script as

run 'wine_repo/embed.py'

'''

# set data path
# Source_Data/data: raw data (chemical spectra, varietals, Parker rating)
# Source_Data/res: intermediate results for plotting
pth_dat = Path.cwd() / 'Source_Data'


def new_code(L):

    # remapping the wine estate letters
    m = dict(zip(['V', 'A', 'S', 'F', 'T', 'G', 'B', 'M'],
                 ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'F']))
    return m[L]


def add_panel_letter(k, ax=None):
    '''
    k is the number of the subplot
    '''
    L = string.ascii_lowercase[k - 1]
    if ax is None:
        ax = plt.gca()
    ax.text(-0.1, 1.15, L, transform=ax.transAxes,
            fontsize=16, va='top', ha='right')  # fontweight='bold',


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def dim_red(algo, chem_type, ax=None, fig=None,
            d3=False, idx=0, rs=8, remap=True):
    '''
    algo in umap, tSNE
    '''

    d = np.load(pth_dat / f'data/{chem_type}.npy',
                allow_pickle=True).flat[0]

    print(len(d), 'wines')
    u = Counter([x.split('_')[0] for x in d.keys()])
    print(' ')
    print(len(u), 'estates')
    print(u)
    u = Counter([x.split('_')[-1] for x in d.keys()])
    print(len(u), 'vintages')
    print(u)

    print('remap wine codes?')

    c0 = ['V', 'A', 'S', 'B', 'F', 'T', 'G', 'M']
    c00 = sorted(list(Counter([x.split('_')[0] for x in d.keys()]).keys()))

    cols = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'y']
    if not set(c00).issubset(set(c0)):
        print('new colors')
        c0 = c00
        cols = distinctipy.get_colors(len(c0))

    Cs = dict(zip(c0, cols))

    x = []
    wines = []
    for wine in d:
        x.append(d[wine])
        wines.append(wine)

    scx = StandardScaler()  # baseline-subtract, division by std
    x = np.array(scx.fit_transform(x))

    print('number of features:', len(x[0]))

    # TSNE, umap, Isomap, locally_linear_embedding
    if d3:
        n_comp = 3
    else:
        n_comp = 2

    if algo == 'umap':
        x_embedded = umap.UMAP(
            n_components=n_comp, n_neighbors=60,
            random_state=rs).fit_transform(x)
    if algo == 'tSNE':
        x_embedded = TSNE(n_components=n_comp, perplexity=30,
                          random_state=rs).fit_transform(x)
    if algo == 'PCA':
        pca = PCA(n_components=n_comp)
        pca.fit(x)
        x_embedded = pca.transform(x)

    if ax is None:
        if d3:
            ax = plt.axes(projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(4, 4))

    xs = x_embedded[:, 0]
    ys = x_embedded[:, 1]
    Bc = [Cs[wine[0]] for wine in wines]

    if d3:
        zs = x_embedded[:, 2]

    if d3:
        ax.scatter(xs, ys, zs, c=Bc, depthshade=False, s=10)

    for i in range(len(x_embedded)):
        wine = wines[i]
        col = Cs[wine[0]]
        if remap:
            wine = '_'.join([new_code(wine.split('_')[0]),
                             wine.split('_')[1]])
        else:
            wine = wines[i]

        if d3:
            ax.text(x_embedded[i, 0], x_embedded[i, 1], x_embedded[i, 2],
                    '  ' + wine, size=6, zorder=1, color=col)
        else:
            ax.plot(-x_embedded[i][0], -x_embedded[i][1],
                    color=col, linestyle='', marker='o', markersize=4)
            ax.annotate('  ' + wine, (-x_embedded[i, 0], -x_embedded[i, 1]),
                        fontsize=6, color=col)

    if d3:
        ax.set_xlabel(f'{algo} dimension 1 [a.u.]')
        ax.set_ylabel(f'{algo} dimension 2 [a.u.]')
        ax.set_zlabel(f'{algo} dimension 3 [a.u.]')
        set_axes_equal(ax)

    else:
        ax.set_xlabel(f'{algo} dimension 1 [a.u.]')
        ax.set_ylabel(f'{algo} dimension 2 [a.u.]')

    ax.set_title(chem_type)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()


def dim_red_grape_ratio(algo, ax=None, rs=8):
    '''
    Get grape ratios per wine (varietals) and embedd
    in low dimensional space

    algo: string, algos = ['tSNE', 'umap']
    '''

    d = np.load(pth_dat / 'data/varietals.npy',
                allow_pickle=True).flat[0]

    c0 = ['V', 'A', 'S', 'B', 'F', 'T', 'G']
    cols = ['b', 'g', 'r', 'c', 'm', 'k', 'y', ]
    Cs = dict(zip(c0, cols))

    x = []
    wines = []
    for wine in d:
        x.append(d[wine])
        wines.append(wine[0] + '_' + wine[1:])

    x = x[:-1]  # mysterious nan as last entry
    wines = wines[:-1]
    scx = StandardScaler()  # baseline-subtract, division by std

    x = np.array(scx.fit_transform(x))

    # TSNE, umap, PCA
    if algo == 'umap':
        x_embedded = umap.UMAP(spread=100,
                               random_state=rs).fit_transform(x)
    if algo == 'tSNE':
        x_embedded = TSNE(n_components=2, perplexity=30,
                          random_state=rs).fit_transform(x)
    if algo == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(x)
        x_embedded = pca.transform(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    for i in range(len(x_embedded)):
        wine = wines[i]
        col = Cs[wine[0]]
        ax.plot(-x_embedded[i][0], -x_embedded[i][1],
                color=col, linestyle='', marker='o', markersize=4)

        wine = '_'.join([new_code(wine.split('_')[0]),
                         wine.split('_')[1]])
        ax.annotate('  ' + wine, (-x_embedded[i, 0], -x_embedded[i, 1]),
                    fontsize=6, color=col)

    plt.xlabel(f'{algo} dimension 1 [a.u.]')
    plt.ylabel(f'{algo} dimension 2 [a.u.]')
    plt.title('Varietals')
    plt.tight_layout()


def plot_tile_main(rs=28):
    '''
    Figure 1

    Show umap/tSNE dimensionality reduction of wines in 2d
    colored by estate.

    Also load in Bordeaux map


    rs: random_seed for dim reduction
    '''

    plt.ioff()
    fig = plt.figure(rs, figsize=(8, 10))
    ax = plt.subplot(3, 2, 1)
    img = mpimg.imread(pth_dat / 'data/bordeaux_map.png')
    plt.imshow(img)
    plt.axis('off')
    add_panel_letter(1)

    ax = plt.subplot(3, 2, 3)
    dim_red('tSNE', 'concat', ax, rs=rs)
    add_panel_letter(2)

    ax = plt.subplot(3, 2, 4)
    dim_red('umap', 'concat', ax, rs=rs)
    add_panel_letter(3)

    ax = plt.subplot(3, 2, 5)
    dim_red_grape_ratio('tSNE', ax, rs=rs)
    add_panel_letter(4)

    ax = plt.subplot(3, 2, 6)
    dim_red_grape_ratio('umap', ax, rs=rs)
    add_panel_letter(5)

    plt.tight_layout()
    plt.show()


def plot_tile_supp():
    '''
    Figure S2

    Show 2d-embedded spectra for all chemical types.
    For 3d embedding, set d3=True in dim_red.
    '''

    fig = plt.figure(figsize=(10, 10))
    k = 1
    for algo in ['umap', 'tSNE', 'PCA']:
        for chem_type in ['concat', 'esters', 'oak', 'offFla']:
            ax = plt.subplot(3, 4, k)
            dim_red(algo, chem_type, ax=ax, rs=8)
            k += 1

    plt.show()


def plot_tile_chem32():
    '''
    Figure 5a,b

    dim reduction for manual 32 (m_concat)
    '''
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    k = 0
    for algo in ['tSNE', 'umap']:
        dim_red(algo, 'm_concat', ax=axs[k], remap=True)
        axs[k].set_title('')
        add_panel_letter(k + 1, ax=axs[k])
        k += 1

    fig.tight_layout()


def varietals_table():
    '''
    Table S2

    For each wine, list estate code, vintage and varietals
    '''

    d = np.load(pth_dat / 'data/varietals.npy',
                allow_pickle=True).flat[0]

    x = []
    wines = []
    for wine in d:
        x.append(d[wine])
        wines.append(wine[0] + '_' + wine[1:])

    cols = ['estate', 'vintage', 'CS [%]', 'M [%]', 'CF [%]', 'PV [%]']
    r = []
    for i in range(len(wines)):
        r.append([wines[i].split('_')[0], wines[i].split('_')[1],
                  x[i][0], x[i][1], x[i][2], x[i][3]])

    df = pd.DataFrame(r, columns=cols)
    dfi.export(df, str(pth_dat / 'figs/table_S2.png'))
    
    
    
