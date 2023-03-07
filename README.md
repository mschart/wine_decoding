# Recovering origin and ratings of Bordeaux wines with raw gas chromatography

All figures can be reproduced using the following three scripts:

* `estate_vintage_clf.py`: Decoding estate and vintage from wine chromatograms
* `parker_clf.py`: Decoding Parker's rating from wine chromatograms
* `embed.py`: Embedding wine chromatograms in 2d (3d) space using umap and tSNE

Each script contains functions to process **raw data** that is found in `Source_Data/data` 
resulting in files for plotting, stored in `Source_Data/res`, precomputed for your convenience. 
Running all analyses from the raw data takes about 3 h on a laptop in total. All figures 
are precomputed in `Source_Data/figs`.

The `Source_Data/data` contains the following:

### Gas chromatography features per wine, three types and concatenation
* `oak.npy`, 80 wines x 30275 GC features, each wine e.g. with name `A_2012`, i.e. estate A, vintage 2012
* `esters.npy`, 80x7881
* `offFla.npy`, 80x10480
* `concat.npy`, 80x48636

### Manually selected compounds of gas chromatography per wine, three types and concatenation
* `m_oak.npy`, 80x13
* `m_esters.npy`, 80x16
* `m_offFla.npy`, 80x3
* `m_concat.npy`, 80x32

### Additional wine info
* `32_compound_names.npy`, chemical names for 32 compounds
* `parker_ratings.npy`, 80x1, Parker ratings (int between 0 and 100)
* `varietals.npy`, 80x4, percentage of 4 grape types per wine 
* `bordeaux_map.png`, map of Bordeaux area


## Prerequisites

In your python environment (for example installed via anaconda, latest version) install required libraries:

`pip install --requirement requirements.txt`

## List of figures with function name and script name to reproduce:

* Figure 1 `plot_tile_main()` in `embed.py`
* Figure 2, 5(c,d) `plot_violin()` in `estate_vintage_clf.py`
* Figure 3, S9 `plot_chunks2()` and `plot_all_survival()` in `estate_vintage_clf.py`
* Figure 4, Table S1, Figure 5e,f `plot_violin_p()` in `parker_clf.py`
* Figure 5a,b `plot_tile_chem32()` in `embed.py`
* Figure S2 `plot_tile_supp()` in `embed.py`
* Figure S3 `plot_best_concat()` in `estate_vintage_clf.py`
* Figure S4 `plot_vintage_decoding_per_wine()` in `estate_vintage_clf.py`
* Figure S5 `PCA_features()` in `estate_vintage_clf.py`
* Figure S6 `plot_weights()` in `estate_vintage_clf.py`
* Figure S7, S8, S11 `plot_all_survival()` in `estate_vintage_clf.py`
* Figure S12 `plot_weights_m_abs()` in `estate_vintage_clf.py`
* Figure S10 `plot_chunk2()` in `parker_clf.py`
* Figure S13 `plot_chunks_m_dists()` in `estate_vintage_clf.py`
* Table S2 `varietals_table()` in `embed.py`

## Example application:
To clone this repository into your home directory, type in a terminal:

`git clone https://github.com/mschart/wine_decoding.git`

To reproduce figure 1, navigate into directory `wine_decoding`,
start an ipython session and type:

```python
# read in script (pip install missing libraries if needed) for figure 1
run '/home/mic/wine_decoding/embed.py'

# activate interactive plotting
plt.ion()

# plot figure 1 by running listed function (loading in raw data directly)
plot_tile_main()
```

Some figures need manual assembly (figure 5, 3, S2). 














