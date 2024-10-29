# coding:utf-8

import os, inspect, sys
import pandas as pd
from numpy import median, mean
import seaborn as sns
import matplotlib.pyplot as plt
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)
from src.metrics.onset import count_dets_per_noun, count_nouns_w_2_dets
from src.analyses.PNAS.aesthethics_plots import rc,color_dict,label_dict,marker_dict

#Columns
COL_productivenouns_model= 'model_n_athe_productive_noun_types'
COL_productivenouns_ch= 'child_n_athe_productive_noun_types'
COL_productivenouns_parents='parent_n_athe_productive_noun_types'
prodcols = {"*CHI:": COL_productivenouns_ch,
           "*MOT:": COL_productivenouns_parents,
           "*MOD": COL_productivenouns_model
           }

def plot_nouns_w_2dets_grouped_by_onset(df, onset_df, ofile, title, addmodel=True):
    """
        PNAS Figure 1.
    """

    # Change session to age
    df = df.merge(onset_df, on='subject', how='left')
    df['age'] = df['session'].map(lambda x: 10 + (x * 4))
    df['onset_children'] = df['onset_children'].map(lambda x: 10 + (x * 4))

    colwrap = 4
    g = sns.FacetGrid(df,
                      col='onset_children',
                      # hue='speaker',
                      sharex=False,
                      sharey=True,
                      col_wrap=colwrap,
                      # height=2.25,
                      margin_titles=True,
                      aspect=1,
                      xlim=(1, 12),
                      )

    # Add sample size for each group
    dfg = onset_df.groupby('onset_children')
    onset_size = [len(v) for k, v in dfg]
    onset_ages = [int(10+k*4) for k, _ in dfg]


    # Add generalization (nouns w 2 dets) for children
    # Note that order matters! if barplot is created before scatterplot, x axes won't be complete
    g = g.map(sns.pointplot, 'session', COL_productivenouns_ch,
              estimator=median, order=list(range(12 + 1)),  # ord,
              color='orange', capsize=.3, errwidth=.95, join=False,
              markersize=0.1,
              label='Children',
              zorder=3, edgecolor="black")

    if addmodel:
        g = g.map(sns.lineplot, 'session', COL_productivenouns_model, lw=1.5, alpha=0.7, label='Model', estimator=median,
                  zorder=1)  # , order=ord)

    # Create index to change session to age in x axes
    ord = list(map(lambda x: 10 + (x * 4), range(0, 12 + 1)))
    for i, ax in enumerate(g.axes):
        # Add subgraph title (onset grouping info)
        g.axes[i].set_title("onset: {} months (n={})".format(onset_ages[i], onset_size[i]),fontsize=11)
        # Add horizontal line at 2
        ax.axhline(y=2, color='black', ls="--", lw=1, alpha=0.7, zorder=2)
        # other tweaks
        ax.set_ylabel("Noun types with a/the", fontsize=13)
        ax.set_ylim(-0.5, 15)
        if i > ((8 / colwrap) + 1):
            ax.set_xlabel("Age (months old)", fontsize=13)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        ax.set_xticklabels(ord)

    if ofile is not None and ofile != '':
        plt.savefig(ofile, dpi=300)

    return g


def plot_nouns_w_2dets_accross_time(df, onset_df, ofile, title, addmodel=True):
    """
        PNAS Figure 2A
    """

    # Change session to age
    df['age'] = df['session'].map(lambda x: 10 + (x * 4))

    # Change style
    sns.set(font_scale=1.2, rc=rc)
    markersize = 4

    # Create graph
    fig, ax = plt.subplots(figsize=(6, 5))

    # Data children
    g = sns.pointplot(data=df, x='session', y=COL_productivenouns_ch,
                      estimator=median,
                      order=list(range(12 + 1)),
                      color='orange', capsize=.3, errwidth=.95, join=False,
                      markersize=markersize,
                      markers=marker_dict['*CHI:'],
                      label='Children',
                      zorder=3, edgecolor="black")

    # Data parents
    sns.pointplot(data=df, x='session', y=COL_productivenouns_parents,
                  estimator=median,
                  order=list(range(12 + 1)),
                  markers=marker_dict['*MOT:'],
                  color=color_dict['*MOT:'],
                  capsize=.3, errwidth=.95, join=False,
                  markersize=markersize,
                  label='Caregivers',
                  zorder=3, edgecolor="black")

    # Set transparency of all the markers
    plt.setp(g.collections, alpha=.6)

    # Data model
    sns.lineplot( data=df, x='session', y=COL_productivenouns_model, estimator=median,
                  lw=2, alpha=0.4,  zorder=1)#, label='Model')  # , order=ord)


    # Aesthetics
    # session to age in LDP
    ord = list(map(lambda x: 10 + (x * 4), range(0, 12 + 1)))
    # add horizontal line at 2
    ax.axhline(y=2, color='black', ls="--", lw=1, alpha=0.7, zorder=2)
    ax.set_ylabel("Noun types with a/the", fontsize=16)
    ax.set_xlim(0.5, 12.5)
    ax.set_xlabel("Age (months old)", fontsize=16)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels(ord)
    plt.tight_layout()

    plt.savefig(ofile, dpi=300)


def plot_scatter_onset(onset_df, ofile, TRAINING, TOPK, TYPEDETS ):
    """
        PNAS Fig 2B
        (working name: onset_{}_{}_{}_{}.png)
    """
    sns.set(font_scale=1.2,
            rc=rc)

    # turn session into age
    onset_df['age_children'] = onset_df['onset_children'].map(lambda x: 10 + (x * 4))
    onset_df['age_model'] = onset_df['onset_model'].map(lambda x: 10 + (x * 4))
    rho = onset_df["age_children"].corr(onset_df["age_model"])
    print("r=", rho)
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlim((0, 60))
    plt.ylim((0, 60))
    plt.grid()
    g = sns.regplot(data=onset_df, x='age_children', y='age_model',
                    x_jitter=0.5, y_jitter=0.5, color='grey')
    plt.xlabel("Onset Age Children")
    plt.ylabel("Onset Age Model")
    plt.tight_layout()
    plt.savefig(ofile, dpi=300)
    return g

def get_onset_df(df):
    """ Returns dataframe with onset session (i.e. first session that meets onset criterion)
        for humans and model. """

    human1stsess = (df.query("child_n_athe_productive_noun_types > 1")
                    .groupby("subject", as_index=False)["session"].min())
    model1stsess = (df.query("model_n_athe_productive_noun_types > 1")
                    .groupby("subject", as_index=False)["session"].min())
    onset_df = model1stsess.merge(human1stsess, how='left', on=["subject"], suffixes=["_model", "_children"])
    onset_df=onset_df.rename(columns={'session_model': 'onset_model', 'session_children': 'onset_children'})
    return onset_df

def compute_counts_gen(df):
    df = count_dets_per_noun(df)
    dfgen = count_nouns_w_2_dets(df, COL_productivenouns_model)
    return dfgen

def load_behavioral(path_behavioral, productivity_col):
    print("Loading children's data at:\n...", path_behavioral)
    chdf = pd.read_csv(path_behavioral)
    # Drop children without available observations (that's subject 110 in LDP):
    chdf = chdf.dropna(subset=[productivity_col])
    chdf[productivity_col] = chdf[productivity_col].astype(int)
    print("Loaded.")
    return chdf

def prepare_data():

    #Load behavioral data (includes generalization counts)
    chdf = load_behavioral(path_behavioral, 'child_n_athe_productive_noun_types')

    #Load model predictions (test input (masked children's sentences) and predictions)
    df=pd.read_csv(os.path.join(path_model_data, 'test_predictions/all_predictions.csv'),
                   sep=';', engine='python', error_bad_lines=False) # skip line 5791

    #Compute counts generalization (number of nouns with 2 dets) on model data
    df = compute_counts_gen(df)

    #Merge generalization of model (n nouns with 2 dets) with behavioral data
    df = chdf.merge(df, on=['subject', 'session'], how='left')

    #Compute first session of generalization for model and children
    onset_df=get_onset_df(df)


    return df, onset_df

def main():
    df, onset_df = prepare_data()
    # Figure 1
    plot_nouns_w_2dets_grouped_by_onset(df, onset_df, ofile='pnas_fig1.pdf', title='', addmodel=True)
    # Figure 2A
    plot_nouns_w_2dets_accross_time(df, onset_df, 'pnas_fig2A.pdf', '')
    # Figure 2B
    plot_scatter_onset(onset_df, 'pnas_fig2B.pdf', TRAINING, TOPK, TYPEDETS)
    # Export onset session data
    onset_df.to_csv(os.path.join(PROJECT_FOLDER, 'src/analyses/PNAS/onset.csv'))

if __name__ == "__main__":
    TRAINING='incremental'
    TOPK=1
    TYPEDETS='athe'
    path_model_data= os.path.join(PROJECT_FOLDER, 'results/results_masked_lm_id/a_the/LDP/incremental_training_top1/')
    path_behavioral = os.path.join(PROJECT_FOLDER, 'data/parent_child_determiner_noun_summary_data.csv')
    main()
