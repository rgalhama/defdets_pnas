# coding:utf-8
import sys, os, inspect, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)

LAST_SESSION=12 #LDP

def load_data(path_to_data, isOmission, verbose=False):
    """
    Loads dataframes of model predictions and concatenates all the sessions.
    Slices data to return only columns of entropy and session.
    Adds a column "isOmission" with the constant value provided in parameter of same name.
    :param path_to_data: path where the csv files are
    :param isOmission:  column
    :return:
    """
    for session in range(1,LAST_SESSION+1):
        if verbose:
            print("Reading data of session ", session)
        df_s=pd.read_csv(os.path.join(path_to_data,
                    "BERT-LDP-incrsession-{}_children_testsession{}.csv".format(session, session)),
                         sep=';', error_bad_lines=False, engine="python")
        if session == 1:
            df=df_s[["session", "entropy"]]
        else:
            df_s=df_s[["session", "entropy"]]
            df=pd.concat([df, df_s])
            del df_s
    df["isOmission"]=isOmission
    return df

def plot_omissions(df, ofile):

    # Create graph
    df["Age"]=df.session.map(lambda x:10+(x*4))
    df=df.rename(columns={'entropy':'Entropy'})
    plt.figure(figsize=(12, 5))  # Adjust the width and height as needed
    ax=sns.boxplot(data=df, x=df.Age, y=df.Entropy, hue=df.isOmission,
                   hue_order=["determiner-omitted", "determiner-produced"], palette="Set2", width=0.6)
    #add vertical line where onset is
    plt.axvline(4, color='grey', ls="--")
    #Get rid of legend title
    h, l = ax.get_legend_handles_labels()
    plt.legend(h[:], l[:], ncol=3, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1., labelspacing=0.0,
               handletextpad=0.35, handlelength=1.5
               )
    sns.despine()

    #Compute number of observations
    nobs=df.groupby(['Age','isOmission'])['Entropy'].count()
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n=" + i for i in nobs]

    # Add them to the plot
    pos = range(len(nobs))
    en=0
    while en < len(nobs):
        position=en-(0.5*(en))
        text=ax.text(position-0.2,
                     12 if en%2==0 else -0.25,
                     str(nobs[en]), #en was tick
                     horizontalalignment='center',
                     size='x-small',
                     color=sns.color_palette("Set2")[0] if en%2==0 else sns.color_palette("Set2")[1],
                     weight='semibold')

        en+=1


    plt.tight_layout()
    plt.savefig(ofile, dpi=300)


def analysis_entropy_omissions(path_det_preds, path_omission_preds):

    # Load dataset of model predictions for determiner slots
    dfd=load_data(path_det_preds, "determiner-produced")

    # Load dataset of model predictions for omitted-determiner slots
    dfo=load_data(path_omission_preds, "determiner-omitted")

    #Concatenate
    df=pd.concat([dfd,dfo])
    del dfd,dfo

    #Drop nans
    df=df.dropna()

    return df

if __name__ == "__main__":
    path_det_preds="../../../results/results_masked_lm_id/a_the/LDP/incremental_training_top1/test_predictions/"
    path_omission_preds="../../../results/results_masked_lm_id/a_the/LDP/incremental_training_top1/test_omissions/"
    df=analysis_entropy_omissions(path_det_preds, path_omission_preds)
    plot_omissions(df, 'pnas_fig3.pdf')