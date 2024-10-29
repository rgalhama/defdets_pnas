# coding:utf-8

import os, sys, inspect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
SCRIPT_FOLDER = os.path.realpath(os.path.abspath(
    os.path.split(inspect.getfile(inspect.currentframe()))[0]))
PROJECT_FOLDER = os.path.join(SCRIPT_FOLDER, os.pardir, os.pardir, os.pardir)
sys.path.insert(0, PROJECT_FOLDER)
sns.set(font_scale=1.2)

def gather_a_the_nouns_parents(df, d, filter_dets=[]):
    """
        Gather all the nouns produced with each determinant in a dictionary d.
        Accumulate them with those of prior sessions.
        If list of filter dets is empty, assume all dets are ok.
    """
    for det_noun in df['parent_det_strs_indef_def_LEMMA']:
        try:
            det, noun = det_noun.strip().split(' ')
        except ValueError:
            pass
        if len(filter_dets) == 0 or det in filter_dets:
            d[det].add(noun)

    return d

def gather_a_the_nouns_test(df):

    dm={'a':set(), 'the':set()}
    dch={'a':set(), 'the':set()}
    #Gather all the nouns produced with each determinant and add them to those of prior sessions
    for det in ["a", "the"]:
        #Model
        nouns_model=df.query("preferred_det == @det")["h_noun"]
        dm[det].update(set(nouns_model))
        #Children
        nouns_model=df.query("h_det == @det")["h_noun"]
        dch[det].update(set(nouns_model))

    return dm, dch

def gather_all_new_combinations(idf, tdf, filter_dets=[]):

    #Accumulators
    dp={'a':set(), 'the':set()}
    dch={'a':set(), 'the':set()}
    dm={'a':set(), 'the':set()}
    new_combinations=[]
    nouns_all_new_combs=set() #keep track of all new nouns, for later use

    #For each session, look into parental input up to that session, and keep track of
    #words that have appeared only with one determinant
    for session in range(1,LAST_SESSION+1):
        #Filter by session
        idf_session=idf.query("session == @session")
        tdf_session=tdf.query("session == @session")
        #Gather noun+det combinations by parents in current session and accumulate with prior sessions
        dp=gather_a_the_nouns_parents(idf_session, dp, filter_dets=filter_dets)
        #Gather noun+det combinations by children/model in current session
        dm_session, dch_session=gather_a_the_nouns_test(tdf_session)
        #Find which combinations are new (current session - input - previous sessions)
        for det in ['a', 'the']:
            new_by_model=dm_session[det] - dp[det] - dm[det]
            new_by_children=dch_session[det] - dp[det] - dch[det]
            new_intersection = new_by_model.intersection(new_by_children)

            #Keep track of number of new combinations
            new_combinations.append({'session':session, 'speaker':'Model', 'det':det,
                                     'nouns_new_combs':str(new_by_model), 'new_detnoun_comb':len(new_by_model), })
            new_combinations.append({'session':session, 'speaker':'Children', 'det':det,
                                     'nouns_new_combs':str(new_by_children),'new_detnoun_comb':len(new_by_children)})
            #Keep track of how many nouns in new combinations overlap between children and model (as 'speaker')
            new_combinations.append({'session':session, 'det':det, 'speaker':'Intersection',
                                     'nouns_new_combs':str(new_intersection),'new_detnoun_comb':len(new_intersection)})

            #Keep track of all the nouns used in novel combinations, both in model and children
            for noun in new_by_model.union(new_by_children):
                nouns_all_new_combs.add(noun)

        #Accumulate for next round
        for det in ['a', 'the']:
            dm[det].update(dm_session[det])
            dch[det].update(dch_session[det])


    return nouns_all_new_combs, pd.DataFrame.from_records(new_combinations)

def gather_all_new_individual_combinations(idf, tdf, filter_dets=[]):
    """
    Finds new combinations of determiner and noun for each subject
    """

    #vars to accumulate data of sessions
    dp={'a':set(), 'the':set()}
    dch={'a':set(), 'the':set()}
    dm={'a':set(), 'the':set()}
    new_combinations=[]
    nouns_all_new_combs=set() #keep track of all new nouns, for later use

    for subject in tdf["subject"].unique():

        #Accumulate input data per individual subject
        dp_sbj_acc={'a':set(), 'the':set()}

        #Filter data by subject
        idf_sbj=idf.query("subject == @subject")
        tdf_sbj=tdf.query("subject == @subject")

        #For each session, look into parental input up to that session, and keep track of
        #words that have appeared only with one determinant
        for session in range(1,LAST_SESSION+1):
            #Filter by session, this time using
            idf_session=idf.query("session == @session")
            idf_sbj_session=idf_sbj.query("session == @session")
            tdf_sbj_session=tdf_sbj.query("session == @session")

            #Gather noun+det combinations by *all* parents in current session and accumulate with prior sessions
            #(model is trained on all parents)
            dp=gather_a_the_nouns_parents(idf_session,dp, filter_dets=filter_dets)
            #Gather noun+det combinations for parent of this subject and accumulate with prior sessions
            #(only for this subject)
            dp_sbj_acc=gather_a_the_nouns_parents(idf_sbj_session, dp_sbj_acc, filter_dets=filter_dets)
            #Gather noun+det combinations by children/model
            dm_session, dch_session=gather_a_the_nouns_test(tdf_sbj_session)

            #Find which combinations are new (current session - input - previous sessions)
            for det in ['a', 'the']:
                new_by_model=dm_session[det] - dp[det] - dm[det]
                new_by_children=dch_session[det] - dp_sbj_acc[det] - dch[det]
                new_intersection = new_by_model.intersection(new_by_children)
                #Store number of new combinations
                new_combinations.append({'subject': subject, 'session':session, 'speaker':'Model', 'det':det,
                                         'nouns_new_combs':str(new_by_model),   'new_detnoun_comb':len(new_by_model),})
                new_combinations.append({'subject': subject, 'session':session, 'speaker':'Children', 'det':det,
                                         'nouns_new_combs':str(new_by_children),'new_detnoun_comb':len(new_by_children)})
                #Store also how many nouns overlap between children and model (as 'speaker')
                new_combinations.append({'subject': subject, 'session':session, 'det':det, 'speaker':'Intersection',
                                         'nouns_new_combs':str(new_intersection),'new_detnoun_comb':len(new_intersection)})

                for noun in new_by_model.union(new_by_children):
                    nouns_all_new_combs.add(noun)

            #Accumulate
            for det in ['a', 'the']:
                dm[det].update(dm_session[det])
                dch[det].update(dch_session[det])

    return nouns_all_new_combs, pd.DataFrame.from_records(new_combinations)

def get_first_session_new_combination(df):
    """
        Find first session of true productivity (i.e. of producing a new combination of det and noun)
    """
    df=df[df['new_detnoun_comb']>0]
    first_session = df.groupby(['subject', 'speaker'], as_index=False).agg({'session': 'min'})
    first_session=first_session.rename(columns={'session':'first_session_new'})
    return first_session

def plot_new_combs_session(df, fname):
    """
        PNAS Fig 4
    """

    #Subset (remove intersection if computed)
    df = df[df.speaker != "Intersection"]

    #Change from session to age
    df['age'] = df['session'].map(lambda x: 10 + (x * 4))

    #Count
    df = df.groupby(['age', 'session', 'speaker']).sum()
    df=df.reset_index()
    df = df.sort_values(by='speaker')

    #Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    g=sns.catplot(data=df, x='age', y='new_detnoun_comb', hue='speaker', #row="subject",
                   kind='point', markers=['o', 'D', 'v'], linestyles=['-', '-', '--'], s=4.5,
                   palette=['orange' , '#00A1C9',  'silver'], legend_out=False)
    g.set_axis_labels("Age (months old)", "New DET-NOUN combinations", fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

def plot_corr_gen(df_orig, fname):
    """
        PNAS Fig 5
    """
    fig,ax=plt.subplots(1,2, sharey=True, sharex=True, figsize=(10,6))
    #for this plot, we are only interested in correlating same speaker (children with children and model with model)
    for i,(onset_speaker, first_new_speaker) in enumerate([("children","children"), ("model","model")]):
        df=df_orig
        df['age_onset_%s'%(onset_speaker)] = df['onset_%s'%(onset_speaker)].map(lambda x: 10 + (x * 4))
        df['age_first_new'] = df['first_session_new'].map(lambda x: 10 + (x * 4))
        df=df[df["speaker"] == first_new_speaker.capitalize()]
        rho=df["age_first_new"].corr(df["age_onset_%s"%(onset_speaker)])
        print(onset_speaker," r=",rho)
        sns.set_style("white")
        #add data to plot
        g=sns.regplot(data=df, x='age_onset_%s'%(onset_speaker),y='age_first_new',
                      ax=ax[i], scatter_kws={'alpha':1.0},
                      marker=['o','D'][onset_speaker=="model"],
                      color=['orange','#00A1C9'][onset_speaker=="model"],
                      x_jitter=0.5, y_jitter=0.5
                      ).set(title="{}\nr={:.2f}".format(onset_speaker.capitalize(),rho))
        #aesthetics
        ax[i].set_xticks(range(14,60,4))
        plt.xlim((12, 60))
        plt.ylim((12, 60))
        ax[i].set_xlabel("Onset (months old)")
        ax[i].set_ylabel("First new combination (months old)")

    plt.savefig(fname, dpi=300)

def main(input_ft, test_ft, path_results, filter_dets):

    # Load parent utterances
    idf=pd.read_csv(input_ft, sep=",")
    # Load data of model and children
    tdf=pd.read_csv(test_ft, sep=';',
                    engine="python",
                    #error_bad_lines=False, #older version
                    on_bad_lines='skip')

    # Find new combinations of determiner and noun
    nouns_new_combs_list, newcombs_df=gather_all_new_combinations(idf, tdf, filter_dets)
    newcombs_df.to_csv("new_combinations_detnoun.csv", index=False)
    
    # Figure 4 PNAS
    plot_new_combs_session(newcombs_df, 'pnas_fig4.pdf')

    # Find the new combinations individually
    nouns_new_combs_list, ind_newcombs_df=gather_all_new_individual_combinations(idf, tdf, filter_dets)

    # Compute first session new combination (onset of "true" productivity)
    df_first=get_first_session_new_combination(ind_newcombs_df)
    #Load onset data (productivity as computed in Study 1) and merge
    df_onset=pd.read_csv('onset.csv')
    df=df_first.merge(df_onset, on=["subject"], how='right')

    # Figure 5 PNAS
    plot_corr_gen(df, 'pnas_fig5.pdf')



if __name__ == "__main__":

    LAST_SESSION=12
    parent_data=os.path.join(PROJECT_FOLDER, 'data/processed/LDP/parents_detnoun_session.csv')
    prediction_data_folder=os.path.join(PROJECT_FOLDER, 'results/results_masked_lm_id/a_the/LDP/wl_incremental_training_top1/test_predictions/')
    model_data=os.path.join(prediction_data_folder, 'all_predictions.csv')
    main(parent_data, model_data, prediction_data_folder, filter_dets=['a', 'the'])
