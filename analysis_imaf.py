# -*- coding: utf-8 -*-
"""
May 29th 2024 - revised May 26th 2025
Claire Vanbuckhave

Analyse data from the pupillometry experiment based on Kay et al. (2022)
    * IMAGERY FIRST VERSION
    
Initial data collection and experiment programmation by Leo Pasturel (Msc) 
between March and April 2022 at the Laboratoire de Psychologie et NeuroCognition (Grenoble).
"""
# =============================================================================
#%% Import libraries 
# =============================================================================
# system
import os
import inspect
import warnings

# visualisation
from matplotlib import pyplot as plt
import seaborn as sns

# operations
import datamatrix
from datamatrix import plot, convert, NAN, functional as fnc, series as srs, operations as ops
from datamatrix.colors.tango import blue, red
from eyelinkparser import parse, defaulttraceprocessor
import numpy as np
from collections import Counter
from statsmodels.formula.api import mixedlm
from scipy.stats import spearmanr, wilcoxon
import pingouin as pg
import pandas as pd
from scipy.stats.distributions import chi2
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_white
# =============================================================================
#%% Prepare useful parameters
# =============================================================================
# Set up visualisation params (aesthetics)
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 13
plt.rcParams['xtick.major.size'] = 11
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2

# Define preprocessing parameters 
smooth_winlen = 51 # most commonly used
downsampling_factor = 10 # get a 100 Hz sample frequency 
factor = 100 # how many samples to have 1 second? (after downsampling)
    
# Define values for depth of each cell within columns.
baseline_length = int(1600/downsampling_factor)
imagery_length = int(6000/downsampling_factor)
rest_1_length = int(4000/downsampling_factor)
stim_length = int(5000/downsampling_factor)
rest_2_length = int(10000/downsampling_factor)

pupil_length = baseline_length + imagery_length + rest_1_length + stim_length + rest_2_length

# Define plot size
FIGSIZE = 10, 5
X = np.linspace(0, pupil_length/factor, pupil_length)

# =============================================================================
#%% Set working directory
# =============================================================================
############### start part to modify
# Get full actual file path
actual_file_path = inspect.getframeinfo(inspect.currentframe()).filename
# Get parent folder path
path = os.path.dirname(os.path.abspath(actual_file_path)) # auto
path = "C:/Users/cvanb/Desktop/expe_repli" # manual
folder_edf = 'data_imaf' # where the .edf datafiles are
quest_path = os.path.join(path, 'results-survey.csv') # the questionnaire data
full_path = os.path.join(path, folder_edf)
############### end part to modify

# =============================================================================
#%% Define useful functions
# =============================================================================
# Create function that will create a simpler datamatrix with values of interest.
def get_data(full_path):
    """Load data into datamatrix."""

    # Get the datafolder that contains all the .edf files
    dm = parse(
        maxtracelen=10000, # max length of a phase
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,
            downsample=None,
            mode='advanced'),
        folder=full_path,
        pupil_size=True,
        gaze_pos=False,    # Don't store gaze-position information to save memory
        time_trace=False,  # Don't store absolute timestamps to save memory
        multiprocess=16)
    
    print(dm.exp_name.unique)
    print(f"Only the replication version: {dm.exp_name.unique[0] != 'pupillo_repli'}")
    
    # Apply depth to each column
    dm.ptrace_baseline.depth = 1600
    dm.ptrace_imagery.depth = 6000
    dm.ptrace_rest_1.depth = 4000
    dm.ptrace_stim.depth = 5000
    dm.ptrace_rest_2.depth = 10000
        
    # Concatenate pupil size of the whole trial for all trials
    dm.pupil = srs.concatenate(
        dm.ptrace_baseline,
        dm.ptrace_imagery,
        dm.ptrace_rest_1,
        dm.ptrace_stim,
        dm.ptrace_rest_2)
    
    # Define stimuli colors
    dm.stim_color = fnc.map_(
        lambda s: 'bright' if '[255.0,255.0,255.0]' in s else 'dark',
        dm.stim_color)
    
    # Remove training trials
    print(Counter(dm.stim_orientation))
    dm = dm.trialid >= 3 
    print(Counter(dm.stim_orientation))

    print(dm.column_names)
    
    return dm

def preprocess_df(df):
    """Preprocess quest data to create columnns etc."""
    # Retrieve the QMI columns 
    QMI = [i for i in df.columns if i.startswith('QMIQ')]
    QMI_visual = [i for i in df.columns if i.startswith('QMIQ02')]
    QMI_audio = [i for i in df.columns if i.startswith('QMIQ03')]
    QMI_touch = [i for i in df.columns if i.startswith('QMIQ04')]
    QMI_motor = [i for i in df.columns if i.startswith('QMIQ05')]
    QMI_taste = [i for i in df.columns if i.startswith('QMIQ06')]
    QMI_smell = [i for i in df.columns if i.startswith('QMIQ07')]
    QMI_feeling = [i for i in df.columns if i.startswith('QMIQ08')]

    # Compute the mean QMI scores
    df['QMI_total'] = df[QMI].mean(axis=1)
    df['QMI_visual'] = df[QMI_visual].mean(axis=1)
    df['QMI_audio'] = df[QMI_audio].mean(axis=1)
    df['QMI_touch'] = df[QMI_touch].mean(axis=1)
    df['QMI_motor'] = df[QMI_motor].mean(axis=1)
    df['QMI_taste'] = df[QMI_taste].mean(axis=1)
    df['QMI_smell'] = df[QMI_smell].mean(axis=1)
    df['QMI_feeling'] = df[QMI_feeling].mean(axis=1)
    
    # Cronbach's alpha for QMI questionnaire
    alpha = pg.cronbach_alpha(data=df[QMI])
    print(f"Cronbach's alpha: {alpha}")

    # Descriptive stats for demographic items
    demo = [i for i in df.columns if i.startswith('Q0')]
    desc_stats_demo = np.round(df[demo].describe(include='all'), 2).transpose()
    df.groupby('Q03Sex').Q02Age.describe(include='all') # Age by sex
    desc_stats_demo.to_csv(path+'/descriptives_demo.csv', encoding='utf-8', index=True)

    # Descriptive stats for QMI and other items
    questlist = [i for i in df.columns if i.startswith('QMI_')]
    desc_stats_quest = np.round(df[questlist].describe(include='all'), 2).transpose() # get descriptives and round to the 2nd decimal
    desc_stats_quest.to_csv(path+'/descriptives_quest.csv', encoding='utf-8', index=True)

    # See scores of those who have a mean score under 4 for the visual subscale of the QMI
    for a in list(df.Q00PARTICIPANT[df.QMI_visual<=4]):
        print(a, df.QMI_visual[df.Q00PARTICIPANT==a])

    # Check if any aphant
    print(df.Q06Aphant.describe(include='all'))

    # Print their scores
    print('YES')
    for a in list(df.Q00PARTICIPANT[df.Q06Aphant=='Y']):
        print(a, df.QMI_visual[df.Q00PARTICIPANT==a])
    print('MAYBE')
    for a in list(df.Q00PARTICIPANT[df.Q06Aphant=='M']):
        print(a, df.QMI_visual[df.Q00PARTICIPANT==a])
    
    return df

def merge_dm_df(dm_to_merge, df):
    """Compute main variables and add them to the dm."""
    # Copy of dm
    dm = dm_to_merge.subject_id != ''
    
    # Adjust subject ID
    for s, sdm in ops.split(dm.subject_id):
        if s == 'KR_SO24':
            dm.subject_id[sdm] = 'S024'
        else:
            dm.subject_id[sdm] = s[3:]
    
    # Match participants between dm and df
    new, df['excluded'] = dm.subject_id.unique, 0
    for i in list(df.index):
        if df.Q00PARTICIPANT[i] in new: 
            df.loc[i, 'excluded'] = 0
        else: 
            df.loc[i, 'excluded'] = 1
    df = df[df.excluded == 0].reset_index()
    print(f'Datamatrix and dataframe match in terms of participants: {list(np.sort(df.Q00PARTICIPANT)) == list(np.sort(new))}')

    # Merge columns of the df to the dm
    dm.QMI_visual, dm.aphant = '', ''
    for p, sdm in ops.split(dm.subject_id):
        sub_df = df[df['Q00PARTICIPANT'] == p] # Subset the df too
        dm.QMI_visual[sdm] = float(sub_df.QMI_visual.iloc[0])
        dm.aphant[sdm] = str(sub_df.Q06Aphant.iloc[0])
    
    return dm

def preprocess(dm_to_process):
    """Apply smoothing, downsample and apply baseline correction."""
    # create a copy of the dm to not overwrite it
    dm = dm_to_process.subject_id != NAN
    
    # Smoothing and downsampling
    dm.pupil = srs.downsample(dm.pupil, by=downsampling_factor)
    #dm.pupil = srs.smooth(dm.pupil, winlen=smooth_winlen)
    
    # Exclude trials with unrealistic baseline-corrected mean pupil size during baseline (outliers) 
    print(f'Before trial exclusion (pupil size): {len(dm)} trials.')
    plt.figure(figsize=(13,8))
    dm.z_pupil = NAN
    for s, sdm in ops.split(dm.subject_id):
        dm.z_pupil[sdm] = ops.z(srs.reduce(sdm.pupil[:,110:baseline_length]))

    plt.title('Mean pupil size')
    sns.distplot(dm.z_pupil)
    plt.axvline(-2);plt.axvline(2)
    plt.xlabel('Mean pupil size (z-scored)')
    plt.tight_layout();plt.legend()
    plt.show()  
    
    unrealistic1 = list(np.array(dm.subject_id[dm.z_pupil > 2.0]))
    unrealistic2 = list(np.array(dm.subject_id[dm.z_pupil < -2.0]))
    nan_ = list(np.array(dm.subject_id[dm.z_pupil == NAN]))
    print(unrealistic1, unrealistic2, nan_)
    print(f'{len(unrealistic1)+len(unrealistic2) + len(nan_)} trials with outlier or NAN pupil sizes.')
    
    dm = dm.z_pupil != NAN 
    dm = dm.z_pupil <= 2.0 
    dm = dm.z_pupil >= -2.0 
    print(f'After trial exclusion (pupil size): {len(dm)} trials.')

    
    # Compute mean pupil size during baseline
    dm.mean_baseline = NAN
    for p, t, sdm in ops.split(dm.subject_id, dm.trialid):
        dm.mean_baseline[sdm] = srs.reduce(sdm.pupil[:, 110:baseline_length])
        
    # Remove the baseline from the pupil size
    dm.pupil = srs.baseline(
        dm.pupil,
        dm.pupil[:,0:baseline_length],
        int(baseline_length-(factor/2)), int(baseline_length), # 500 last ms before start of imagery
        method='subtractive')   
    
    # Compute mean pupil size during imagery
    dm.mean_pupil, dm.mean_perception, dm.mean_rest = NAN, NAN, NAN
    for p, t, sdm in ops.split(dm.subject_id, dm.trialid):
        dm.mean_pupil[sdm] = srs.reduce(sdm.pupil[:, baseline_length+100:baseline_length+imagery_length]) #start at end of first auditory cue
        dm.mean_perception[sdm] = srs.reduce(sdm.pupil[:, pupil_length-rest_2_length-stim_length:pupil_length-rest_2_length])
        dm.mean_rest[sdm] = srs.reduce(sdm.pupil[:, baseline_length+imagery_length:baseline_length+imagery_length+rest_1_length])

    # Compute mean pupil size differences between conditions during imagery
    dm.pupil_change, dm.percept_change, dm.rest_change = NAN, NAN, NAN
    dm.bl_change = NAN
    for p, sdm in ops.split(dm.subject_id):
        mean_dark, mean_bright = sdm.mean_pupil[sdm.stim_color=='dark'].mean, sdm.mean_pupil[sdm.stim_color=='bright'].mean
        dm.pupil_change[sdm] = mean_dark - mean_bright
        
        mean_dark, mean_bright = sdm.mean_perception[sdm.stim_color=='dark'].mean, sdm.mean_perception[sdm.stim_color=='bright'].mean
        dm.percept_change[sdm] = mean_dark - mean_bright
        
        mean_dark, mean_bright = sdm.mean_rest[sdm.stim_color=='dark'].mean, sdm.mean_rest[sdm.stim_color=='bright'].mean
        dm.rest_change[sdm] = mean_dark - mean_bright
    
    dm_grouped = ops.group(dm, by=[dm.subject_id, dm.stim_color]) # Group per participant and stim brightness 
    
    for col in dm_grouped.column_names:     # Make sure to have only unique mean values for each variable per participant 
        if type(dm_grouped[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_grouped[col] = srs.reduce(dm_grouped[col], operation=np.nanmean)
    
    # Compute mean pupil size for each condition
    dm_grouped.pupil = ops.SeriesColumn(shape=(dm.pupil.depth,))
    for p, c, sdm in ops.split(dm_grouped.subject_id, dm_grouped.stim_color):
        dm_sub = dm.subject_id == p # 
        dm_grouped.pupil[sdm] = dm_sub.pupil[dm_sub.stim_color==c].mean#.reshape((1, dm.pupil.depth))
    
    dm_grouped.rating = np.round(dm_grouped.rating) # to not have infinite meaningless values

    return dm, dm_grouped

def plot_dm(dm_to_plot, grouped=True, all_plots=True, title=None):
    """Plot the main plots: pupil traces for the whole trial, only during imagery and barplots."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create copy of dm to not overwrite it
    dm = dm_to_plot.subject_id != NAN
    
    # Pupil size changes relative to baseline for the whole trial
    plot.new(size=FIGSIZE)
    plt.title(title)
    plt.ylim(bottom=-2000, top=2000)
    plt.xlim(0, pupil_length/factor)
    plt.xticks(range(0,27,2), labels=range(0,27,2))
    plt.xticks(range(0,27,1), labels=None)
    plt.axvline(baseline_length/factor, color='black', linestyle='dotted') # Add vertical line for end of baseline/start of stim
    plt.axvline((baseline_length + imagery_length)/factor, color='black', linestyle='dotted') # Add vertical line for end of stim/start of rest
    plt.axvline((baseline_length + imagery_length + rest_1_length)/factor, color='black', linestyle='dotted') # Add vertical line for end of rest/start of imagery
    plt.axvline((baseline_length + imagery_length + rest_1_length + stim_length)/factor, color='black', linestyle='dotted') # Add vertical line for end of stim/start of rest
    plt.axhline(0, color='black', linestyle='dotted') # Add horizontal line for baseline level
    for stim_color, _dm in ops.split(dm.stim_color):
        plot.trace(
            _dm.pupil,
            x=X, 
            color=blue[1] if stim_color == 'dark' else red[1],
            label=f"{stim_color.title()} (N={len(_dm)})",
            err='se')
    # Add annotations
    plt.annotate('Fixation', rotation=90, xy=(100, 270), xycoords='figure points')
    plt.annotate('Imagery', xy=(130, 300), xycoords='figure points')
    plt.annotate('Rest', xy=(250, 300), xycoords='figure points')
    plt.annotate('Perception', xy=(330, 300), xycoords='figure points')
    plt.annotate('Rest', xy=(440, 300), xycoords='figure points')
    plt.ylabel('Pupil-size changes\nfrom baseline (a.u.)', fontsize=18)
    plt.xlabel('Time (s)', fontsize=18)
    plt.legend(loc='lower right', frameon=True, ncol=1)
    plt.show()
    
    if all_plots==True:
        # Show pupil size changes to baseline only for the imagery period
        plt.figure(figsize=(8, 8))
        for stim_color, _dm in ops.split(dm.stim_color): # Split conditions white vs black
            plot.trace(
                _dm.pupil[:, baseline_length:baseline_length+imagery_length],
                x=np.linspace(0, imagery_length/factor, imagery_length),
                color=red[1] if stim_color == 'bright' else blue[1],
                label=f'{stim_color.title()} (N={len(_dm)})',
                err='se')
        plt.xticks(np.arange(0, 7, 1), labels=np.arange(1.6, 8.6, 1), fontsize=18);plt.yticks(fontsize=18)
        plt.axhline(0, linestyle='dotted', color='black')
        plt.xlim([0, imagery_length/factor])
        plt.show()
        
        # Mean pupil size during imagery per brightness condition 
        dm_df = convert.to_pandas(dm) # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
        plt.figure(figsize=(18,5))
        plt.subplot(1,3,1)
        sns.barplot(x='stim_color', y='mean_pupil', hue=None, data=dm_df, palette=[blue[1], red[1]], order=['dark', 'bright'], 
                    errorbar=('se', 1), alpha=0.5,edgecolor="black", errcolor="black", errwidth=1.5, capsize = 0.1, linewidth=2)
        sns.stripplot(x='stim_color', y='mean_pupil', hue=None, data=dm_df, palette=[blue[1], red[1]], order=['dark', 'bright'], alpha=0.6, legend=False, size=8)
        plt.xlabel('Stimulus type', fontsize=18);plt.ylabel('Mean pupil size change from\nbaseline during imagery (a.u.)', fontsize=18)
        plt.xticks(range(0,2), ['Dark', 'Bright'], fontsize=18);plt.yticks(fontsize=15)
        plt.axhline(0, linestyle='solid', color='black') 
    
        # Individual pupil size differences during imagery (dark - bright) per vividness rating
        if grouped == False:
            dm_plot = ops.group(dm, by=[dm.subject_id, dm.rating])     # Group per participant 
        else:
            dm_plot = ops.group(dm, by=[dm.subject_id])     # Group per participant 
        for col in dm_plot.column_names:     # Make sure to have only unique mean values for each variable per participant 
            if type(dm_plot[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                dm_plot[col] = srs.reduce(dm_plot[col], operation=np.nanmean)
    
        dm_df = convert.to_pandas(dm_plot) # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
    
        plt.subplot(1,3,2)
        sns.barplot(x='rating', y='pupil_change', hue=None, data=dm_df, palette='Greens', 
                    errorbar=('se', 1), alpha=0.5,edgecolor="black", errcolor="black", errwidth=1.5, capsize = 0.1, linewidth=2)
        sns.stripplot(x='rating', y='pupil_change', hue=None, data=dm_df, color='green', alpha=0.3, legend=False, size=8, dodge=False)
        plt.xlabel('Trial-by-trial vividness ratings', fontsize=20);plt.ylabel('Pupil-size mean differences\nduring imagery (a.u.)', fontsize=18)
        plt.xticks(fontsize=18);plt.yticks(fontsize=18)
        plt.yticks(np.arange(-400, 400, 200), fontsize=18);plt.ylim([-400, 300])
        plt.axhline(0, linestyle='solid', color='black') 
        
        plt.subplot(1,3,3)
        #test_correlation(dm_grouped, y='pupil_change', x='rating', lab='Trial-by-trial ratings', alt='two-sided', color='green')
        test_correlation(dm_grouped, y='pupil_change', x='QMI_visual', lab=None, alt='two-sided', color='purple')
        plt.ylabel('Pupil-size mean differences\nduring imagery (a.u.)', fontsize=18);plt.xlabel('Visual imagery abilities\n(QMI visual)', fontsize=18)
        plt.yticks(np.arange(-400, 400, 200), fontsize=18);plt.xticks(np.arange(1, 8, 1), np.arange(1, 8, 1), fontsize=15)
        plt.xlim([2, 7.2]);plt.ylim([-400, 300])
        plt.tight_layout()
        plt.show()
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)  

def count_nonnan(a):
    return np.sum(~np.isnan(a))

def check_blinks(dm_blinks):
    """Check number of blinks per condition and per participant."""
    dm_blinks.n_blinks=''
    for p, t, sdm in ops.split(dm_blinks.subject_id, dm_blinks.stim_color):
        dm_blinks.n_blinks[sdm] = srs.reduce(sdm.blinkstlist_imagery, operation=count_nonnan) 
    
    # aggregate data by subject and condition
    pm = ops.group(dm_blinks, by=[dm_blinks.subject_id, dm_blinks.stim_color])
        
    # calculate mean blink rate per condition
    pm.mean_blink_rate = srs.reduce(pm.n_blinks)
    df_blinks = convert.to_pandas(dm_blinks)
    df_pm = convert.to_pandas(pm)

    # Plot the mean blink rate as a function of experimental condition and participant
    plt.figure(figsize=(20,8))
    x = sns.pointplot(
        x="stim_color",
        y="n_blinks",
        hue="subject_id",
        data=df_blinks,
        ci=None,
        palette=sns.color_palette(['indianred']),
        markers='.')
    plt.setp(x.lines, alpha=.4)
    plt.setp(x.collections, alpha=.4)
    sns.pointplot(
        x="stim_color",
        y="mean_blink_rate",
        data=df_pm,
        linestyles='solid',
        color='crimson',
        markers='o',
        scale=2)
    plt.xlabel('Condition', fontsize=30)
    plt.ylabel('Mean number of blinks', fontsize=30)
    plt.legend([], [], frameon=False)
    plt.xticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    
    # Check fixations 
    plt.figure(figsize=(15,8));plt.suptitle('Gaze / eye position', fontsize=35)
    plt.subplot(1,2,1);plt.title('Bright', fontsize=30)
    x = np.array(dm_blinks.fixxlist_imagery[dm_blinks.stim_color=='bright'])
    y = np.array(dm_blinks.fixylist_imagery[dm_blinks.stim_color=='bright'])
    x = x.flatten()
    y = y.flatten()
    plt.hexbin(x, y, gridsize=25)
    plt.axhline(500, color='white');plt.axvline(500, color='white')
    plt.yticks(fontsize=25);plt.xticks(fontsize=25);plt.xlabel('x', fontsize=25);plt.ylabel('y', fontsize=25)
    plt.subplot(1,2,2);plt.title('Dark', fontsize=30)
    x = np.array(dm_blinks.fixxlist_imagery[dm_blinks.stim_color=='dark'])
    y = np.array(dm_blinks.fixylist_imagery[dm_blinks.stim_color=='dark'])
    x = x.flatten()
    y = y.flatten()
    plt.hexbin(x, y, gridsize=25)
    plt.axhline(500, color='white');plt.axvline(500, color='white')
    plt.yticks(fontsize=25);plt.xticks(fontsize=25);plt.xlabel('x', fontsize=25);plt.ylabel('y', fontsize=25)
    plt.tight_layout()
    plt.show()

    return dm_blinks
# =============================================================================
#%% Preprocess data & descriptives
# =============================================================================
dm_ = get_data(full_path) # load the data into a datamatrix
df_ = pd.read_csv(quest_path, sep=',') # Get questionnaires data
df = preprocess_df(df_)

df['Q02Age'].describe()
df.groupby('Q03Sex')['Q02Age'].describe()
df['Q04Vision'].describe()
df['QMI_visual'].describe()
QMI = [i for i in df.columns if i.startswith('QMI_')]
df[QMI].describe()

dm_b = check_blinks(dm_)
print(f'Blinks: M = {dm_b.n_blinks.mean}, STD = {dm_b.n_blinks.std}, min = {dm_b.n_blinks.min}, max = {dm_b.n_blinks.max})')    

dm_processed, dm_grouped = preprocess(dm_b) # preprocess pupil data
print(f'Vividness: M = {dm_b.rating.mean}, STD = {dm_b.rating.std}, min = {dm_b.rating.min}, max = {dm_b.rating.max})')    

dm_processed = merge_dm_df(dm_processed, df)
dm_grouped = merge_dm_df(dm_grouped, df)

# Check positive score rate
# Perception
N = len(dm_grouped.subject_id.unique)
pos_effects = set(dm_grouped.subject_id[dm_grouped.percept_change > 0])
print(f'Positive effects: {np.round(len(pos_effects)/N * 100,2)}% ({len(pos_effects)}/{N})')

#Imagery
N = len(dm_grouped.subject_id.unique)
pos_effects = set(dm_grouped.subject_id[dm_grouped.pupil_change > 0])
print(f'Positive effects: {np.round(len(pos_effects)/N * 100,2)}% ({len(pos_effects)}/{N})')
print(f'Vividness (positive): {np.round(dm_grouped.rating[dm_grouped.pupil_change > 0].mean,2)} (STD = {np.round(dm_grouped.rating[dm_grouped.pupil_change > 0].std, 2)})')
print(f'QMI visual (positive): {np.round(dm_grouped.QMI_visual[dm_grouped.pupil_change > 0].mean,2)} (STD = {np.round(dm_grouped.QMI_visual[dm_grouped.pupil_change > 0].std, 2)})')

print(f'Vividness (neg or null): {np.round(dm_grouped.rating[dm_grouped.pupil_change <= 0].mean,2)} (STD = {np.round(dm_grouped.rating[dm_grouped.pupil_change <= 0].std, 2)})')
print(f'QMI visual (neg or null): {np.round(dm_grouped.QMI_visual[dm_grouped.pupil_change <= 0].mean,2)} (STD = {np.round(dm_grouped.QMI_visual[dm_grouped.pupil_change <= 0].std, 2)})')

print(f'Vividness (dark): {np.round(dm_processed.rating[dm_processed.stim_color == "dark"].mean,2)} (STD = {np.round(dm_processed.rating[dm_processed.stim_color == "dark"].std, 2)})')
print(f'Vividness (light): {np.round(dm_processed.rating[dm_processed.stim_color == "bright"].mean,2)} (STD = {np.round(dm_processed.rating[dm_processed.stim_color == "bright"].std, 2)})')

# =============================================================================
#%% Statistical analyses (functions)
# =============================================================================
# Stats
def lm_pupil(dm_tst, formula=False, re_formula="1", pupil_change=False, reml=False, method='Powell', bl=False):
    """Test how brightness (dark vs. light) affects the mean pupil size."""  
    # Copy of dm
    dm_test = dm_tst.subject_id != ''    
    
    # Remove Nans
    dm_valid_data = dm_test.mean_pupil != NAN # remove NaNs 
    dm_valid_data = dm_valid_data.mean_perception != NAN # remove NaNs 
        
    if bl==True:
        dm_valid_data = dm_valid_data.mean_baseline != NAN # remove NaNs 
        
    if pupil_change == True:
        # Suppress warnings because it's annoying
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        dm_valid_data = ops.group(dm_valid_data, by=[dm_valid_data.subject_id]) # add dm_sub.response_lang if necessary

        # Make sure to have only unique mean values for each variable per participant 
        for col in dm_valid_data.column_names:
            if type(dm_valid_data[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                dm_valid_data[col] = srs.reduce(dm_valid_data[col]) # Compute the mean per subtype 
        
        # Unable back the warnings
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)
        warnings.filterwarnings("default", category=UserWarning)  
    
        dm_valid_data = dm_valid_data.pupil_change != NAN # make sure there's always at least 2 stories to compare

    # The model
    md = mixedlm(formula, dm_valid_data, 
                     groups='subject_id',
                     re_formula=re_formula)
    
    mdf = md.fit(reml=reml, method=method)
        
    return mdf

def test_correlation(dm_c, x, y, alt='two-sided', color='red', lab='vividness', fig=True):
    """Test the correlations between pupil measures and questionnaire measures using Spearman's correlation."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Group per participant 
    dm_cor = ops.group(dm_c, by=[dm_c.subject_id])

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_cor.column_names:
        if type(dm_cor[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_cor[col] = srs.reduce(dm_cor[col], operation=np.nanmean)
            
    # The variables to test the correlation
    x, y = dm_cor[x], dm_cor[y]
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    # Compute spearman's rank correlation
    cor=spearmanr(x, y, alternative=alt)

    N = len(dm_c.subject_id.unique)
    if cor.pvalue  > 1.0:
        pval = 1.0
    elif cor.pvalue  < 0.001:
        pval = '{:.1e}'.format(cor.pvalue )
    else:
        pval = np.round(cor.pvalue, 3)
        
    if fig == False:
        res = fr'{chr(961)} = {round(cor.correlation, 3)}, p = {pval}, n = {N}'
    else:
        res = fr'{chr(961)} = {round(cor.correlation, 3)}, p = {pval}, n = {N}'
    print(res)
    
    # Plot the correlations (linear regression model fit)
    # if lab != False:
    #     label = fr'{lab}: {res}'
    # else:
    #     label = ''
    
    if fig == True:
        sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=False, color=color, label=lab, x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.5, 's': 100}, robust=True)
        plt.legend(frameon=False, markerscale=1.5, loc='upper left', fontsize=15)
        # use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression)
        sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=True, color=color, label=None, x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.0}, line_kws={'linestyle': 'dashed', 'alpha':0.6, 'linewidth': 8})
        
    return res

def compare_models(model1, model2, ddf):
    """Null hypothesis: The simpler model is true. 
    Log-likelihood of the model 1 for H0 must be <= LLF of model 2."""
    print(f'Log-likelihood of model 1 <= model 2: {model1.llf <= model2.llf}')
    
    ratio = (model1.llf - model2.llf)*-2
    p = chi2.sf(ratio, ddf) # How many more DoF does M2 has as compared to M1?
    if p >= .05:
        print(f'The simpler model is the better one (LLF M1: {round(model1.llf,3)}, LLF M2: {round(model2.llf,3)}, ratio = {round(ratio,3)}, df = {ddf}, p = {round(p,4)})')
    else:
        print(f'The simpler model is not the better one (LLF M1: {round(model1.llf,3)}, LLF M2: {round(model2.llf,3)}, ratio = {round(ratio,3)}, df = {ddf}, p = {round(p,4)})')

def check_assumptions(model):
    """Check assumptions for normality of residuals and homoescedasticity.
    Code from: https://www.pythonfordatascience.org/mixed-effects-regression-python/#assumption_check"""
    plt.rcParams['font.size'] = 40
    print('Assumptions check:')
    fig = plt.figure(figsize = (25, 16))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    fig.suptitle(f'{model.model.formula} (n = {model.model.n_groups})')
    # Normality of residuals
    ax1 = plt.subplot(2,2,1)
    sns.distplot(model.resid, hist = True, kde_kws = {"fill" : True, "lw": 4}, fit = stats.norm)
    ax1.set_title("KDE Plot of Model Residuals (Red)\nand Normal Distribution (Black)", fontsize=30)
    ax1.set_xlabel("Residuals")
    
    # Q-Q PLot
    ax2 = plt.subplot(2,2,2)
    sm.qqplot(model.resid, dist = stats.norm, line = 's', ax = ax2, alpha=0.5, markerfacecolor='black', markeredgecolor='black')
    ax2.set_title("Q-Q Plot", fontsize=30)
    
    # Shapiro
    labels1 = ["Statistic", "p-value"]
    norm_res = stats.shapiro(model.resid)
    print('Shapir-Wilk test of normality')
    for key, val in dict(zip(labels1, norm_res)).items():
        print(key, val)
    lab1 = f'Shapiro (normality): Statistic = {np.round(norm_res[0],3)}, p = {np.round(norm_res[1],3)}'

    # Homogeneity of variances
    ax3 = plt.subplot(2,2,3)
    sns.scatterplot(y = model.resid, x = model.fittedvalues, alpha=0.8)
    ax3.set_title("RVF Plot", fontsize=30)
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("Residuals")
    
    ax4 = plt.subplot(2,2,4)
    sns.boxplot(x = model.model.groups, y = model.resid)
    plt.xticks(range(0, len(model.model.group_labels)), range(1, len(model.model.group_labels)+1), fontsize=15)
    ax4.set_title("Distribution of Residuals for Weight by Litter", fontsize=30)
    ax4.set_ylabel("Residuals")
    ax4.set_xlabel("Litter")
    
    # White’s Lagrange Multiplier Test for Heteroscedasticity
    print('White’s Lagrange Multiplier Test for Heteroscedasticity')
    het_white_res = het_white(model.resid, model.model.exog)
    labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    for key, val in dict(zip(labels, het_white_res)).items():
        print(key, val)
    lab2 = f'LM Test (homoscedasticity): LM Statistic = {np.round(het_white_res[0],3)}, p = {np.round(het_white_res[1],3)}'
    
    fig.supxlabel(f'{lab1}\n{lab2}')
    plt.tight_layout()
    plt.show()
    
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=UserWarning)

# =============================================================================
#Analyses
# =============================================================================
m0 = lm_pupil(dm_grouped, formula='mean_perception ~ stim_color', re_formula='1', reml=True)
check_assumptions(m0)
print(m0.summary())

m1 = lm_pupil(dm_grouped, formula='mean_pupil ~ stim_color', re_formula='1')
check_assumptions(m1)
print(m1.summary())

m2 = lm_pupil(dm_grouped, formula='mean_pupil ~ stim_color * rating', re_formula='1')
check_assumptions(m2)
print(m2.summary())

compare_models(m1, m2, 2) # compare models

# m3 = lm_pupil(dm_grouped, formula='pupil_change ~ rating', re_formula='1', pupil_change=True, reml=True)
# check_assumptions(m3)
# print(m3.summary())

test_correlation(dm_grouped, 'pupil_change', 'rating', alt='two-sided', fig=False)
test_correlation(dm_grouped, 'pupil_change', 'QMI_visual', alt='two-sided', fig=False)
test_correlation(dm_grouped, 'pupil_change', 'percept_change', alt='two-sided', fig=False)
test_correlation(dm_grouped, 'rating', 'QMI_visual', alt='two-sided', fig=False)

# Supp analyses 
# Baseline checks
wilcoxon(dm_grouped.mean_baseline[dm_grouped.stim_color=='bright'], dm_grouped.mean_baseline[dm_grouped.stim_color=='dark'], alternative='less')
mA = lm_pupil(dm_processed, formula='mean_baseline ~ stim_color', re_formula='1 + stim_orientation', reml=True, bl=True)
print(mA.summary())

# Trial-level analyses
m0 = lm_pupil(dm_processed, formula='mean_perception ~ stim_color', re_formula='1 + stim_orientation', reml=True)
check_assumptions(m0)
print(m0.summary())

m1 = lm_pupil(dm_processed, formula='mean_pupil ~ stim_color', re_formula='1 + stim_orientation')
check_assumptions(m1)
print(m1.summary())

m2 = lm_pupil(dm_processed, formula='mean_pupil ~ stim_color * rating', re_formula='1 + stim_orientation')
check_assumptions(m2)
print(m2.summary())

compare_models(m1, m2, 2) # compare models

# Exploratory
dm_1 = dm_grouped.rating > 2 # high imagers
dm_2 = dm_grouped.rating <= 2 # low imagers

test_correlation(dm_1, 'percept_change', 'pupil_change', alt='two-sided', fig=False)
test_correlation(dm_1, 'rest_change', 'rating', alt='less', fig=False)
N = len(dm_1.subject_id.unique)
pos_effects = set(dm_1.subject_id[dm_1.pupil_change > 0])
print(f'Positive effects: {np.round(len(pos_effects)/N * 100,2)}% ({len(pos_effects)}/{N})')

test_correlation(dm_2, 'percept_change', 'pupil_change', alt='two-sided', fig=False)
test_correlation(dm_2, 'rest_change', 'rating', alt='less', fig=False)
N = len(dm_2.subject_id.unique)
pos_effects = set(dm_2.subject_id[dm_2.pupil_change > 0])
print(f'Positive effects: {np.round(len(pos_effects)/N * 100,2)}% ({len(pos_effects)}/{N})')

test_correlation(dm_grouped, 'pupil_change', 'mean_perception', alt='two-sided', fig=False)

test_correlation(dm_grouped, 'percept_change', 'rating', alt='two-sided', fig=False)
test_correlation(dm_grouped, 'rest_change', 'rating', alt='less', fig=False)

dm_plot = dm_grouped.subject_id != ''     
for s, sdm in ops.split(dm_plot.subject_id):    
    dm_plot.rating[sdm] = sdm.rating.mean

for v, sdm in ops.split(dm_plot.rating):    
    plot_dm(sdm, all_plots=False, title=f'Vividness rating == {v}') # visualise grouped data
    
# =============================================================================
#%% Visualise 
# =============================================================================
# Main effects
plot_dm(dm_grouped, grouped=True) # visualise grouped data

# Individual effects
plt.rcParams['font.size'] = 40
dm_grouped = ops.sort(dm_grouped, by=dm_grouped.pupil_change)
dm_df = convert.to_pandas(dm_grouped)

fig = plt.figure(figsize=(20,13)) # 25 13
ax1=plt.subplot(1,1,1)
plt.title('Imagery first (IF) version')
sns.pointplot(data=dm_df, x='subject_id', y='pupil_change', hue=None, hue_order=None, scale=3.0, palette=['black'], join=True)
#sns.pointplot(data=dm_df, x='subject_id', y='pupil_change', hue='aphant', hue_order=['N', 'Y', 'M'], scale=3.0, palette=['black', 'red', 'orange'])
plt.xlabel('Participants', color='black');plt.ylabel('Pupil-size mean differences\n(dark - bright) (a.u)', color='black', fontsize=40)
handles, labels = ax1.get_legend_handles_labels()
#plt.legend('')
#fig.legend(handles, ['No', 'Yes', 'Maybe'], loc='upper center', title='Do you think this definition of aphantasia fits you?', ncol=3, frameon=False, fontsize=40)
ax1.spines['bottom'].set_visible(True)
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_color('black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.xticks([]);plt.yticks(color='black')
plt.xlim([-1, 51])
plt.axhline(0, color='black')
plt.show()

# Supp Correlations
plt.figure(figsize=(8,8))
plt.title('Imagery-first version')
test_correlation(dm_grouped, x='pupil_change', y='percept_change', lab='Perception', alt='greater', color='violet')
test_correlation(dm_grouped, x='pupil_change', y='rest_change', lab='Rest', alt='greater', color='orange')
plt.xlabel('Pupil-size changes during imagery (a.u.)', fontsize=20);plt.ylabel('Pupil-size differences\nduring perception or rest (a.u.)', fontsize=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
plt.title('Imagery-first version')
test_correlation(dm_grouped, x='mean_perception', y='percept_change', lab='Perception', alt='less', color='violet')
test_correlation(dm_grouped, x='mean_perception', y='pupil_change', lab='Imagery', alt='greater', color='orange')
plt.xlabel('Mean pupil size during perception (a.u.)', fontsize=20);plt.ylabel('Pupil-size differences (a.u.)', fontsize=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
plt.title('Imagery-first version')
test_correlation(dm_grouped, x='mean_rest', y='rest_change', lab='Rest', alt='less', color='violet')
test_correlation(dm_grouped, x='mean_rest', y='pupil_change', lab='Imagery', alt='less', color='orange')
plt.xlabel('Mean pupil size during rest (a.u.)', fontsize=20);plt.ylabel('Pupil-size differences (a.u.)', fontsize=20)
plt.tight_layout()
plt.show()

dm_bright = dm_grouped.stim_color == 'bright'
dm_dark = dm_grouped.stim_color == 'dark'

plt.figure(figsize=(8,6))
plt.title('Imagery-first version')
test_correlation(dm_bright, 'mean_pupil', 'mean_perception', lab='bright', color=red[1])
test_correlation(dm_dark, 'mean_pupil', 'mean_perception', lab='dark', color=blue[1])
plt.xlabel('Mean pupil size during imagery');plt.ylabel('Mean pupil size during perception')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.title('Imagery-first version')
test_correlation(dm_bright, 'mean_pupil', 'mean_rest', lab='bright', color=red[1])
test_correlation(dm_dark, 'mean_pupil', 'mean_rest', lab='dark', color=blue[1])
plt.xlabel('Mean pupil size during imagery');plt.ylabel('Mean pupil size during rest')
plt.tight_layout()
plt.show()