#%% Initial statements
# Third party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_set_backend
import pandas as pd
from collections import OrderedDict
import sys
from termcolor import colored
import pickle
import pdb
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare, wilcoxon
import statsmodels.stats.multitest as smm
from itertools import combinations
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import simpson
import seaborn as sns
from statannot import add_stat_annotation
import scikit_posthocs as sp


#%%
def load_and_stack_participant_data(results_dir):
    """
    Load all participant data from the results directory and stack them into a single dictionary.
    
    Parameters:
    results_dir (str): Directory where processed participant data (pkl files) are saved.
    
    Returns:
    dict: Dictionary containing stacked participant data.
    """
    # Initialize an empty dictionary to store stacked features for all participants
    stacked_data = {}
    
    # Iterate through each .pkl file in the results directory
    for file_name in os.listdir(results_dir):
        if file_name.endswith('_features.pkl'):
            participant_id = file_name.split('_features.pkl')[0]
            
            # Load the data from the .pkl file
            with open(os.path.join(results_dir, file_name), 'rb') as file:
                participant_data = pickle.load(file)
            
            # Add the loaded data to the stacked_data dictionary, under the participant ID
            stacked_data[participant_id] = participant_data
    
    return stacked_data

def permutation_test(condition1, condition2, n_permutations=1000, ci_level=0.95):
    """
    Perform a permutation test for paired comparisons of Global Field Power (GFP).
    
    condition1: Global Field Power for condition 1 (shape: subjects x timepoints)
    condition2: Global Field Power for condition 2 (shape: subjects x timepoints)
    n_permutations: Number of permutations to run
    
    Returns the p-value and the observed mean difference.
    """
    # Calculate the observed difference in GFP between the two conditions
    observed_diff = np.mean(condition1 - condition2)
    
    # Initialize a list to store permutation differences
    perm_diffs = []
    
    # Perform permutation testing
    for _ in range(n_permutations):
        # Randomly swap the labels for each subject
        swap_labels = np.random.choice([0, 1], size=condition1.shape[0])
        
        # Create permuted conditions by swapping each subject individually
        perm_condition1 = condition1.copy()
        perm_condition2 = condition2.copy()
        for i in range(condition1.shape[0]):
            if swap_labels[i] == 1:
                perm_condition1[i, :], perm_condition2[i, :] = perm_condition2[i, :], perm_condition1[i, :]
        
        # Calculate the mean difference for this permutation
        perm_diff = np.mean(perm_condition1 - perm_condition2)
        perm_diffs.append(perm_diff)
    
    # Convert permutation differences to numpy array
    perm_diffs = np.array(perm_diffs)
    # Calculate the p-value as the proportion of permuted differences greater than or equal to the observed difference
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_permutations
    
    # Sort permutation differences to compute confidence intervals
    sorted_perm_diffs = np.sort(perm_diffs)
    
    # Calculate the percentiles for the confidence interval
    lower_bound = np.percentile(sorted_perm_diffs, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(sorted_perm_diffs, (1 + ci_level) / 2 * 100)
    
    # Return observed difference, p-value, and confidence intervals
    return observed_diff, p_value, (lower_bound, upper_bound)

# Function to compute the average over selected regions
def average_roi(data, roi, mapping):
    roi_indices = [i for i, electrode in enumerate(mapping.keys()) if electrode in roi]
    return np.mean(data[roi_indices,:], axis=0)
    
def plot_figure_2_4(all_participant_data):

    # Extract evoked data for each condition
    evoked_average = {
        'neutro': [participant_data['Evoked']['neutro'] for participant_data in all_participant_data.values()],
        'piacevole': [participant_data['Evoked']['piacevole'] for participant_data in all_participant_data.values()],
        'spiacevole': [participant_data['Evoked']['spiacevole'] for participant_data in all_participant_data.values()]
    }
    # Calculate the mean evoked data across participants for each condition
    evoked_n = np.mean(np.stack(evoked_average['neutro']), axis=0)
    evoked_p = np.mean(np.stack(evoked_average['piacevole']), axis=0)
    evoked_s = np.mean(np.stack(evoked_average['spiacevole']), axis=0)
    times = all_participant_data['S01']['GFP times']['piacevole']
        
    #% ---> Figure 2A
    fig, axs = plt.subplots(3,1, sharex = True, sharey = True, figsize = (6,5))
    axs[0].plot(times,evoked_n.T, '-', color = 'blue', alpha = 0.1)
    axs[1].plot(times,evoked_p.T, '-', color = 'green', alpha = 0.1)
    axs[2].plot(times,evoked_s.T, '-', color = 'red', alpha = 0.1)
    axs[2].set_xlim([-0.1,0.4])
    axs[2].set_ylim([-5E-6,5E-6])
    axs[0].axvline(ymin= 0, linestyle = '-', linewidth = 5, alpha = 0.4, color = 'grey')
    axs[1].axvline(ymin= 0, linestyle = '-', linewidth = 5, alpha = 0.4, color = 'grey')
    axs[2].axvline(ymin= 0, linestyle = '-', linewidth = 5, alpha = 0.4, color = 'grey')
    axs[2].set_xlabel('Time [s]')
    axs[0].set_ylabel('Early ERP [V]')
    axs[1].set_ylabel('Early ERP [V]')
    axs[2].set_ylabel('Early ERP [V]')
    axs[0].set_title('Neutral odor')
    axs[1].set_title('Pleasant odor')
    axs[2].set_title('Unpleasant odor')

    print('-- PTP calculation in 0-100 ms')
    ix_times = (times > 0) & (times < 0.1)
    t_ptp = times[ix_times]
    
    # Neutral condition
    tmp = evoked_n[:, ix_times]
    m = np.argmax(np.max(tmp, axis=0) - np.min(tmp, axis=0))
    axs[0].axvline(x=t_ptp[m] - 0.003, linestyle='--', linewidth=2, alpha=0.4, color='black')
    max_ptp = np.max(tmp[:, m]) - np.min(tmp[:, m])
    print(f'Mean latency on max ptp (Neutral): {t_ptp[m]}, Potential: {max_ptp}')
    
    # Pleasant condition
    tmp = evoked_p[:, ix_times]
    m = np.argmax(np.max(tmp, axis=0) - np.min(tmp, axis=0))
    axs[1].axvline(x=t_ptp[m], linestyle='--', linewidth=2, alpha=0.4, color='black')
    max_ptp = np.max(tmp[:, m]) - np.min(tmp[:, m])
    print(f'Mean latency on max ptp (Pleasant): {t_ptp[m]}, Potential: {max_ptp}')
    
    # Unpleasant condition
    tmp = evoked_s[:, ix_times]
    m = np.argmax(np.max(tmp, axis=0) - np.min(tmp, axis=0))
    axs[2].axvline(x=t_ptp[m], linestyle='--', linewidth=2, alpha=0.4, color='black')
    max_ptp = np.max(tmp[:, m]) - np.min(tmp[:, m])
    print(f'Mean latency on max ptp (Unpleasant): {t_ptp[m]}, Potential: {max_ptp}')
    
    print('-- PTP calculation in 0-300 ms')
    ix_times = (times > 0) & (times < 0.3)
    t_ptp = times[ix_times]
    
    # Neutral condition
    tmp = evoked_n[:, ix_times]
    m = np.argmax(np.max(tmp, axis=0) - np.min(tmp, axis=0))
    axs[0].axvline(x=t_ptp[m], linestyle='--', linewidth=2, alpha=0.7, color='black')
    max_ptp = np.max(tmp[:, m]) - np.min(tmp[:, m])
    print(f'Mean latency on max ptp (Neutral): {t_ptp[m]}, Potential: {max_ptp}')
    
    # Pleasant condition
    tmp = evoked_p[:, ix_times]
    m = np.argmax(np.max(tmp, axis=0) - np.min(tmp, axis=0))
    axs[1].axvline(x=t_ptp[m], linestyle='--', linewidth=2, alpha=0.7, color='black')
    max_ptp = np.max(tmp[:, m]) - np.min(tmp[:, m])
    print(f'Mean latency on max ptp (Pleasant): {t_ptp[m]}, Potential: {max_ptp}')
    
    # Unpleasant condition
    tmp = evoked_s[:, ix_times]
    m = np.argmax(np.max(tmp, axis=0) - np.min(tmp, axis=0))
    axs[2].axvline(x=t_ptp[m], linestyle='--', linewidth=2, alpha=0.7, color='black')
    max_ptp = np.max(tmp[:, m]) - np.min(tmp[:, m])
    print(f'Mean latency on max ptp (Unpleasant): {t_ptp[m]}, Potential: {max_ptp}')
    
    fig.tight_layout()
    fig.savefig('Figure 2.svg', format = 'svg')
        
    #% ---> Figure 2A: analysis
    evoked_ptp = {'Width': [], 'Latency': [],
                  'Condition': [], 'Window': []}
    for cond in ['neutro', 'piacevole', 'spiacevole']:
        for window in [0.1,0.33]:
            w_name = '0_'+str(window)
            for participant_data in all_participant_data.values():                
                ix_times = (times > 0) & (times < window)
                t_ptp = times[ix_times]
                tmp = participant_data['Evoked'][cond][:,ix_times]
                m = np.argmax(np.max(tmp,axis = 0) - np.min(tmp,axis = 0))
                evoked_ptp['Width'].append(np.max(np.max(tmp,axis = 0) - np.min(tmp,axis = 0)))
                evoked_ptp['Latency'].append(t_ptp[np.argmax(np.max(tmp,axis = 0) - np.min(tmp,axis = 0))])
                evoked_ptp['Condition'].append(cond)
                evoked_ptp['Window'].append(w_name)
    
    evoked_ptp = pd.DataFrame(evoked_ptp)
    palette = {"neutro": "blue", "piacevole": "green", "spiacevole": "red"}
    box_pairs = [("neutro", "piacevole"), ("neutro", "spiacevole"), ("piacevole", "spiacevole")]

    # Create subplots
    fig_2box, axs_2box = plt.subplots(2, 2, figsize=(6,6), sharey=False)
    
    # Boxplot for Latency comparison for Window 0.1
    data_0_0_1 = evoked_ptp[evoked_ptp['Window'] == '0_0.1']
    bp = sns.boxplot(data=evoked_ptp[evoked_ptp['Window'] == '0_0.1'], 
                x='Condition', y='Latency', ax=axs_2box[0,0],
                palette = palette, showfliers = False)
    axs_2box[0,0].set_title('Window: 0-100ms')
    axs_2box[0,0].set_ylabel('Latency of max. difference\n in ERP amplitude [s]')
    add_stat_annotation(bp, data=evoked_ptp[evoked_ptp['Window'] == '0_0.1'],
                    x='Condition', y='Latency', box_pairs=box_pairs,
                    test='t-test_paired', text_format='star', loc='inside', verbose=2)
    conditions = data_0_0_1['Condition'].unique()
    latency_data = [data_0_0_1[data_0_0_1['Condition'] == cond]['Latency'] for cond in conditions]
    friedman_result = friedmanchisquare(*latency_data)
    print(f"Friedman test result for Latency (0-0.1ms): {friedman_result}")
    
    # Boxplot for Latency comparison for Window 0.3
    data_0_0_33 = evoked_ptp[evoked_ptp['Window'] == '0_0.33']
    bp = sns.boxplot(data=evoked_ptp[evoked_ptp['Window'] == '0_0.33'],
                x='Condition', y='Latency', ax=axs_2box[0,1],
                palette = palette, showfliers = False)
    axs_2box[0,1].set_title('Window: 0-300ms')
    axs_2box[0,1].set_ylabel('Latency of max. difference\n in ERP amplitude [s]')
    add_stat_annotation(bp, data=evoked_ptp[evoked_ptp['Window'] == '0_0.33'],
                    x='Condition', y='Latency', box_pairs=box_pairs,
                    test='t-test_paired', text_format='star', loc='inside', verbose=2)
    latency_data = [data_0_0_33[data_0_0_33['Condition'] == cond]['Latency'] for cond in conditions]
    friedman_result = friedmanchisquare(*latency_data)
    print(f"Friedman test result for Latency (0-0.3ms): {friedman_result}")

    # Boxplot for Width comparison for Window 0.1
    bp = sns.boxplot(data=evoked_ptp[evoked_ptp['Window'] == '0_0.1'], 
                x='Condition', y='Width', ax=axs_2box[1,0],
                palette = palette, showfliers = False)
    axs_2box[1,0].set_ylabel('Maximum difference in\nERP amplitude [V]')
    add_stat_annotation(bp, data=evoked_ptp[evoked_ptp['Window'] == '0_0.1'],
                    x='Condition', y='Width', box_pairs=box_pairs,
                    test='t-test_paired', text_format='star', loc='inside', verbose=2)
    width_data = [data_0_0_1[data_0_0_1['Condition'] == cond]['Width'] for cond in conditions]
    friedman_result = friedmanchisquare(*width_data)
    print(f"Friedman test result for Width (0-0.1ms): {friedman_result}")

    # Boxplot for Width comparison for Window 0.3
    bp = sns.boxplot(data=evoked_ptp[evoked_ptp['Window'] == '0_0.33'], 
                x='Condition', y='Width', ax=axs_2box[1,1],
                palette = palette, showfliers = False)
    add_stat_annotation(bp, data=evoked_ptp[evoked_ptp['Window'] == '0_0.33'],
                    x='Condition', y='Width', box_pairs=box_pairs,
                    test='t-test_paired', text_format='star', loc='inside', verbose=2)
    width_data = [data_0_0_33[data_0_0_33['Condition'] == cond]['Width'] for cond in conditions]
    friedman_result = friedmanchisquare(*width_data)
    print(f"Friedman test result for Width (0-0.3ms): {friedman_result}")
    
    axs_2box[1,1].set_ylabel('Maximum difference in\nERP amplitude [V]')
    axs_2box[1,1].set_ylim([0.5E-5, 3.2E-5])
    axs_2box[1,0].set_ylim([0.5E-5, 3.2E-5])
    
    for a in axs_2box.ravel():
        a.set_xlabel('')
        a.set_xticklabels(['Neutral', 'Pleasant','Unpleasant'], rotation = 30)
    fig_2box.tight_layout()
    fig_2box.savefig('Figure 3.svg', format = 'svg')
    
    #% ---> Figure 2B
    roi_frontal = ['EEG AF7', 'EEG AF3', 'EEG Fp1', 'EEG Fp2', 'EEG AF4', 'EEG AF8', 'EEG F7', 'EEG F5', 
                   'EEG F3', 'EEG F1', 'EEG F2', 'EEG F4', 'EEG F6', 'EEG F8', 'EEG Fpz', 'EEG Fz']
    roi_central = ['EEG FC5', 'EEG FC3', 'EEG FC1', 'EEG FC2', 'EEG FC4', 'EEG FC6', 'EEG Cz', 'EEG FCz']
    roi_temporal = ['EEG FT7', 'EEG FT8', 'EEG T3', 'EEG T4', 'EEG T5', 'EEG T6', 'EEG TP7', 'EEG TP8']
    roi_parietal = ['EEG P5', 'EEG P3', 'EEG P1', 'EEG P2', 'EEG P4', 'EEG P6', 'EEG Pz', 
                    'EEG CP1', 'EEG CP2', 'EEG CP3', 'EEG CP4', 'EEG CP5', 'EEG CP6', 'EEG CPz']
    roi_occipital = ['EEG PO7', 'EEG PO3', 'EEG O1', 'EEG O2', 'EEG PO4', 'EEG PO8', 'EEG Oz', 'EEG POz']
    mapping =  {
               'EEG AF7': 'eeg', 'EEG AF3': 'eeg', 'EEG Fp1': 'eeg', 'EEG Fp2': 'eeg',
               'EEG AF4': 'eeg', 'EEG AF8': 'eeg', 'EEG F7': 'eeg', 'EEG F5': 'eeg',
               'EEG F3': 'eeg', 'EEG F1': 'eeg', 'EEG F2': 'eeg', 'EEG F4': 'eeg',
               'EEG F6': 'eeg', 'EEG F8': 'eeg', 'EEG FT7': 'eeg', 'EEG FC5': 'eeg',
               'EEG FC3': 'eeg', 'EEG FC1': 'eeg', 'EEG FC2': 'eeg', 'EEG FC4': 'eeg',
               'EEG FC6': 'eeg', 'EEG FT8': 'eeg', 'EEG T3': 'eeg', 'EEG C5': 'eeg',
               'EEG C3': 'eeg', 'EEG C1': 'eeg', 'EEG C2': 'eeg', 'EEG C4': 'eeg',#28
               'EEG C6': 'eeg', 'EEG T4': 'eeg', 'EEG TP7': 'eeg', 'EEG CP5': 'eeg',
               'EEG CP3': 'eeg', 'EEG CP1': 'eeg', 'EEG CP2': 'eeg', 'EEG CP4': 'eeg',
               'EEG CP6': 'eeg', 'EEG TP8': 'eeg', 'EEG T5': 'eeg', 'EEG P5': 'eeg',
               'EEG P3': 'eeg', 'EEG P1': 'eeg', 'EEG P2': 'eeg', 'EEG P4': 'eeg',
               'EEG P6': 'eeg', 'EEG T6': 'eeg', 'EEG Fpz': 'eeg', 'EEG PO7': 'eeg',
               'EEG PO3': 'eeg', 'EEG O1': 'eeg', 'EEG O2': 'eeg', 'EEG PO4': 'eeg',
               'EEG PO8': 'eeg', 'EEG Oz': 'eeg', 'EEG AFz': 'eeg', 'EEG Fz': 'eeg',
               'EEG FCz': 'eeg', 'EEG Cz': 'eeg', 'EEG CPz': 'eeg', 'EEG Pz': 'eeg',
               'EEG POz': 'eeg'
               }
        
    rois = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    roi_data = {
        'Frontal': [average_roi(evoked_n, roi_frontal, mapping), average_roi(evoked_p, roi_frontal, mapping), average_roi(evoked_s, roi_frontal, mapping)],
        'Central': [average_roi(evoked_n, roi_central, mapping), average_roi(evoked_p, roi_central, mapping), average_roi(evoked_s, roi_central, mapping)],
        'Temporal': [average_roi(evoked_n, roi_temporal, mapping), average_roi(evoked_p, roi_temporal, mapping), average_roi(evoked_s, roi_temporal, mapping)],
        'Parietal': [average_roi(evoked_n, roi_parietal, mapping), average_roi(evoked_p, roi_parietal, mapping), average_roi(evoked_s, roi_parietal, mapping)],
        'Occipital': [average_roi(evoked_n, roi_occipital, mapping), average_roi(evoked_p, roi_occipital, mapping), average_roi(evoked_s, roi_occipital, mapping)]
    }
    # Plotting
    fig2, axs = plt.subplots(len(rois), 1, sharex=True, sharey=True, figsize=(4,6))
    for i, roi in enumerate(rois):
        axs[i].plot(times, roi_data[roi][0], '-b', alpha=0.7, label='Neutral odor')       # Neutro condition
        axs[i].plot(times, roi_data[roi][1], '-r', alpha=0.7, label='Pleasant odor')    # Piacevole condition
        axs[i].plot(times, roi_data[roi][2], '-g', alpha=0.7, label='Unpleasant odor')   # Spiacevole condition
        axs[i].axvline(x=0, ymin=0, linestyle='--', linewidth=2, alpha=0.7, color='grey')
        axs[i].set_title(f'{roi} ROI')
        axs[i].set_ylabel('Early ERP [V]')
    axs[0].legend(loc='upper right')
    axs[-1].set_xlim([-0.1, 0.4])
    axs[-1].set_ylim([-3E-6, 3E-6])
    plt.xlabel('Time [s]')
    fig2.tight_layout()
    
    fig2.savefig('Figure 4.svg', format = 'svg')
    # GROUPED ROIS
    # Find 0 to 100ms negative deflection in occipital ROI
    # Modify tmin tmax, min and argmin to study different rois
    time_min = 0.01
    time_max = 0.1

    time_indices = np.where((times >= time_min) & (times <= time_max))[0]
    df = pd.DataFrame()
    for condition in ['neutro', 'piacevole', 'spiacevole']:
        for patient_idx, data in enumerate(evoked_average[condition]):
            mean_data = average_roi(data, roi_occipital, mapping)
            data_in_range = mean_data[time_indices]
            peak_value = np.min(data_in_range)
            peak_index = np.argmin(data_in_range)
            peak_latency = times[time_indices][peak_index]
            # Append to the dataframe
            new_row = [condition, patient_idx, peak_latency, peak_value]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.columns = ['Condition', 'Patient', 'Peak Latency (ms)', 'Peak Value']
    df.to_excel('Data_Figure_2.xlsx')
    
    # Compute central estimators
    mean_df = df.groupby('Condition').agg({
        'Peak Latency (ms)': 'median',
        'Peak Value': 'median'
    }).reset_index()
    mean_df.columns = ['Condition', 'Mean Peak Latency (ms)', 'Mean Peak Value']
    iqr_df = df.groupby('Condition').agg({
        'Peak Latency (ms)': lambda x: x.quantile(0.75) - x.quantile(0.25),
        'Peak Value': lambda x: x.quantile(0.75) - x.quantile(0.25)
    }).reset_index()
    iqr_df.columns = ['Condition', 'IQR Peak Latency (ms)', 'IQR Peak Value']
    
    # Statistical test: latencies
    anova_results = stats.f_oneway(
        df[df['Condition'] == 'neutro']['Peak Latency (ms)'],
        df[df['Condition'] == 'piacevole']['Peak Latency (ms)'],
        df[df['Condition'] == 'spiacevole']['Peak Latency (ms)']
    )
    print(f"ANOVA results: F={anova_results.statistic}, p-value={anova_results.pvalue}")
    if anova_results.pvalue < 0.05:
        tukey = pairwise_tukeyhsd(
            endog=df['Peak Latency (ms)'],  # Dependent variable (peak latency)
            groups=df['Condition'],  # Independent variable (condition)
            alpha=0.05  # Significance level
        )
        print(tukey)
    else:
        print("ANOVA test is not significant. No post-hoc test performed.")
        

    # Statistical test: peaks
    anova_results = stats.f_oneway(
        df[df['Condition'] == 'neutro']['Peak Value'],
        df[df['Condition'] == 'piacevole']['Peak Value'],
        df[df['Condition'] == 'spiacevole']['Peak Value']
    )
    print(f"ANOVA results: F={anova_results.statistic}, p-value={anova_results.pvalue}")
    if anova_results.pvalue < 0.05:
        tukey = pairwise_tukeyhsd(
            endog=df['Peak Latency (ms)'],  # Dependent variable (peak latency)
            groups=df['Condition'],  # Independent variable (condition)
            alpha=0.05  # Significance level
        )
        print(tukey)
    else:
        print("ANOVA test is not significant. No post-hoc test performed.")
            
        
    return fig, fig2

def plot_figure_5_6(all_participant_data):
    
    # GFP plot
    palette = {"neutro": "blue", "piacevole": "green", "spiacevole": "red"}
    sigma = 5
    ixs = 8
    gfp = {
        'neutro': [gaussian_filter1d(participant_data['GFP mean']['neutro'], sigma = ixs*sigma) for participant_data in all_participant_data.values()],
        'piacevole': [gaussian_filter1d(participant_data['GFP mean']['piacevole'], sigma = ixs*sigma) for participant_data in all_participant_data.values()],
        'spiacevole': [gaussian_filter1d(participant_data['GFP mean']['spiacevole'], sigma = ixs*sigma) for participant_data in all_participant_data.values()]
    }

    neu_m = np.mean(np.stack(gfp['neutro']), axis=0)
    neu_s = np.std(np.stack(gfp['neutro']), axis=0)/sigma
    pia_m = np.mean(np.stack(gfp['piacevole']), axis=0)
    pia_s = np.std(np.stack(gfp['piacevole']), axis=0)/sigma
    spia_m = np.mean(np.stack(gfp['spiacevole']), axis=0)
    spia_s = np.std(np.stack(gfp['spiacevole']), axis=0)/sigma
    
    times = all_participant_data['S01']['GFP times']['piacevole']        
        
    # Function to compute window means per participant
    def compute_window_means_per_participant(gfp_data, times, windows):
        windows = [(0, 12), (6, 12), (0, 6)]
        window_means = {window: [] for window in windows}
        for participant_data in gfp_data:
            for window in windows:
                start, end = window
                idxs = (times >= start) & (times < end)
                mean_value = np.mean(participant_data[idxs])
                window_means[window].append(mean_value)
        return window_means
    conditions = ['neutro', 'piacevole', 'spiacevole']
    windows = [(0, 12), (0, 6), (6, 12)]
    
    # Compute means for each condition
    gfp_window_means = {
        'neutro': compute_window_means_per_participant(gfp['neutro'], times, windows),
        'piacevole': compute_window_means_per_participant(gfp['piacevole'], times, windows),
        'spiacevole': compute_window_means_per_participant(gfp['spiacevole'], times, windows),
    }
    data = []
    for condition in conditions:
        for window, mean_values in gfp_window_means[condition].items():
            for value in mean_values:
                data.append({'Condition': condition, 'Window': f'{window[0]}-{window[1]}s', 'GFP Mean': value})
    df = pd.DataFrame(data)
    palette = {"neutro": "blue", "piacevole": "green", "spiacevole": "red"}
    legend_labels = {'neutro': 'Neutral odor', 'piacevole': 'Pleasant odor', 'spiacevole': 'Unpleasant odor'}
    # Create the plot using plt.subplots
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.boxplot(x='Window', y='GFP Mean', hue='Condition', data=df, palette=palette, ax=ax, showfliers = False)
    ax.set_ylabel('GFP mean value [V]')
    ax.set_xlabel('Time Windows (s)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_labels[label] for label in labels], title='Condition')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0-12s', '0-6s', '6-12s'])
    plt.tight_layout()
    plt.show()
    fig.savefig('Figure 6.svg', format = 'svg')

    # Function to perform Friedman test and post-hoc pairwise tests
    def perform_friedman_tests(df, conditions, windows):
        import itertools
        from statsmodels.stats.multitest import multipletests
        from scipy.stats import ttest_rel

        results = {}
        post_hoc_results = {}
    
        for window in windows:
            wname = f"{window[0]}-{window[1]}s"
            # Get data for each condition in the current window
            window_data = [df[(df['Condition'] == cond) & (df['Window'] == wname)]['GFP Mean'].values for cond in conditions]
            print(wname)
            # Perform Friedman test
            stat, p_value = friedmanchisquare(*window_data)
            results[window] = {'stat': stat, 'p_value': p_value}
    
            # If the Friedman test is significant, perform post-hoc tests
            if p_value < 0.05:
                pairwise_comparisons = list(itertools.combinations(conditions, 2))
                post_hoc_p_values = []
                for cond1, cond2 in pairwise_comparisons:
                    data1 = df[(df['Condition'] == cond1) & (df['Window'] == wname)]['GFP Mean'].values
                    data2 = df[(df['Condition'] == cond2) & (df['Window'] == wname)]['GFP Mean'].values
                    _, p_val = ttest_rel(data1, data2)
                    post_hoc_p_values.append(p_val)
    
                # Correct p-values using Bonferroni correction
                corrected_p_values = multipletests(post_hoc_p_values, method='bonferroni')[1]
                post_hoc_results[window] = dict(zip(pairwise_comparisons, corrected_p_values))
    
        return results, post_hoc_results

    # Perform tests
    friedman_results, post_hoc_results = perform_friedman_tests(df, conditions, windows)
    
    # Calculate max values for time ranges for boxplot
    time_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
    data_for_boxplot = []
    data_for_growing_avg = []

    for condition, values in gfp.items():
        for i, participant_gfp in enumerate(values):
            for start, end in time_ranges:
                mask = (times >= start) & (times < end)
                max_value = np.max(participant_gfp[mask])
                data_for_boxplot.append({
                    'Time Range': f'{start}-{end}s',
                    'Condition': condition.capitalize(),
                    'Max GFP': max_value
                })
            
            # Calculate growing average (0-1, 0-2, ..., 0-11)
            for end in range(1, 12):
                mask = (times >= 0) & (times < end)
                avg_value = np.max(participant_gfp[mask])
                data_for_growing_avg.append({
                    'Time Range': f'0-{end}s',
                    'Condition': condition.capitalize(),
                    'Max GFP': avg_value
                })
    
    # Convert to DataFrame for seaborn
    boxplot_data = pd.DataFrame(data_for_boxplot)
    growing_avg_data = pd.DataFrame(data_for_growing_avg)

    max_time_neu = times[np.argmax(neu_m)]
    max_time_pia = times[np.argmax(pia_m)]
    max_time_spia = times[np.argmax(spia_m)]

    fig, ax = plt.subplots(1,1, figsize = (5,5))
    ax.plot(times, spia_m, label='Unpleasant odor', color='red', linewidth = 2)
    ax.fill_between(times, spia_m - spia_s, spia_m + spia_s, color='red', alpha=0.2)
    ax.plot(times, pia_m, label='Pleasant odor', color='green', linewidth = 2)
    ax.fill_between(times, pia_m - pia_s, pia_m + pia_s, color='green', alpha=0.2)
    ax.plot(times, neu_m, label='Neutral odor', color='blue', linewidth = 2)
    ax.fill_between(times, neu_m - neu_s, neu_m + neu_s, color='blue', alpha=0.2)
    ax.set_ylim([4E-6, 6.5E-6])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(r'GFP [$\mu$V]')
    ax.set_xlabel('Time [s]')
    ax.set_xlim([-1,12])
    ax.axvline(ymin= 0, linestyle = '--', linewidth = 2, alpha = 0.7, color = 'grey')
    ax.axvline(x = 6, ymin= 0, linestyle = '--', linewidth = 2, alpha = 0.7, color = 'grey')
    ax.axvspan(0, 6, color='grey', alpha=0.2, label="Stimulus presentation")


    fig.legend()
    fig.tight_layout()
    fig.savefig('Figure 5.svg', format = 'svg')
    
    # # Plot boxplot
    # fig, axs = plt.subplots(2,1,figsize=(8, 6))
    # sns.boxplot(data=boxplot_data, x='Time Range', y='Max GFP',
    #             hue='Condition', palette='Set2', ax = axs[0], showfliers = False)
    # sns.boxplot(data=growing_avg_data, x='Time Range', y='Max GFP',
    #             hue='Condition', palette='Set2', ax = axs[1], showfliers = False)
    # axs[1].set_xlabel('Time Range [s]')
    # ax.legend(title='Condition')
    # fig.tight_layout()
    # fig.savefig('temporary1.svg')
    
    #% Growing average
    results_piac_spia_dict = {}
    results_neu_piac_dict = {}
    results_neu_spia_dict = {}
    # Define the sliding window range (0 to 1s up to 0 to 20s)
    for end_time in range(1, 12): 
        print(end_time)
        
        condition_neu = np.stack(gfp['neutro'])[:, times < end_time]
        condition_piac = np.stack(gfp['piacevole'])[:, times < end_time]
        condition_spia = np.stack(gfp['spiacevole'])[:, times < end_time]
        
        # Perform permutation test for 'Pleasant - Unpleasant' and store results in dictionary
        observed_diff, p_value, (lower_bound, upper_bound) = permutation_test(condition_piac, condition_spia)
        results_piac_spia_dict[f"0-{end_time}s"] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'ci': (lower_bound, upper_bound)
        }
        # Perform permutation test for 'Pleasant - Unpleasant' and store results in dictionary
        observed_diff, p_value, (lower_bound, upper_bound) = permutation_test(condition_neu, condition_piac)
        results_neu_piac_dict[f"0-{end_time}s"] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'ci': (lower_bound, upper_bound)
        }
        # Perform permutation test for 'Pleasant - Unpleasant' and store results in dictionary
        observed_diff, p_value, (lower_bound, upper_bound) = permutation_test(condition_neu, condition_spia)
        results_neu_spia_dict[f"0-{end_time}s"] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'ci': (lower_bound, upper_bound)
        }
        
    # # Plot
    figs1, axs = plt.subplots(2,1,figsize = (5,5))
    ci, observed_diff = [], []
    for end_time in range(1, 12):  # 1s to 20s
        if results_piac_spia_dict[f"0-{end_time}s"]['p_value'] < 0.05: 
            observed_diff.append(results_piac_spia_dict[f"0-{end_time}s"]['observed_diff'])
        else:
            observed_diff.append(-2)
        ci.append(results_piac_spia_dict[f"0-{end_time}s"]['ci'])
        
    ci = [c[1]-c[0] for c in ci]              
    axs[0].errorbar(range(1,12), observed_diff, yerr = ci, fmt = 'o', capsize = 8,
                  elinewidth = 2)

    ci, observed_diff = [], []
    for end_time in range(1, 12):  # 1s to 20s
        if results_neu_piac_dict[f"0-{end_time}s"]['p_value'] < 0.05: 
            observed_diff.append(results_neu_piac_dict[f"0-{end_time}s"]['observed_diff'])
        else:
            observed_diff.append(-2)

        ci.append(results_piac_spia_dict[f"0-{end_time}s"]['ci'])
    ci = [c[1]-c[0] for c in ci]              
    axs[0].errorbar(range(1,12), observed_diff, yerr = ci, fmt = 'd', capsize = 8,
                  elinewidth = 2)

    ci, observed_diff = [], []
    for end_time in range(1, 12):  # 1s to 20s
        if results_neu_spia_dict[f"0-{end_time}s"]['p_value'] < 0.05: 
            observed_diff.append(results_neu_spia_dict[f"0-{end_time}s"]['observed_diff'])
        else:
            observed_diff.append(-2)
        ci.append(results_piac_spia_dict[f"0-{end_time}s"]['ci'])
        
    ci = [c[1]-c[0] for c in ci]              
    axs[0].errorbar(range(1,12), observed_diff, yerr = ci, fmt = 'x', capsize = 8,
                  elinewidth = 2, color = 'purple')

    axs[0].set_ylim([-0.2E-6, 1.2E-6])

    yticks = [f"{0}-{end_time}s" for end_time in range(1, 12)]
    axs[0].set_xticks(range(1,12),labels = yticks, rotation = -45)
    axs[0].set_ylabel('GFP\nobserved difference')
    axs[0].set_title('Sliding window')

    #% Fixed average
    results_piac_spia_dict = {}
    results_neu_piac_dict = {}
    results_neu_spia_dict = {}
    for end_time in range(1, 13):  # 1s to 20s
        print(end_time)
        ix_times = (times > end_time-1) & (times < end_time)
        
        
        condition_neu = np.stack(gfp['neutro'])[:, ix_times]
        condition_piac = np.stack(gfp['piacevole'])[:, ix_times]
        condition_spia = np.stack(gfp['spiacevole'])[:, ix_times]
        
        # Perform permutation test for 'Pleasant - Unpleasant' and store results in dictionary
        observed_diff, p_value, (lower_bound, upper_bound) = permutation_test(condition_piac, condition_spia)
        results_piac_spia_dict[f"0-{end_time}s"] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'ci': (lower_bound, upper_bound)
        }
        # Perform permutation test for 'Pleasant - Unpleasant' and store results in dictionary
        observed_diff, p_value, (lower_bound, upper_bound) = permutation_test(condition_neu, condition_piac)
        results_neu_piac_dict[f"0-{end_time}s"] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'ci': (lower_bound, upper_bound)
        }
        # Perform permutation test for 'Pleasant - Unpleasant' and store results in dictionary
        observed_diff, p_value, (lower_bound, upper_bound) = permutation_test(condition_neu, condition_spia)
        results_neu_spia_dict[f"0-{end_time}s"] = {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'ci': (lower_bound, upper_bound)
        }

    # Plot
    ci, observed_diff = [], []
    for end_time in range(1, 12):  # 1s to 20s
        if results_piac_spia_dict[f"0-{end_time}s"]['p_value'] < 0.05: 
            observed_diff.append(results_piac_spia_dict[f"0-{end_time}s"]['observed_diff'])
        else:
            observed_diff.append(-2)
        ci.append(results_piac_spia_dict[f"0-{end_time}s"]['ci'])
    ci = [c[1]-c[0] for c in ci]              
    axs[1].errorbar(range(1,12), observed_diff, yerr = ci, fmt = 'o', capsize = 8,
                  elinewidth = 2)
    ci, observed_diff = [], []
    for end_time in range(1, 12):  # 1s to 20s
        if results_neu_piac_dict[f"0-{end_time}s"]['p_value'] < 0.05: 
            observed_diff.append(results_neu_piac_dict[f"0-{end_time}s"]['observed_diff'])
        else:
            observed_diff.append(-2)
        ci.append(results_piac_spia_dict[f"0-{end_time}s"]['ci'])
    ci = [c[1]-c[0] for c in ci]              
    axs[1].errorbar(range(1,12), observed_diff, yerr = ci, fmt = 'd', capsize = 8,
                  elinewidth = 2)
    ci, observed_diff = [], []
    for end_time in range(1, 12):  # 1s to 20s
        if results_neu_spia_dict[f"0-{end_time}s"]['p_value'] < 0.05: 
            observed_diff.append(results_neu_spia_dict[f"0-{end_time}s"]['observed_diff'])
        else:
            observed_diff.append(-2)
        ci.append(results_piac_spia_dict[f"0-{end_time}s"]['ci'])
    ci = [c[1]-c[0] for c in ci]              
    axs[1].errorbar(range(1,12), observed_diff, yerr = ci, fmt = 'x', capsize = 8,
                  elinewidth = 2, color = 'purple')
    axs[1].set_ylim([-0.2E-6, 1.2E-6])
    yticks = [f"{end_time-1}-{end_time}s" for end_time in range(1, 12)]
    axs[1].set_xticks(range(1,12),labels = yticks, rotation = -45)
    axs[1].set_ylabel('GFP\nobserved difference')
    axs[1].set_title('Fixed window')

    figs1.tight_layout()
    figs1.savefig('Figure 7.svg', format = 'svg')
    
    return fig, figs1


def plot_figure_8(all_participant_data):
    roi_ix = {'Frontal':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          'Central':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
          'Temporal':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          'Parietal':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
          'Occipital':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]}
    rois = roi_ix.keys()
    #['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    TFR_power = {
        'neutro': [participant_data[f'TFR power']['neutro'] for participant_data in all_participant_data.values()],
        'piacevole': [participant_data[f'TFR power']['piacevole'] for participant_data in all_participant_data.values()],
        'spiacevole': [participant_data[f'TFR power']['spiacevole'] for participant_data in all_participant_data.values()]
    }
    
    TFR_power = {
        'neutro': np.stack(TFR_power['neutro']).mean(axis=0),
        'piacevole': np.stack(TFR_power['piacevole']).mean(axis=0),
        'spiacevole': np.stack(TFR_power['spiacevole']).mean(axis=0)
    }

    # Filter channels for occipital and frontal regions
    occipital_ix = np.array(roi_ix['Occipital'], dtype=bool)
    frontal_ix = np.array(roi_ix['Frontal'], dtype=bool)
    temporal_ix = np.array(roi_ix['Temporal'], dtype=bool)
    central_ix = np.array(roi_ix['Central'], dtype=bool)
    parietal_ix = np.array(roi_ix['Parietal'], dtype=bool)
    
    freqs = all_participant_data['S01']['TFR freqs']['neutro']
    times = np.linspace(-1, 12, 4438)
    time_mask = (times >= 0) & (times <= 6)
    vmax = [8e-9, 8e-9, 8e-9, 8e-9, 8e-9, 8e-9]
    
    fig, axs = plt.subplots(3, 6, figsize=(18, 12), sharex=True, sharey=True)
    
    conditions = ['neutro', 'piacevole', 'spiacevole']
    titles = ['Neutral odor', 'Pleasant odor', 'Unpleasant odor']
    regions = [None, frontal_ix, central_ix, temporal_ix, parietal_ix, occipital_ix]
    region_titles = ['All channels', 'Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    ims = []
    # Create plots
    for row, condition in enumerate(conditions):
        for col, region in enumerate(regions):


            if region is None:
                data = TFR_power[condition].mean(axis=0)
            else:
                data = TFR_power[condition][region].mean(axis=0)
            
            im = axs[row, col].pcolormesh(times, freqs, data, shading='auto', vmin=0, vmax=vmax[col])
            if row == 0:
                ims.append(im)
                
            axs[row, col].set_ylim([3, 30])
            axs[row, col].set_xlim([-1, 12])
            if row == 0:
                if col >= 1:
                    axs[row, col].set_title(region_titles[col] + ' ROI',fontsize = 12)
                else:
                    axs[row, col].set_title('All channels',fontsize = 12)

            if col == 0:
                axs[row, col].set_ylabel(f'{titles[row]}\n\n\nFrequency [Hz]',fontsize = 12)
            if row == 2:
                axs[row, col].set_xlabel('Time [s]',fontsize = 12)
            axs[row, col].axvline(0, linestyle='--', linewidth=2, alpha=0.9, color='grey')
            axs[row, col].axvline(6, linestyle='--', linewidth=2, alpha=0.9, color='grey')
            axs[row, col].tick_params(axis='both', which='major', labelsize=12)
            
    # Add shared colorbars below each column
    for col, region in enumerate(regions):
        cbar_ax = fig.add_axes([0.065 + col * 0.16, 0.08, 0.1, 0.02])  # Adjust the position and size as needed
        cbar = fig.colorbar(ims[col], cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Power [V²]')
    
    fig.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space for colorbars
    fig.savefig('Figure_8.png', dpi = 300)
    fig.savefig('Figure_8.svg')
        
    time_mask = (times >= 0) & (times <= 6)
    TFR_power = {
        'neutro': [participant_data[f'TFR power']['neutro'] for participant_data in all_participant_data.values()],
        'piacevole': [participant_data[f'TFR power']['piacevole'] for participant_data in all_participant_data.values()],
        'spiacevole': [participant_data[f'TFR power']['spiacevole'] for participant_data in all_participant_data.values()]
    }
    
    TFR_power = {
        'neutro': np.stack(TFR_power['neutro']),
        'piacevole': np.stack(TFR_power['piacevole']),
        'spiacevole': np.stack(TFR_power['spiacevole'])
    }
    for roi in rois:
        roi_mask = np.array(roi_ix[roi], dtype=bool)
        for band, (fmin, fmax) in bands.items():
            # Select frequencies within the band
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            # Average TFR over time, frequency, and ROI
            neutral = TFR_power['neutro'][:,roi_mask,:,:][:,:,freq_mask,:][:,:,:,time_mask].mean(axis=(1,2))
            pleasant = TFR_power['piacevole'][:,roi_mask,:,:][:,:,freq_mask,:][:,:,:,time_mask].mean(axis=(1,2))
            unpleasant = TFR_power['spiacevole'][:,roi_mask,:,:][:,:,freq_mask,:][:,:,:,time_mask].mean(axis=(1,2))

            # Perform permutation test
            observed_diff, p_value_np, (lower_bound, upper_bound) = permutation_test(neutral, pleasant)
            observed_diff, p_value_ns, (lower_bound, upper_bound) = permutation_test(neutral, unpleasant)
            observed_diff, p_value_ps, (lower_bound, upper_bound) = permutation_test(pleasant, unpleasant)

            # Print results
            print(f"ROI: {roi}, Band: {band}")
            print(f"  Neutral vs Pleasant: p-value={p_value_np}")
            print(f"  Neutral vs Unpleasant: p-value={p_value_ns}")
            print(f"  Pleasant vs Unpleasant: p-value={p_value_ps}")
            
    return fig

# Sample function to plot figure 8 with boxplots
def plot_figure_8_with_boxplots(all_participant_data):
    from scipy.stats import friedmanchisquare, ttest_rel
    from statsmodels.stats.multitest import multipletests
    import itertools
    from statannot import add_stat_annotation

    roi_ix = {
        'All channels': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool),
        'Frontal': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool),
        'Central': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=bool),
        'Temporal': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool),
        'Parietal': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=bool),
        'Occipital': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    }

    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }

    TFR_power = {
        'neutro': np.stack([participant_data['TFR power']['neutro'] for participant_data in all_participant_data.values()]),
        'piacevole': np.stack([participant_data['TFR power']['piacevole'] for participant_data in all_participant_data.values()]),
        'spiacevole': np.stack([participant_data['TFR power']['spiacevole'] for participant_data in all_participant_data.values()])
    }

    # Prepare the data for boxplots
    data = {
        'Band': [],
        'ROI': [],
        'Window': [],
        'Condition': [],
        'Power': []
    }

    freqs = np.linspace(1, 30, 20)  # Example frequency range
    times = np.linspace(0, 12, 4438)  # Example time range
    windows = [(0, 12), (0, 6), (6, 12)]

    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        for roi, roi_mask in roi_ix.items():
            for window in windows:
                time_mask = (times >= window[0]) & (times < window[1])
                for condition, power_data in TFR_power.items():
                    avg_power = power_data[:, roi_mask, :, :][:, :, freq_mask, :][:, :, :, time_mask].mean(axis=(1, 2, 3))
                    for power_value in avg_power:
                        data['Band'].append(band)
                        data['ROI'].append(roi)
                        data['Window'].append(f'{window[0]}-{window[1]}')
                        data['Condition'].append(condition.capitalize())
                        data['Power'].append(power_value)

    df = pd.DataFrame(data)

    # Perform statistical tests for each Band, ROI, and window
    conditions = ['Neutro', 'Piacevole', 'Spiacevole']
    friedman_results = {}
    post_hoc_results = {}

    for band in bands.keys():
        friedman_results[band] = {}
        post_hoc_results[band] = {}
        for roi in roi_ix.keys():
            # print(roi)
            friedman_results[band][roi] = {}
            post_hoc_results[band][roi] = {}
            for window in windows:
                wname = f"{window[0]}-{window[1]}"
                window_data = [df[(df['Band'] == band) & (df['Condition'] == cond) & (df['Window'] == wname) & (df['ROI'] == roi)]['Power'].values for cond in conditions]
                stat, p_value = friedmanchisquare(*window_data)
                friedman_results[band][roi][wname] = {'stat': stat, 'p_value': p_value}

                if p_value < 0.05:
                    # pdb.set_trace()
                    print(band, roi, window)
                    print(stat, p_value)
                    pairwise_comparisons = list(itertools.combinations(conditions, 2))
                    valid_pairs = []
                    valid_p_values = []

                    for cond1, cond2 in pairwise_comparisons:
                        data1 = df[(df['Band'] == band) & (df['Condition'] == cond1) & (df['Window'] == wname) & (df['ROI'] == roi)]['Power'].values
                        data2 = df[(df['Band'] == band) & (df['Condition'] == cond2) & (df['Window'] == wname) & (df['ROI'] == roi)]['Power'].values

                        if len(data1) > 0 and len(data2) > 0:
                            _, p_val = ttest_rel(data1, data2)
                            valid_pairs.append((cond1, cond2))
                            valid_p_values.append(p_val)

                    # corrected_p_values = multipletests(valid_p_values, method='bonferroni')[1]
                    post_hoc_results[band][roi][wname] = dict(zip(valid_pairs, valid_p_values))
                    print(post_hoc_results[band][roi][wname])
    # pdb.set_trace()       
    # Create a 4x5 grid of boxplots
    fig, axs = plt.subplots(4, 6, figsize=(20, 12), sharex=True)
    rois = list(roi_ix.keys())
    bands_list = list(bands.keys())

    for row, band in enumerate(bands_list):
        for col, roi in enumerate(rois):
            sns.boxplot(
                data=df[(df['Band'] == band) & (df['ROI'] == roi)],
                x='Window', y='Power', hue='Condition',
                ax=axs[row, col], palette={'Neutro': 'blue', 'Piacevole': 'green', 'Spiacevole': 'red'},
                showfliers = False
            )
            # Set titles and labels
            if row == 0:
                axs[row, col].set_title(f'{roi} ROI', fontsize = 12)
            if col == 0:
                axs[row, col].set_ylabel(f'Power [V²]', fontsize = 12)
            else:
                axs[row, col].set_ylabel('')
            if row == 3:
                axs[row, col].set_xlabel('Time window [s]', fontsize = 12)
            else:
                axs[row, col].set_xlabel('')
            axs[row, col].legend_.remove()
        
            if col > 0:
                axs[row, col].set_yticks([])
            if row < 3:
                axs[row, col].set_xticks([])


            #     axs[row, col].set_ylim([0, 0.5E-8])
            # if row == 2 or row == 3:
            axs[row, col].set_ylim([0, 1E-8])
                
            # # Add pairwise statistical annotations for each window if Friedman test is significant
            # for window in windows:
            #     wname = f"{window[0]}-{window[1]}"
            #     if friedman_results[band][roi][wname]['p_value'] < 0.05:
            #         pairs = [(cond1, cond2) for cond1, cond2 in post_hoc_results[band][roi][wname].keys()]
            #         p_values = [post_hoc_results[band][roi][wname][pair] for pair in pairs]
            #         add_stat_annotation(
            #             axs[row, col],
            #             data=df[(df['Band'] == band) & (df['ROI'] == roi)],
            #             x='Window', y='Power',
            #             box_pairs=pairs,
            #             perform_stat_test=False,
            #             pvalues=p_values,
            #             text_format='star',
            #             loc='inside',
            #             verbose=0
            #         )
                    
    # Add a single legend
    legend_labels = {'Neutro': 'Neutral odor', 'Piacevole': 'Pleasant odor', 'Spiacevole': 'Unpleasant odor'}
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, [legend_labels[label] for label in labels], title='Condition')
    fig.tight_layout()
    
    fig.savefig('Figure 9.svg')
    plt.show()
    pdb.set_trace()


#%%
np.random.seed(42)
perm_tests = 0
# Directory handling
code_dir = os.getcwd()
main_dir = code_dir.replace('\Code', '')
data_dir = main_dir + "\Data"
res_dir = main_dir + "\Results"

plt.close('all')
all_participant_data = load_and_stack_participant_data(res_dir)
fig2, fig4 = plot_figure_2_4(all_participant_data)
fig5, fig6 = plot_figure_5_6(all_participant_data) 

fig8 = plot_figure_8(all_participant_data)
fig8 = plot_figure_8_with_boxplots(all_participant_data)

pdb.set_trace()









