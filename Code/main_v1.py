# main_combined.py

# Imports
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
import mne
from mne import Epochs, set_eeg_reference
from mne.io import read_raw_edf
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from mne_icalabel import label_components
from contextlib import contextmanager
import pdb

# Configurations and Directory Setup
code_dir = os.getcwd()
main_dir = code_dir.replace('\\Code', '')
data_dir = os.path.join(main_dir, "Data")
res_dir = os.path.join(main_dir, "Results")

@contextmanager
def suppress_output():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout = old_stdout

def filter_events(events):
    """Filter consecutive identical events."""
    filtered_events = []
    i = 0
    while i < len(events):
        if i < len(events) - 1 and np.array_equal(events[i], events[i + 1]):
            i += 1
        filtered_events.append(events[i])
        i += 1
    return np.array(filtered_events)

def expand_dict(dictt, types, fadd):
    """Expand dictionary to initialize necessary EEG feature keys with multiple PSD windows."""
    windows = ['0-12s', '0-6s', '6-12s']
    for k in dictt.keys():
        for t in types:
            dictt[k][t] = []
    for fe in fadd:
        for fr in ['delta', 'theta', 'alpha', 'beta']:
            for window in windows:
                dictt[f'{fe} ({window}): {fr}'] = {}
    for fe in fadd:
        for window in windows:
            dictt[f'{fe} ({window})'] = {}
    for window in windows:
        dictt[f'Frequency ({window})'] = {}
    return dictt



def permutation_test(condition1, condition2, n_permutations=20000, ci_level=0.95):
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


#%% Options
optex = {
    'seed': 42,
    'data_dir': data_dir,
    'montage': 'standard_1020',
    't_min_ep': -1,
    't_max_ep': 12,
    'iter_ica':15,
    'eeg_bands': {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)},
    'mapping': {
               'EEG AF7': 'eeg', 'EEG AF3': 'eeg', 'EEG Fp1': 'eeg', 'EEG Fp2': 'eeg',
               'EEG AF4': 'eeg', 'EEG AF8': 'eeg', 'EEG F7': 'eeg', 'EEG F5': 'eeg',
               'EEG F3': 'eeg', 'EEG F1': 'eeg', 'EEG F2': 'eeg', 'EEG F4': 'eeg',
               'EEG F6': 'eeg', 'EEG F8': 'eeg', 'EEG FT7': 'eeg', 'EEG FC5': 'eeg',
               'EEG FC3': 'eeg', 'EEG FC1': 'eeg', 'EEG FC2': 'eeg', 'EEG FC4': 'eeg',
               'EEG FC6': 'eeg', 'EEG FT8': 'eeg', 'EEG T3': 'eeg', 'EEG C5': 'eeg',
               'EEG C3': 'eeg', 'EEG C1': 'eeg', 'EEG C2': 'eeg', 'EEG C4': 'eeg',
               'EEG C6': 'eeg', 'EEG T4': 'eeg', 'EEG TP7': 'eeg', 'EEG CP5': 'eeg',
               'EEG CP3': 'eeg', 'EEG CP1': 'eeg', 'EEG CP2': 'eeg', 'EEG CP4': 'eeg',
               'EEG CP6': 'eeg', 'EEG TP8': 'eeg', 'EEG T5': 'eeg', 'EEG P5': 'eeg',
               'EEG P3': 'eeg', 'EEG P1': 'eeg', 'EEG P2': 'eeg', 'EEG P4': 'eeg',
               'EEG P6': 'eeg', 'EEG T6': 'eeg', 'EEG Fpz': 'eeg', 'EEG PO7': 'eeg',
               'EEG PO3': 'eeg', 'EEG O1': 'eeg', 'EEG O2': 'eeg', 'EEG PO4': 'eeg',
               'EEG PO8': 'eeg', 'EEG Oz': 'eeg', 'EEG AFz': 'eeg', 'EEG Fz': 'eeg',
               'EEG FCz': 'eeg', 'EEG Cz': 'eeg', 'EEG CPz': 'eeg', 'EEG Pz': 'eeg',
               'EEG POz': 'eeg', 'ECG': 'ecg', 'EOG': 'eog', 'MK': 'misc'
               }
}


#%% Execution
# EEG Types for Different Stimuli
eeg_types = ['piacevole', 'neutro', 'spiacevole']
band_f = ['PSD']
# Processing and Analysis Code
os.chdir(data_dir)
participant_folders = os.listdir(data_dir)

# Process each participant folder
for participant in participant_folders:
    participant_path = os.path.join(data_dir, participant)
    
    if os.path.isdir(participant_path):
        # Save participant's processed data
        with open(os.path.join(res_dir, f'{participant}_features.pkl'), 'rb') as file:
            dict_features = pickle.load(file)
            
        print(colored(f'Processing participant: {participant}', 'red'))
        
        # # Initialize dictionary for storing features
        # dict_features = expand_dict({'GFP mean':{},
        #                              'GFP std':{},
        #                              'GFP peak':{},
        #                              'GFP latency':{},
        #                              'GFP times':{},
        #                              'TFR power':{},
        #                              'TFR freqs':{},
        #                              'Evoked' : {}, 
        #                              }, eeg_types, band_f)
        # Find and process each EEG file for the specified stimuli in the participant's folder
        for typ in eeg_types:
            # Construct the expected EEG filename for the stimulus
            eeg_file = f"{participant.replace('_','^')}_{typ}.edf"
            os.chdir(participant_path)
            # Only process if the file exists
            print(colored(f'Processing EEG for stimulus: {typ}', 'blue'))
            
            # Load EEG data
            raw = read_raw_edf(eeg_file, preload=True, verbose=False)
            raw.set_channel_types(optex.get('mapping', {}), verbose=False)
            rename_dict = {name: name.replace('EEG ', '') for name in optex['mapping'] if optex['mapping'][name] == 'eeg'}
            raw.rename_channels(rename_dict)
            raw.set_montage(optex['montage'], match_case=True, on_missing='raise', verbose=False)
            raw.pick_types(meg=False, eog=False, resp=False, emg=False, misc=False, eeg=True, ecg=False, verbose=False)
            raw, _ = set_eeg_reference(raw, copy=False, ref_channels='average', ch_type='eeg', verbose=False)
            raw.notch_filter(freqs=[50])
            raw.filter(l_freq=1.5, h_freq=30)
            
            # Apply ICA
            ica = ICA(n_components=15, max_iter=optex['iter_ica'],
                      random_state=optex['seed'], method='infomax', fit_params=dict(extended=True))
            ica.fit(raw)
            ic_labels = label_components(raw, ica, method='iclabel')
            exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label != "brain"]
            ica.apply(raw, exclude=exclude_idx)
            
            # Event-related processing
            events, event_id = mne.events_from_annotations(raw)
            events = filter_events(events)
            epochs = Epochs(raw, events, event_id,
                            tmin=optex['t_min_ep'], tmax=optex['t_max_ep'], baseline=(-1, 0), preload=True)
            epochs.drop_bad()
            # dict_features['Evoked'][typ] = epochs.average().get_data()

            
            # # Global Field Power (GFP) Analysis
            # gfp = np.std(epochs.get_data(picks='eeg'), axis=1).transpose()
            # gfp_mean = np.mean(gfp, axis=1)
            # gfp_std = np.std(gfp, axis=1)
            # gfp_peak = gfp_mean[np.argmax(gfp_mean)]
            # gfp_latency = epochs.times[np.argmax(gfp_mean)]
            # dict_features['GFP mean'][typ] = gfp_mean
            # dict_features['GFP std'][typ] = gfp_std
            # dict_features['GFP peak'][typ] = gfp_peak
            # dict_features['GFP latency'][typ] = gfp_latency 
            # dict_features['GFP times'][typ] = epochs.times
            
            # # Define different time windows
            # windows = {
            #     '0-12s': (0, 12),
            #     '0-6s': (0, 6),
            #     '6-12s': (6, 12)
            # }
            
            # # Compute PSD for each window and each frequency band
            # for window_label, (tmin, tmax) in windows.items():
            #     window_epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
            
            #     # Calculate PSD for each frequency band in the specified window
            #     psd, freq = window_epochs.compute_psd().get_data(return_freqs=True)
            #     dict_features[f'PSD ({window_label})'][typ] = psd
            #     dict_features[f'Frequency ({window_label})'][typ] = freq
            #     for band, (start, stop) in optex['eeg_bands'].items():
            #         indices = np.where((freq >= start) & (freq < stop))[0]
            #         psd_band = np.mean(np.trapz(psd[:, :, indices], axis=-1))
            #         dict_features[f'PSD ({window_label}): {band}'][typ] = psd_band
        
            
            # Time-Frequency Representation (TFR) using Morlet Wavelets
            pdb.set_trace()
            freqs = np.logspace(*np.log10([3, 50]), num=20)
            n_cycles = freqs / 2.0  # Vary the number of cycles with frequency
            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, decim=3)
            dict_features['TFR power'][typ] = power.data
            dict_features['TFR freqs'][typ] = freqs
            
        # Save participant's processed data
        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, f'{participant}_features.pkl'), 'wb') as file:
            pickle.dump(dict_features, file)
            
        # pdb.set_trace()



