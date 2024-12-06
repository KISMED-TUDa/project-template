import numpy as np
import pyedflib
from scipy.io import savemat
from scipy.signal import ShortTimeFFT, iirnotch, filtfilt, resample
from constants import *
import torch
from typing import List, Any, Tuple
import os
import pandas as pd

def getOrderedChannels(file_name:str, verbose:bool, labels:List[str|Any], channel_names:List[str]) -> List[int]:
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels

def getEDFsignals(edf:pyedflib.EdfReader) -> np.ndarray:
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals

def split_into_segments(data:np.ndarray, segment_len:int) -> np.ndarray:
    target_size = int(segment_len * np.ceil(data.shape[-1] / segment_len))
    data = np.pad(data, ((0, 0), (int(np.floor((target_size-data.shape[-1])/2)), int(np.ceil((target_size-data.shape[-1])/2))))) # Pad zeros right and left
    data = np.split(data, indices_or_sections=data.shape[1]//segment_len, axis=1) 
    data = np.stack(data, axis=0) # Shape : N_segments x N_channels x Segment_len
    return data

def apply_notch_filter(data:np.ndarray, fs:float, notch_args:dict[str, Any]):
    b, a = iirnotch(fs=fs, **notch_args)
    return filtfilt(b, a, data, axis=-1)

def preprocess_data(data:np.ndarray, fs:float, notch_args:dict[str, Any], resample_freq:float, segment_len:int) -> np.ndarray:
    data = apply_notch_filter(data, fs, notch_args) # Shape : N_channels x N_datapoints
    data = resample(data, num=int(data.shape[1]*resample_freq/fs), axis=-1) # Shape : N_channels x N_datapoints_new
    data = split_into_segments(data, segment_len) # Shape : N_segments x N_channels x Segment_len 
    data = (data-data.mean(axis=2, keepdims=True)) / (data.std(axis=2, keepdims=True) + 1e-30) # Standardize and avoid division through zero
    return data

def calculate_spectrogram(data:np.ndarray, STFT:ShortTimeFFT, max_n_parallel:int) -> torch.Tensor:
    spectograms = []
    for i in range(0, data.shape[0], max_n_parallel):                                 # Shape : N_segments x N_channels x N_bins x N_timepoints
        spectograms.append(np.abs(STFT.stft(data[i:i+max_n_parallel, ...], axis=-1))) # Need to split the data up into max 500 segments each for STFT otherwise it takes forever
    spectograms = np.concatenate(spectograms, axis=0)
    spectograms = spectograms.reshape(-1, spectograms.shape[2], spectograms.shape[3]) # Shape : N_segments * N_channels x N_bins x N_timepoints

    # Min-Maxing
    spectograms -= spectograms.min(axis=(1, 2), keepdims=True)
    spectograms /= (spectograms.max(axis=(1, 2), keepdims=True) + 1e-30) # Range : [0, 1] and avoid division through zero

    return torch.from_numpy(spectograms).unsqueeze(1).float()

def generate_label_TUSZ(label_path:str, N_segments:int, N_montages:int, fs:float, segment_len:int, montage_names:list[str]) -> torch.Tensor:
    df = pd.read_csv(label_path, comment='#')
    labels = torch.zeros((N_segments, N_montages), dtype=int) # Shape : N_segments x N_channels (Initialize labels with normal (0))
    segment_start_indices = np.arange(N_segments)*segment_len # Start indices of each segment # Shape : N_segments

    for i in range(len(df)):
        channel, onset, offset, label = df[["channel", "start_time", "stop_time", "label"]].iloc[i]
        label_int = CONST_TUSZ_CLASS_NAME_TO_LABEL_SEIZURE_NORMAL[label.upper()]
        if(label_int == Label_seizure_normal.NORMAL):
            continue
        label_start_idx = onset*fs # Start sample for the label
        label_end_idx = offset*fs # End sample for the label

        idx_channel = montage_names.index(channel.upper()) # idx for the channel that is currently labeled

        seizure_segments = np.logical_and(segment_start_indices > label_start_idx, segment_start_indices <= label_end_idx) # Set the segments that should be affected by the label to True
        transition_indices = np.argwhere(np.diff(seizure_segments))
        if(transition_indices.shape[0] > 1):
            transition_indices[1] = transition_indices[1]+1
        seizure_segments[transition_indices] = True # Mark transition segments also as True 
                                                    # Shape : N_segments (tensor with True at segments that should be labeled with the new label)
        labels[seizure_segments, idx_channel] = label_int.value # Set the label for the affected segments
    return labels.float()

def get_montages_from_edf(edf_path:str) -> Tuple[np.ndarray, float, List[str]]:
    if("03_tcp_ar_a" in edf_path.lower()): # A1 and A2 are not available, use a reduced set of electrodes/montages.
        channels = ORDERED_CHANNELS_REDUCED
        montages_idx = MONTAGES_CHANNELS_IDX_REDUCED
        montage_names = MONTAGES_NAMES_REDUCED
    else:
        channels = ORDERED_CHANNELS
        montages_idx = MONTAGES_CHANNELS_IDX
        montage_names = MONTAGES_NAMES
    f = pyedflib.EdfReader(edf_path)
    orderedChannels = getOrderedChannels(edf_path, False, f.getSignalLabels(), channels)

    signals = getEDFsignals(f) # Get unordered signals
    signals = np.array(signals[orderedChannels, :]) # Order them according to ORDERED_CHANNELS

    montages = np.stack([signals[i, :] - signals[j, :] for i, j in montages_idx], axis=0) # Calculate Montages
    fs = f.getSampleFrequency(0)

    f.close() # Close edf file as it isn't needed anymore
    return montages, fs, montage_names

def tusz_to_segment_mat(data_path : str, save_path : str, resample_freq : float, segment_len : int, spectrogram_args : dict, notch_args : dict) -> None:
    STFT = ShortTimeFFT(**spectrogram_args)
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if not name.lower().endswith(".edf"):
                continue
            
            edf_path = os.path.join(path, name)

            save_path_0 = os.path.join(save_path, name.lower().replace(".edf", "_0.mat"))
            if(os.path.exists(save_path_0)): # Skip segments that have already been calculated
                print("Skipped %s as it has already been processed"%edf_path)
                continue
            
            montages, fs, montage_names = get_montages_from_edf(edf_path)

            # Preprocessing
            montages = preprocess_data(montages, fs, notch_args, resample_freq, segment_len) # Shape : N_segments x N_montages x Segment_len
            
            # Create labels
            csv_path = edf_path.replace(".edf", ".csv")
            labels = generate_label_TUSZ(csv_path, montages.shape[0], montages.shape[1], resample_freq, segment_len, montage_names) # Shape : N_segments x N_montages

            # Calculate spectogram
            spectrograms = calculate_spectrogram(montages, STFT, max_n_parallel=500) # Shape : N_segments * N_channels x 1 x N_bins x N_timepoints
            spectrograms = spectrograms.reshape(montages.shape[0], montages.shape[1], spectrograms.shape[2], spectrograms.shape[3]) # Shape : N_segments x N_channels x N_bins x N_timepoints

            for i in range(spectrograms.shape[0]): # Save each segment in one mat file
                cur_save_path = os.path.join(save_path, name.lower().replace(".edf", "_%d.mat"%i))
                data_dict = {
                    "spectrogram": spectrograms[i, ...],
                    "label" : labels[i, ...],
                    "channels" : montage_names,
                    "fs" : resample_freq
                }
                savemat(cur_save_path, data_dict)