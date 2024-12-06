from enum import Enum

class Label_seizure_normal(Enum):
    NORMAL = 0
    SEIZURE = 1
    
CONST_TUSZ_CLASS_NAME_TO_LABEL_SEIZURE_NORMAL = {
    "BCKG" : Label_seizure_normal.NORMAL,
    "SEIZ" : Label_seizure_normal.SEIZURE,
    "FNSZ" : Label_seizure_normal.SEIZURE,
    "GNSZ" : Label_seizure_normal.SEIZURE,
    "SPSZ" : Label_seizure_normal.SEIZURE,
    "CPSZ" : Label_seizure_normal.SEIZURE,
    "ABSZ" : Label_seizure_normal.SEIZURE,
    "TNSZ" : Label_seizure_normal.SEIZURE,
    "CNSZ" : Label_seizure_normal.SEIZURE,
    "TCSZ" : Label_seizure_normal.SEIZURE,
    "ATSZ" : Label_seizure_normal.SEIZURE,
    "MYSZ" : Label_seizure_normal.SEIZURE,
    "NESZ" : Label_seizure_normal.SEIZURE
}

ORDERED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG A1',
    'EEG A2',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']

MONTAGES_NAMES = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

ORDERED_CHANNELS_REDUCED = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']

MONTAGES_NAMES_REDUCED = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

MONTAGES_CHANNELS_IDX = [[ORDERED_CHANNELS.index("EEG " + channel) for channel in montage.split("-")] for montage in MONTAGES_NAMES]

MONTAGES_CHANNELS_IDX_REDUCED = [[ORDERED_CHANNELS_REDUCED.index("EEG " + channel) for channel in montage.split("-")] for montage in MONTAGES_NAMES_REDUCED]