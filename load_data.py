"""
Modules used to import a series of datasets from a selected working directory.
There are four datasets that can be imported from this module.
    1: WAY-EEG-GAl
    2: Motor Imagery
    3: Motor Execution
    4: Spinal Cord Injury
"""
import mne
from mne.channels import read_montage
from mne import create_info
from mne.io import RawArray
import pandas as pd
import numpy as np

from mne.channels import read_montage
from mne import create_info
from mne.io import RawArray

from mne.preprocessing import create_eog_epochs, ICA
from mne import Epochs, pick_types

trial_start_end = dict(elbow_Flex=1536, elbow_Flex_TEnd=101394,
                       elbow_Extend=1537, elbow_Extend_TEnd=101395,
                       supination=1538, supination_TEnd=101396,
                       pronation=1539, pronation_TEnd=101397,
                       hand_Close=1540, hand_Close_TEnd=101398,
                       hand_Open=1541, hand_Open_TEnd=101399,
                       rest=1542, rest_TEnd=101400)
chosen_events = dict(elbow_Flex=1536, elbow_Extend=1537, supination=1538, pronation=1539, hand_Close=1540,
                     hand_Open=1541, rest=1542)

def read_execution_imagination(file_location, paradigm = 'Healthy'):
    """
    Create the Raw Data Array for data coming from the Upper Limb Movement Dataset.
    Args:
        file_location: A string value representing individual datasets.
        paradigm: A string value representing Healthy or Spinal Cord Injury
            Health: Subjects from the upper limb dataset
            SCI_Offline: Subjects from the Spinal Cord Injury offline dataset
            SCI_Online: Subjects from the Spinal Cord Injury online dataset
    Returns:
        raw: A Raw Array for further manipulation.
    """
    raw = mne.io.read_raw_edf(file_location)
    eog_names = ["eog-r", "eog-m", "eog-l"]
    glove_sensors = ["thumb_near", "thumb_far", "thumb_index", "index_near", "index_far", "index_middle",
                 "middle_near", "middle_far", "middle_ring", "ring_near", "ring_far", "ring_little", "litte_near",
                     "litte_far", "thumb_palm", "wrist_bend", "roll", "pitch", "gesture"]
    stim_channel = ["STIM"]

    if paradigm == 'Healthy':
        print('Loading a dataset with a ' + paradigm + ' population.')
        eeg_names = ["F3", "F1", "Fz", "F2", "F4", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h",
                    "FFC6h", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FTT7h", "FCC5h",
                    "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h", "FTT8h", "C5", "C3", "C1", "Cz",
                    "C2", "C4", "C6", "TTP7h", "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
                    "TTP8h", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h",
                    "CPP1h", "CPP2h", "CPP4h", "CPP6h", "P3", "P1", "Pz", "P2", "P4", "PPO1h",
                    "PPO2h"]

        exo_sensors = ["handPosX", "handPosY", "handPosZ", "elbowPosX", "elbowPosY", "elbowPosZ", "ShoulderAdd",
                               "ShoulderFlex", "ShoulderRot", "Elbow", "ProSupination", "Wrist", "GripPressure"]
        stim_channel = ["STIM"]
        channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['misc']*len(glove_sensors) + ['stim']*len(exo_sensors) + ['stim']
        montage = read_montage('standard_1005', eeg_names)
        channel_names = eeg_names + eog_names + glove_sensors + exo_sensors + stim_channel
        # Create all data into a RawArray
        info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)
        k = raw.get_data()
        # Seperate k into pandas dataframes of data
        eeg_data = k[0:61, :]
        eog_data = k[61:64, :]
        glove_data = k[64:83, :]
        exo_data = k[83:96, :]
        stim_data = k[96:97, :][0]

        teeg = pd.DataFrame(eeg_data.T, columns=eeg_names)
        teog = pd.DataFrame(eog_data.T, columns=eog_names)
        tglove = pd.DataFrame(glove_data.T, columns=glove_sensors)
        texo = pd.DataFrame(exo_data.T, columns=exo_sensors)
        tstim = pd.DataFrame(stim_data.T, columns=stim_channel)
        t_df = pd.concat([teeg.reset_index(drop=True), teog, tglove, texo, tstim], axis=1)
        j = t_df.dropna()
        raw = RawArray(j.T, info, verbose=False).load_data()

    if paradigm == 'SCI_Offline':
        print('Loading the ' + paradigm + ' dataset')
        eeg_names = ["AFz", "F3", "F1", "Fz", "F2", "F4",
                    "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
                    "FC5", "FCC6h", "C5", "C3", "C1", "Cz",
                    "C2", "C4", "C6", "CCP5h", "CCP3h", "CCP1h",
                    "CCP2h", "CCP4h", "CPP2h", "CPP4h", "CPP6h","P5",
                    "P3", "P1", "Pz", "P2", "P4", "P6",
                    "PPO1h", "PPO2h", "POz", "FC3", "FC1", "FCz",
                    "FC2", "FC4", "FC6", "FCC5h", "FCC3h", "FCC1h",
                    "FCC2h", "FCC4h", "CCP6h", "CP5", "CP3", "CP1",
                    "CPz", "CP2", "CP4", "CP6", "CPP5h", "CPP3h","CPP1h"]

        channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['stim']
        montage = read_montage('standard_1005', eeg_names)
        channel_names = eeg_names + eog_names + stim_channel
        info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)
        k = raw.get_data()
        # Seperate k into pandas dataframes of data
        eeg_data = k[0:61, :]
        eog_data = k[61:64, :]
        stim_data = k[96:97, :][0]

        teeg = pd.DataFrame(eeg_data.T, columns=eeg_names)
        teog = pd.DataFrame(eog_data.T, columns=eog_names)
        tstim = pd.DataFrame(stim_data.T, columns=stim_channel)
        t_df = pd.concat([teeg.reset_index(drop=True), teog, tstim], axis=1)
        j = t_df.dropna()
        raw = RawArray(j.T, info, verbose=False).load_data()

    if paradigm == 'SCI_Online':
        print('Loading the ' + paradigm + ' dataset')
        eeg_names = ["AFz", "F3", "F1", "Fz", "F2", "F4",
                    "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h","FFC6h",
                    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4",
                    "FC6", "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h",
                    "FCC6h", "C5", "C3", "C1", "Cz", "C2",
                    "C4", "C6", "CCP5h", "CCP3h", "CCP1h", "CCP2h",
                    "CCP4h", "CCP6h", "CP5", "CP3", "CP1", "CPz",
                    "CP2", "CP4", "CP6", "CPP5h", "CPP3h","CPP1h",
                    "CPP2h", "CPP4h", "CPP6h","P5", "P3", "P1",
                    "Pz", "P2", "P4", "P6", "PPO1h", "PPO2h", "POz"]

        channel_type = ['eeg']*len(eeg_names) + ['eog']*len(eog_names) + ['misc']*len(glove_sensors) + ['stim']*len(exo_sensors) + ['stim']
        montage = read_montage('standard_1005', eeg_names)
        channel_names = eeg_names + eog_names + glove_sensors + stim_channel
        info = create_info(channel_names, sfreq=512.0, ch_types=channel_type, montage=montage)
        k = raw.get_data()
        # Seperate k into pandas dataframes of data
        eeg_data = k[0:61, :]
        eog_data = k[61:64, :]
        glove_data = k[64:83, :]
        stim_data = k[83:84, :][0]

        teeg = pd.DataFrame(eeg_data.T, columns=eeg_names)
        teog = pd.DataFrame(eog_data.T, columns=eog_names)
        tglove = pd.DataFrame(glove_data.T, columns=glove_sensors)
        tstim = pd.DataFrame(stim_data.T, columns=stim_channel)
        t_df = pd.concat([teeg.reset_index(drop=True), teog, tglove, texo, tstim], axis=1)
        j = t_df.dropna()
        raw = RawArray(j.T, info, verbose=False).load_data()
    return raw

def create_raw_data(dataset, subject, trial, wd=r'D:/Datasets/RealTime/BCIDash/datasets/'):
    """
    Create a Raw Data Array for one subjecs and one trial for all datasets.
    Args:
        dataset: A string value reperesenting each dataset
                1: WAY-EEG-GAl
                2: Motor Imagery
                3: Motor Execution
                4: Spinal Cord Injury
        subject: An integer value for each subject.
        trial: An integer value for each trial.
        wd: By default the working directory is set. The working directory can be changes manually
    Returns:
        raw: A raw array of eeg, sensors and stim.
    """
    if dataset == 'WAY-EEG-GAL':
        print(dataset + ' Selected')
        print('Attempting to load data from subject ' + str(subject) + ' and trial ' + str(trial))

    if dataset == 'Motor Imagination':
        print(dataset + ' Selected')
        print('Attempting to load data from subject ' + str(subject) + ' and trial ' + str(trial))
        file_location = wd +r'S%02d_MI/motorimagination_subject%d_run%d.gdf' % (subject, subject, trial)
        raw = read_execution_imagination(file_location, 'Healthy')
        print('The ' + dataset + ' dataset has been loaded.')

    if dataset == 'Motor Execution':
        print(dataset + ' Selected')
        print('Attempting to load data from subject ' + str(subject) + ' and trial ' + str(trial))
        file_location = wd +r'S%02d_ME/motorexecution_subject%d_run%d.gdf' % (subject, subject, trial)
        raw = read_execution_imagination(file_location, 'Healthy')
        print('The ' + dataset + ' dataset has been loaded.')

    if dataset == 'Spinal Cord Injury Offline':
        print(dataset + ' Selected')
        print('Attempting to load data from subject ' + str(subject) + ' and trial ' + str(trial))
        file_location = wd + r'P%02d/P%02d Run %d.gdf' % (subject, subject, trial)
        raw = read_execution_imagination(file_location, 'SCI Offline')
        print('The ' + dataset + ' dataset has been loaded.')

    if dataset == 'Spinal Cord Injury Online Train':
        print(dataset + ' Selected')
        print('Attempting to load data from subject ' + str(subject) + ' and trial ' + str(trial))
        file_location = wd + r'P09 Online Session %d Train/P09 Run %d.gdf' % (subject, trial)
        raw = read_execution_imagination(file_location, 'SCI Online')
        print('The ' + dataset + ' dataset has been loaded.')

    if dataset == 'Spinal Cord Injury Online Test':
        print(dataset + ' Selected')
        print('Attempting to load data from subject ' + str(subject) + ' and trial ' + str(trial))
        file_location = wd + r'P09 Online Session %d Test/P09 Run %d.gdf' % (subject, trial)
        raw = read_execution_imagination(file_location, 'SCI Online')
        print('The ' + dataset + ' dataset has been loaded.')

    else:
        print('Not a known dataset. Check the dataset string.')

    return raw

def filter_with_ica(raw_data, event_id, thresh):
    """Use ICA to filter data for artifacts.

    Args:
        raw_data: Raw array of data
        events: Event Array
        event_id: A Dictionary of events
        thresh: A float that indicates a threshold

    Resturns:
        raw: ICA and bandpass filtered data
    """
    raw = raw_data.copy().crop(2, raw_data.times.max())
    #raw = raw_data.copy()
    raw.filter(3, 42, n_jobs=1, fir_design='firwin')
    # Run ICA
    method = 'fastica'
    # Choose other parameters
    n_components, decim, random_state = 25, 3, 42  # if float, select n_components by explained variance of PCA
    ica = ICA(n_components=n_components, method=method, random_state=random_state)
    ica.fit(raw, picks=pick_types(raw.info, eeg=True, misc=False, stim=False, eog=False), decim=decim)
    eog_epochs = create_eog_epochs(raw)  # get single EOG trials
    eog_inds, scores = ica.find_bads_eog(eog_epochs, threshold=thresh)  # find via correlation
    ica.exclude.extend(eog_inds)
    ica.apply(raw)
    return raw

def create_event_array_for_movement_onset(dataset, subject, trial, param5=False):
    """Create a function to label event of motor activation

    Args:
        subject: Integer representing thr subject
        trial: Integer representing the trial
        param5: show plot: True or False

    Returns:
        unique_events: An event array to be used for future imports
        raw_original: The original raw array
    """
    raw = create_raw_data(dataset, subject, trial, wd=r'D:/Datasets/RealTime/BCIDash/datasets/')
    #raw_original = raw.copy().crop(10, raw.times.max())
    #raw_filtered = raw_original.copy()
    raw_filtered = raw.copy()
    picks = pick_types(raw.info, eeg=False, stim=False, include=["Elbow", "ProSupination", "GripPressure"])
    raw_filtered.filter(None, 10, fir_design='firwin', picks=picks)

    # Pick the channels to investigate
    pick_elbow = pick_types(raw_filtered.info, eeg=False, stim=False, include=["Elbow"])
    pick_wrist = pick_types(raw_filtered.info, eeg=False, stim=False, include=["ProSupination"])
    pick_grip = pick_types(raw_filtered.info, eeg=False, stim=False, include=["GripPressure"])
    pick_hand_position_x = pick_types(raw_filtered.info, eeg=False, stim=False, include=["handPosX"])

    # Get the events
    tmin, tmax = 0, 3
    events = mne.find_events(raw_filtered.copy().pick_channels(["STIM"]))

    # Find the index of each event
    # Subtract 1 from the index to get the start time of each trial
    elbow_f_i = np.where(events[:, 2] == 1536)[0] - 1
    elbow_e_i = np.where(events[:, 2] == 1537)[0] - 1
    wrist_s_i = np.where(events[:, 2] == 1538)[0] - 1
    wrist_p_i = np.where(events[:, 2] == 1539)[0] - 1
    hand_c_i = np.where(events[:, 2] == 1540)[0] - 1
    hand_o_i = np.where(events[:, 2] == 1541)[0] - 1

    all_indexes = [elbow_f_i, elbow_e_i, wrist_s_i, wrist_p_i, hand_c_i, hand_o_i]

    # Get epoch data
    elbow_f_epochs = Epochs(raw_filtered.copy(), events, dict(elbow_Flex=1536), tmin, tmax, proj=True, picks=pick_elbow,
                            baseline=None, preload=True)
    elbow_e_epochs = Epochs(raw_filtered.copy(), events, dict(elbow_Extend=1537), tmin, tmax, proj=True,
                            picks=pick_elbow, baseline=None, preload=True)
    wrist_p_epochs = Epochs(raw_filtered.copy(), events, dict(pronation=1539), tmin, tmax, proj=True, picks=pick_wrist,
                            baseline=None, preload=True)
    wrist_s_epochs = Epochs(raw_filtered.copy(), events, dict(supination=1538), tmin, tmax, proj=True, picks=pick_wrist,
                            baseline=None, preload=True)
    hand_c_epochs = Epochs(raw_filtered.copy(), events, dict(hand_Close=1540), tmin, tmax, proj=True, picks=pick_grip,
                           baseline=None, preload=True)
    hand_o_epochs = Epochs(raw_filtered.copy(), events, dict(hand_Open=1541), tmin, tmax, proj=True, picks=pick_hand_position_x,
                           baseline=None, preload=True)

    all_epochs = [elbow_f_epochs, elbow_e_epochs, wrist_s_epochs, wrist_p_epochs, hand_c_epochs]

    elbow_f = elbow_f_epochs.get_data()
    elbow_e = elbow_e_epochs.get_data()
    wrist_p = wrist_p_epochs.get_data()
    wrist_s = wrist_s_epochs.get_data()
    hand_c = hand_c_epochs.get_data()
    hand_o = hand_o_epochs.get_data()

    all_data = [elbow_f, elbow_e, wrist_s, wrist_p, hand_c, hand_o]

    # Event time, 0, EventLabelNumber
    set_thresh = 0.20
    for i in range(len(all_data)):  # For all event types elbow flex etc.
        time_index = []
        start_event_index = events[np.where(events[:, 2] == list(chosen_events.values())[i]), 0]
        start_event_index = start_event_index[0].astype(int)

        for j in range(len(all_data[i])):  # For each instance of elbow flex
            first_point = all_data[i][j][0][0]
            last_point = all_data[i][j][0][-1]
            if first_point > last_point:  # first larger than last?
                this_range = first_point - last_point
                data_thresh = first_point - set_thresh * this_range
                for idx, h in enumerate(all_data[i][j][0]):
                    if h < data_thresh:
                        time_index.append(idx)  # Get time index for each instance of elbow flex
                        break
            else:
                this_range = last_point - first_point
                data_thresh = first_point + set_thresh * this_range
                for idx, h in enumerate(all_data[i][j][0]):
                    if h > data_thresh:
                        time_index.append(idx)  # Get time index for each instance of elbow flex
                        break

        stim_times = start_event_index + time_index
        zero_array = np.zeros(len(stim_times))
        stim_label_array = zero_array + 5000 + i
        # Create an array of the three arrays
        new_events = np.vstack([stim_times, zero_array, stim_label_array]).T
        events = np.vstack([events, new_events])

    # Combine arrays by odering the time index
    sorted_events_index = np.argsort(events[:, 0], axis=0)
    reorder_events = events[sorted_events_index]
    reorder_events = reorder_events.astype(int)
    unique_events = np.unique(reorder_events, axis=0)
    revised_events = dict(elbow_Flex=1536, flex=5000, elbow_Extend=1537, extend=5001,
                          supination=1538, sup=5002, pronation=1539, pro=5003,
                          hand_Close=1540, close=5004, hand_Open=1541, hopen=5005, rest=1542)
    if param5 is True:
        raw.plot(block=True, scalings='auto', events=events, event_id=revised_events)
    raw.add_events(events=unique_events, stim_channel="STIM")
    #return unique_events, raw_original
    return unique_events, raw
