import sys
import qtmodern.styles
import qtmodern.windows
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import mne
from mne import Epochs, pick_types
from load_data import create_raw_data, create_event_array_for_movement_onset
import numpy as np
class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        #self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def generate_test_epochs(self, dataset, subject, trial):
        events, raw = create_event_array_for_movement_onset(dataset, subject, trial, param5=False)
        picks = pick_types(raw.info, eeg=True, stim=False, eog=False)
        tmin, tmax = tmin, tmax = -2., 1.
        all_event_id = dict(flex = 5000, extend = 5001, sup = 5002, pro = 5003, close = 5004, hopen = 5005, rest = 1542)
        event_id = dict(flex = 5000, extend = 5001, sup = 5002, pro = 5003, close = 5004, hopen = 5005, rest = 1542)
        test_epochs = Epochs(raw.copy(), events, all_event_id, tmin, tmax, proj=True, picks=picks,
                                baseline=None, preload=True)
        return test_epochs

    def export_configuration_electrodes(self, configuration, dataset, subject, trial, test_epochs):

        if configuration == "All":
            #picks = pick_types(test_epochs.info, eeg=True)
            picks = None
            self.show_me_the_mask(test_epochs, picks)
        if configuration == "32-elec":
            picks = pick_types(test_epochs.info, include= ['Cz','blah'])
            self.show_me_the_mask(test_epochs, picks)
        if configuration == "16-central":
            picks = pick_types(test_epochs.info, include= ['Cz','C1'])
            self.show_me_the_mask(test_epochs, picks)
        if configuration == "Emotiv":
            channel_list = ['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7',
                            'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']
            picks = pick_types(test_epochs.info, include= channel_list)
            self.show_me_the_mask(test_epochs, picks)
        if configuration == "5-central":
            picks = pick_types(test_epochs.info, include= ['Cz','C1'])
            self.show_me_the_mask(test_epochs, picks)

    def show_me_the_mask(self, test_epochs, test_mask): # Currently creates a popup
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # Add the mask
        values = [0] * len(test_epochs.info['ch_names'])
        values = np.array(values)
        revised_values = values.reshape(len(values),1)
        evoked_test_data = mne.EvokedArray(revised_values, test_epochs.info)
        if test_mask is None:
            evoked_test_data.plot_topomap(ch_type='eeg', scalings=1,
                     cmap='Reds', cbar_fmt='-%0.1f',
                    size=5, show_names=False, colorbar =False, axes=ax)
        else:
            mask_values = np.zeros(61, dtype=int)
            #test_mask = [0, 1, 2, 3, 4, 5]
            mask_values[test_mask] = 1
            mask_values = [[i] for i in mask_values]
            mask_value = np.array(mask_values)
            evoked_test_data.plot_topomap(ch_type='eeg', scalings=1,
                     cmap='Reds', cbar_fmt='-%0.1f',
                    size=5, show_names=False, mask=mask_value, colorbar =False, axes=ax)

        self.draw()
        plt.close(1)

    def show_me_the_sensors(self, dataset, subject, trial): # Currently does not create a popup
        raw = create_raw_data(dataset, subject, trial)
        self.figure.clear()
        #self.figure.close()
        ax = self.figure.add_subplot(111)
        #raw.plot_sensors(ch_type='eeg', ch_groups = np.array([[1, 2],[3, 4]]), axes=ax, show_names = False)
        raw.plot_sensors(ch_type='eeg', axes=ax, show_names = False)
        self.draw()
