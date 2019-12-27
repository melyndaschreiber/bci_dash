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
from visualize_montages import PlotCanvas

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("EEG Data Analysis")
        # self.left = 10
        # self.top = 10
        # self.width = 640
        # self.height = 400
        # Creat the layout for the first row
        self.layout_row1 = QHBoxLayout()
        self.dataset_label = QLabel("Dataset") # Text in the middle
        self.dataset_combo_box = QComboBox()
        self.dataset_combo_box.addItems(["Motor Imagination", "Motor Execution",
                                        "Spinal Cord Injury Offline", "Spinal Cord Injury Online Train",
                                        "Spinal Cord Injury Online Test", "WAY-EEG-GAL"])
        self.subject_label = QLabel("Subject") # Text in the middle
        self.subject_combo_box = QComboBox()
        self.subject_combo_box.addItems(["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5",
                                        "Subject 6", "Subject 7", "Subject 8", "Subject 9", "Subject 10",
                                        "Subject 11", "Subject 12", "Subject 13", "Subject 14", "Subject 15"])
        self.trial_label = QLabel("Trial") # Text in the middle
        self.trial_combo_box = QComboBox()
        self.trial_combo_box.addItems(["Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5",
                                        "Trial 6", "Trial 7", "Trial 8", "Trial 9", "Trial 10",
                                        "Trial 11", "Trial 12", "Trial 13", "Trial 14", "Trial 15"])

        all_widgets_row1 = [self.dataset_label, self.dataset_combo_box, self.subject_label,
                    self.subject_combo_box, self.trial_label, self.trial_combo_box]

        for w in all_widgets_row1:
            self.layout_row1.addWidget(w)

        # Add a large push QPushButton
        self.load_data_button = QPushButton('Qt Export')
        self.load_data_button.clicked.connect(self.load_data_click)
        self.real_time_button = QPushButton('Real Time')
        #self.real_time_button.clicked.connect(self.real_time_click)
        self.graph_options = QHBoxLayout()
        self.graph_options.addWidget(self.real_time_button)
        self.graph_options.addWidget(self.load_data_button)

        self.trial_info = QVBoxLayout()
        self.trial_info.addLayout(self.graph_options)
        self.trial_info.addLayout(self.layout_row1)

        # create the layout for the second row
        self.layout_row2 = QHBoxLayout()
        self.artifact_label = QLabel("Artifact Rejection")
        self.artifact_combo_box = QComboBox()
        self.artifact_combo_box.addItems(["None", "ICA and Eye"])

        self.sample_rate_label = QLabel("Sample Rate")
        self.sample_rate_combo_box = QComboBox()
        self.sample_rate_combo_box.addItems(["512", "256", "128"])

        self.config_label = QLabel("Configurations")
        self.config_combo_box = QComboBox()
        self.config_combo_box.addItems(["All", "32-elec", "16-central", "Emotiv", "5-central"])

        all_widgets_row2 = [self.artifact_label, self.artifact_combo_box, self.sample_rate_label,
                            self.sample_rate_combo_box, self.config_label, self.config_combo_box]

        for w in all_widgets_row2:
            self.layout_row2.addWidget(w)

        self.trial_info.addLayout(self.layout_row2)
        self.m = PlotCanvas(self)
        self.trial_info.addWidget(self.m)
        self.trial_info.addStretch()

        self.widget = QWidget()
        self.widget.setLayout(self.trial_info)
        self.setCentralWidget(self.widget)

    def load_data_click(self):
        # Get all the information from the dash and use it to laod data
        dataset = self.dataset_combo_box.currentText()
        subject_str = self.subject_combo_box.currentText()
        subject_num = subject_str.split()[1]
        trial_str = self.trial_combo_box.currentText()
        trial_num = trial_str.split()[1]
        print('Plotting the current montage.')
        test_epochs = self.m.generate_test_epochs(dataset, int(subject_num), int(trial_num))
        #self.m.show_me_the_mask(test_epochs)
        #self.m.show_me_the_sensors(dataset, int(subject_num), int(trial_num))
        confguration = self.config_combo_box.currentText()
        self.m.export_configuration_electrodes(confguration, dataset, int(subject_num), int(trial_num), test_epochs)
        #raw.plot()

app = QApplication(sys.argv) # Creat the applicaiton
window = MainWindow() # Creat the windwo object
qtmodern.styles.dark(app)
mw = qtmodern.windows.ModernWindow(window)
mw.show()
#window.show() # Show the window
app.exec_() # Execute the program
