import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random
from load_data import create_raw_data
import mne
import numpy as np

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0,0)

        button = QPushButton('Load Montage', self)
        button.clicked.connect(self.load_data_click)

        button.setToolTip('This s an example button')
        button.move(500,0)
        button.resize(140,100)

        self.show()

    def load_data_click(self):
        print('Plotting the current montage.')
        self.m.show_me_the_figure('Motor Execution', 1, 1)


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # Can't seem to get rid of the text
        #self.figure.text(0.1, 0.5, 'Load data to see the montage.', dict(size=10))

    def default_montage(self):
        self.figure.text.remove()
        ax = self.figure.add_subplot(111)
        biosemi_layout = mne.channels.read_layout('biosemi')
        midline = np.where([name.endswith('z') for name in biosemi_layout.names])[0]
        biosemi_layout.plot(picks=midline, axes = ax)
        self.draw()

    def show_me_the_figure(self, dataset, subject, trial):
        raw = create_raw_data(dataset, subject, trial)
        ax = self.figure.add_subplot(111)
        raw.plot_sensors(ch_type='eeg', axes=ax, show_names = True)
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
