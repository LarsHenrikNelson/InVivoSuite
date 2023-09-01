import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .gui_widgets import ListView


class LFPWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        self.load_layout = QVBoxLayout()
        self.main_layout.addLayout(self.load_layout)
        self.load_widget = ListView()
        self.load_widget.setMaximumWidth(300)
        self.load_widget.clicked.connect(self.set_acq_spinbox)
        self.load_layout.addWidget(self.load_widget)
        self.exp_manager = {}
        self.load_widget.setData(self.exp_manager)

        self.del_sel_button = QPushButton("Delete selection")
        self.del_sel_button.setMaximumWidth(300)
        self.load_layout.addWidget(self.del_sel_button)
        self.del_sel_button.clicked.connect(self.delSelection)

        self.pbar = QProgressBar(self)
        self.pbar.setMaximumWidth(300)
        self.load_layout.addWidget(self.pbar)

        self.plot_check_list = QFormLayout()
        self.main_layout.addLayout(self.plot_check_list)

        self.plot_spinbox = QSpinBox()
        self.plot_spinbox.valueChanged.connect(self.plot_acq)
        self.plot_check_list.addRow("Acquisition", self.plot_spinbox)

        self.plot_bursts = QCheckBox()
        self.plot_check_list.addRow("Plot bursts", self.plot_bursts)

        self.channel_map = QCheckBox()
        self.plot_check_list.addRow("Map channels", self.channel_map)

        self.plot_layout = QVBoxLayout()
        self.main_layout.addLayout(self.plot_layout)
        self.main_plot = pg.PlotWidget(useOpenGl=True)
        self.plot_layout.addWidget(self.main_plot)
        self.ste_plot = pg.PlotWidget(useOpenGl=True)
        self.plot_layout.addWidget(self.ste_plot)
        self.access_plot = pg.PlotWidget(useOpenGl=True)
        self.access_plot.setMaximumHeight(200)
        self.access_plot.plotItem.setMouseEnabled(x=False)
        self.access_plot.plotItem.setMouseEnabled(y=False)
        self.plot_layout.addWidget(self.access_plot)

        self.region = pg.LinearRegionItem()
        self.region.sigRegionChanged.connect(self.update)
        self.main_plot.sigRangeChanged.connect(self.updateRegion)
        self.ste_plot.sigRangeChanged.connect(self.updateRegion)

        # Set the initial bounds of the region and its layer
        # position.
        self.region.setRegion([0, 30])
        self.region.setZValue(10)

    def update(self):
        """
        This functions is used for the draggable region.
        See PyQtGraphs documentation for more information.
        """
        self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        self.main_plot.setXRange(minX, maxX, padding=0)
        self.ste_plot.setXRange(minX, maxX, padding=0)

    def updateRegion(self, window, viewRange):
        """
        This functions is used for the draggable region.
        See PyQtGraphs documentation for more information
        """
        rgn = viewRange[0]
        self.region.setRegion(rgn)

    def updateProgress(self, value):
        if isinstance(value, (int, float)):
            self.pbar.setValue(value)
        elif isinstance(value, str):
            self.pbar.setFormat(value)

    def delSelection(self):
        self.load_widget.clearData()
        self.exp_manager = {}
        self.load_widget.setData(self.exp_manager)
        self.main_plot.clear()
        self.access_plot.clear()
        self.ste_plot.clear()

    def set_acq_spinbox(self):
        id = self.load_widget.getAcqID()
        channels = self.exp_manager[id].num_channels
        self.plot_spinbox.setRange(1, channels)

    def plot_acq(self, num):
        self.main_plot.clear()
        self.access_plot.clear()
        self.ste_plot.clear()
        id = self.load_widget.getAcqID()
        acq = self.exp_manager[id].acq(
            num - 1, "lfp", map_channel=self.channel_map.isChecked()
        )
        x = np.arange(acq.size) / 1000
        self.main_plot.plot(x=x, y=acq, name="main")
        self.access_plot.plot(x=x, y=acq, name="access")
        self.access_plot.addItem(self.region, ignoreBounds=True)
        fs = self.exp_manager[id].get_grp_attr("lfp", "sample_rate")
        wlen = self.exp_manager[id].get_grp_attr("lfp_bursts", "wlen")
        window = self.exp_manager[id].get_grp_attr("lfp_bursts", "window")
        ste = self.exp_manager[id].get_short_time_energy(
            acq,
            wlen=wlen,
            window=window,
            fs=fs,
        )
        self.ste_plot.plot(x=x, y=ste)
        if self.plot_bursts.isChecked():
            b = self.exp_manager[id].get_lfp_burst_indexes(
                num - 1, map_channel=self.channel_map.isChecked()
            )
            for i in range(b.shape[0]):
                self.main_plot.plot(
                    x=x[int(b[i, 0]) : int(b[i, 1])],
                    y=acq[int(b[i, 0]) : int(b[i, 1])],
                    name=i,
                    pen="r",
                )
        self.exp_manager[id].close()
