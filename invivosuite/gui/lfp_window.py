import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
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

        self.acq_type_selection = QComboBox(self)
        self.acq_type_selection.addItems(["wideband", "spike", "lfp"])
        self.plot_check_list.addRow("Signal", self.acq_type_selection)

        # self.plot_bursts = QCheckBox()
        # self.plot_check_list.addRow("Plot bursts", self.plot_bursts)

        self.channel_map = QCheckBox()
        self.plot_check_list.addRow("Map channels", self.channel_map)

        self.cmr = QCheckBox()
        self.plot_check_list.addRow("CMR", self.cmr)

        self.cmr_probe = QLineEdit()
        self.plot_check_list.addRow("CMR probe", self.cmr_probe)

        self.probe = QLineEdit()
        self.plot_check_list.addRow("Probe", self.probe)
        self.left_button = QToolButton()
        self.left_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.left_button.clicked.connect(self.leftButton)
        self.left_button.setArrowType(Qt.LeftArrow)
        self.left_button.setMinimumWidth(70)
        self.left_button.setMaximumWidth(70)
        self.plot_check_list.addRow(self.left_button)

        self.right_button = QToolButton()
        self.right_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.right_button.clicked.connect(self.rightButton)
        self.right_button.setArrowType(Qt.RightArrow)
        self.right_button.setMinimumWidth(70)
        self.right_button.setMaximumWidth(70)
        self.plot_check_list.addRow(self.right_button)

        self.plot_layout = QVBoxLayout()
        self.main_layout.addLayout(self.plot_layout)
        self.main_plot = pg.PlotWidget(useOpenGl=True)
        self.plot_layout.addWidget(self.main_plot)
        # self.ste_plot = pg.PlotWidget(useOpenGl=True)
        # self.plot_layout.addWidget(self.ste_plot)
        self.access_plot = pg.PlotWidget(useOpenGl=True)
        self.access_plot.setMaximumHeight(200)
        self.access_plot.plotItem.setMouseEnabled(x=False)
        self.access_plot.plotItem.setMouseEnabled(y=False)
        self.plot_layout.addWidget(self.access_plot)

        self.region = pg.LinearRegionItem()
        self.region.sigRegionChanged.connect(self.update)
        self.main_plot.sigRangeChanged.connect(self.updateRegion)
        # self.ste_plot.sigRangeChanged.connect(self.updateRegion)

        # Set the initial bounds of the region and its layer
        # position.
        self.region.setRegion([0, 30])
        self.region.setZValue(10)

    def leftButton(self):
        if (self.start - 3000000) > 0:
            self.current_chunk[0] -= 3000000
            self.current_chunk[1] -= 3000000
        else:
            self.current_chunk[0] = 0
            self.current_chunk[1] = 3000000
        self.plot()

    def rightButton(self):
        if self.end >= (self.current_chunk[1] + 3000000):
            self.current_chunk[0] += 3000000
            self.current_chunk[1] += 3000000
        else:
            self.current_chunk[0] = self.end - 3000000
            self.current_chunk[1] = self.end
        self.plot()

    def update(self):
        """
        This functions is used for the draggable region.
        See PyQtGraphs documentation for more information.
        """
        self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        self.main_plot.setXRange(minX, maxX, padding=0)
        # self.ste_plot.setXRange(minX, maxX, padding=0)

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
        # self.ste_plot.clear()

    def set_acq_spinbox(self):
        ident = self.load_widget.getAcqID()
        channels = self.exp_manager[ident].n_chans
        self.plot_spinbox.setRange(1, channels)

    def plot_acq(self, num):
        ident = self.load_widget.getAcqID()
        self.start = self.exp_manager[ident].start
        self.current_chunk = [0, 3000000]
        self.end = self.exp_manager[ident].end
        self.plot()
        # fs = self.exp_manager[id].get_grp_attr("lfp", "sample_rate")
        # wlen = self.exp_manager[id].get_grp_attr("lfp_bursts", "wlen")
        # window = self.exp_manager[id].get_grp_attr("lfp_bursts", "window")
        # ste = self.exp_manager[id].get_short_time_energy(
        #     acq,
        #     wlen=wlen,
        #     window=window,
        #     fs=fs,
        # )
        # baseline = self.exp_manager[id].get_ste_baseline(ste)
        # self.ste_plot.plot(x=x, y=ste)
        # self.ste_plot.plot(x=x, y=baseline, pen="r")
        # if self.plot_bursts.isChecked():
        #     b = self.exp_manager[id].get_lfp_burst_indexes(
        #         num - 1, map_channel=self.channel_map.isChecked()
        #     )
        #     for i in range(b.shape[0]):
        #         self.main_plot.plot(
        #             x=x[int(b[i, 0]) : int(b[i, 1])],
        #             y=acq[int(b[i, 0]) : int(b[i, 1])],
        #             name=i,
        #             pen="r",
        #

    def plot(self):
        self.main_plot.clear()
        self.access_plot.clear()
        self.main_plot.setAutoVisible(y=True)
        self.main_plot.enableAutoRange()
        # self.ste_plot.clear()
        num = self.plot_spinbox.value()
        ident = self.load_widget.getAcqID()
        acq = self.exp_manager[ident].acq(
            acq_num=num - 1,
            acq_type=self.acq_type_selection.currentText(),
            map_channel=self.channel_map.isChecked(),
            ref_probe=self.cmr_probe.text(),
            ref=self.cmr.isChecked(),
            probe=self.probe.text(),
            start=self.current_chunk[0],
            end=self.current_chunk[1],
        )
        x = np.arange(self.current_chunk[0], self.current_chunk[1]) / 40000
        self.main_plot.plot(x=x, y=acq, name="main")
        self.access_plot.plot(x=x, y=acq, name="access")
        self.access_plot.addItem(self.region, ignoreBounds=True)
