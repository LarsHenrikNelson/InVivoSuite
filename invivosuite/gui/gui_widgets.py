from pathlib import Path, PurePath

from PySide6.QtCore import (
    QAbstractListModel,
    QMutex,
    QObject,
    QRunnable,
    Qt,
    # QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QLineEdit,
    QListView,
    QSpinBox,
    QWidget,
)

from ..acq import AcqManager


class LineEdit(QLineEdit):
    """
    This is a subclass of QLineEdit that returns values that are usable for
    Python.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def toInt(self):
        if len(self.text()) == 0:
            x = None
        else:
            y = float(self.text())
            x = int(y)
        return x

    def toText(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = self.text()
        return x

    def toFloat(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = float(self.text())
        return x


class ThreadWorker(QRunnable):
    """
    This class is used to create a 'runner' in a different thread than the
    main GUI. This prevents that GUI from freezing during saving.
    """

    def __init__(self, exp_manager, function: str, **kwargs):
        super().__init__()

        self.exp_manager = exp_manager
        self.signals = WorkerSignals()
        self.function = function
        self.kwargs = kwargs
        self.mutex = QMutex()

    @Slot()
    def run(self):
        self.mutex.lock()
        self.exp_manager.set_callback(self.signals.progress.emit)
        if self.function == "save":
            self.exp_manager.save_data(**self.kwargs)
        elif self.function == "analyze":
            self.exp_manager.analyze_exp(**self.kwargs)
        elif self.function == "load":
            self.exp_manager.load_exp(**self.kwargs)
        elif self.function == "create_exp":
            self.exp_manager.create_exp(**self.kwargs)
        self.signals.finished.emit("Finished")
        self.mutex.unlock()


class WorkerSignals(QObject):
    """
    This is general 'worker' that provides feedback from events to the window.
    The 'worker' also provides a 'runner' to the main GUI thread to prevent it
    from freezing when there are long running events.
    """

    file = Signal(object)
    progress = Signal(object)
    finished = Signal(str)
    path = Signal(object)


class ListView(QListView):
    """
    This is a custom listview that allows for drag and drop loading of
    scanimage matlab files.
    """

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(self.SelectionMode.SingleSelection)
        self.setModel(ListModel())

    def dragEnterEvent(self, e):
        """
        This function will detect the drag enter event from the mouse on the
        main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        """
        This function will detect the drag move event on the main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    @Slot()
    def dropEvent(self, e):
        """
        This function will enable the drop file directly on to the
        main window. The file location will be stored in the self.filename
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            self.model().addData(e.mimeData().urls())
        else:
            e.ignore()

    def clearData(self):
        self.model().clearData()
        self.model().layoutChanged.emit()

    def deleteSelection(self, index_list):
        rows = [i.row() for i in index_list]
        indexes = sorted(rows, reverse=True)
        for i in indexes:
            self.model().deleteSelection(i)
            self.model().layoutChanged.emit()
            self.clearSelection()

    def addAcq(self, urls):
        self.model().addData(urls)
        self.model().layoutChanged.emit()

    def setAnalysisType(self, analysis):
        self.model().setAnalysisType(analysis)

    def setData(self, exp_manager):
        self.model().setLoadData(exp_manager)

    def getAcqID(self):
        index = self.currentIndex().row()
        return self.model().acq_names[index]


class ListModel(QAbstractListModel):
    """
    The model contains all the load data for the list view. Qt works using a
    model-view-controller framework so the view should not contain any data
    analysis or loading, it just facilities the transfer of the data to the model.
    The model is used to add, remove, and modify the data through the use of a
    controller which in Qt is often built into the models.
    """

    def __init__(self):
        super().__init__()
        self.acq_names = []
        self.signals = WorkerSignals()

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            x = self.acq_names[index.row()]
            return x

    def rowCount(self, index):
        return len(self.acq_names)

    def setAnalysisType(self, analysis):
        """
        This function sets the analysis type for the model which is used to set the
        acquisition type that is created in dropEvent().
        """
        self.analysis_type = analysis

    def deleteSelection(self, index):
        keys = list(self.exp_manager.exp_dict[self.analysis_type].keys())
        keys.sort(key=lambda d: int(d.split("_")[1]))

        # Need to catch cases where the index does not exist anymore.
        # I cannot figure out why Qt keeps returning the extra index.
        if index > len(keys):
            pass
        else:
            self.removeRow(index)
            key = keys[index]
            del self.exp_manager.exp_dict[self.analysis_type][key]
            self.layoutChanged.emit()
            self.sortNames()

    def clearData(self):
        self.acq_dict = {}
        self.acq_names = []
        self.exp_manager = None

    def addData(self, urls):
        if not isinstance(urls[0], str):
            urls = [PurePath(str(url.toLocalFile())) for url in urls]
        for value in urls:
            temp = AcqManager()
            temp.load_hdf5(value)
            self.exp_manager[value.stem] = temp
            self.acq_names.append(value.stem)
        self.layoutChanged.emit()

    def updateProgress(self, value):
        self.signals.progress.emit(value)

    def setLoadData(self, exp_manager):
        self.exp_manager = exp_manager


class StringBox(QSpinBox):
    def __init__(self, parent=None):
        super(StringBox, self).__init__(parent)
        strings = []
        self.setStrings(strings)

    def setStrings(self, strings):
        strings = list(strings)
        self._strings = tuple(strings)
        self._values = dict(zip(strings, range(len(strings))))
        self.setRange(0, len(strings) - 1)

    def textFromValue(self, value):
        # returning string from index
        # _string = tuple
        return self._strings[value]


class DragDropWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.signals = WorkerSignals()

    def dragEnterEvent(self, e):
        """
        This function will detect the drag enter event from the mouse on the
        main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        """
        This function will detect the drag move event on the main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    @Slot()
    def dropEvent(self, e):
        """
        This function will enable the drop file directly on to the
        main window. The file location will be stored in the self.filename
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            url = e.mimeData().urls()[0]
            fname = PurePath(str(url.toLocalFile()))
            if fname.suffix == ".yaml":
                self.signals.file.emit(fname)
            elif Path(fname).is_dir():
                self.signals.path.emit(Path(fname))
            else:
                e.ignore()
        else:
            e.ignore()
