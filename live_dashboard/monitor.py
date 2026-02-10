"""
Real-time Training Monitor for VAE Models.

This application launches a standalone GUI window to visualize training metrics
parsed from a live log file. It utilizes PyQt6 for the interface and
PyQtGraph for high-performance plotting.

Designed for a "cyberpunk/terminal" aesthetic with high readability.
"""

import sys
import os
import time
import re
import glob
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QTextEdit, QGridLayout,
    QFileDialog, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QTextCursor

import pyqtgraph as pg

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Directory where log files are stored. Use '.' for current directory.
    'LOG_DIR': '..logs/',

    # Pattern to match log files (e.g., 'run_*.log').
    'LOG_PATTERN': 'run_*.log',

    # Update frequency for file checking (in seconds).
    'REFRESH_RATE': 0.1,

    # Max lines to keep in the log console to save memory.
    'MAX_LOG_LINES': 1000,

    # Default window size.
    'WINDOW_SIZE': (1280, 800)
}

# ==============================================================================
# THEME & STYLING
# ==============================================================================

COLORS = {
    'bg': '#0a0c0f',         # Deepest background
    'surface': '#13171e',    # Panel background
    'border': '#2a3b4d',     # Subtle borders
    'accent_1': '#00ff9d',   # Success/Green (Terminal style)
    'accent_2': '#00d0ff',   # Info/Cyan
    'accent_3': '#ff5c5c',   # Error/Divergence Red
    'text_main': '#e0e6ed',  # Primary text
    'text_dim': '#5c6e7f',   # Secondary/Metadata text
    'grid': '#2a3b4d'        # Chart grid lines
}

STYLE_SHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg']};
}}
QWidget {{
    background-color: {COLORS['bg']};
    color: {COLORS['text_main']};
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 10pt;
}}
QFrame {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 0px;
}}
QLabel {{
    border: none;
    background-color: transparent;
    font-weight: normal; 
}}
QTextEdit {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    color: {COLORS['text_main']};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 9pt;
}}
QPushButton {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['accent_2']};
    color: {COLORS['accent_2']};
    padding: 6px 12px;
    border-radius: 0px;
    font-family: 'JetBrains Mono';
}}
QPushButton:hover {{
    background-color: {COLORS['accent_2']}15;
}}
"""


class LogWorker(QThread):
    """
    Background worker thread that tails a log file and parses lines in real-time.

    Attributes:
        data_updated (pyqtSignal): Emitted when a training metric is parsed.
        log_line_added (pyqtSignal): Emitted when a new raw line is read.
    """
    data_updated = pyqtSignal(dict)
    log_line_added = pyqtSignal(str)

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self.running = True
        self.file_pos = 0

        # Regex patterns derived from the provided log format
        self.re_train = re.compile(
            r"Epoch\s+(\d+)\s*-\s*Train Loss:\s*([\d.]+)\s*\(Recon:\s*([\d.]+),\s*KL:\s*([\d.]+)"
        )
        self.re_val = re.compile(
            r"Epoch\s+(\d+)\s*-\s*Val Loss:\s*([\d.]+)"
        )
        self.re_sample = re.compile(
            r"Generated sample - Density:\s*([\d.]+)%"
        )

    def run(self):
        """Main execution loop for the thread."""
        while self.running:
            if not os.path.exists(self.log_path):
                time.sleep(1)
                continue

            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.file_pos)
                lines = f.readlines()
                self.file_pos = f.tell()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    self.log_line_added.emit(line)
                    self._parse_line(line)

            time.sleep(CONFIG['REFRESH_RATE'])

    def _parse_line(self, line: str):
        """Parses a single log line and emits structured data."""
        data_packet = {}

        # 1. Parse Training Data
        m_train = self.re_train.search(line)
        if m_train:
            data_packet['type'] = 'train'
            data_packet['epoch'] = int(m_train.group(1))
            data_packet['loss'] = float(m_train.group(2))
            data_packet['recon'] = float(m_train.group(3))
            data_packet['kl'] = float(m_train.group(4))
            self.data_updated.emit(data_packet)
            return

        # 2. Parse Validation Data
        m_val = self.re_val.search(line)
        if m_val:
            data_packet['type'] = 'val'
            data_packet['epoch'] = int(m_val.group(1))
            data_packet['val_loss'] = float(m_val.group(2))
            self.data_updated.emit(data_packet)
            return

        # 3. Parse Sampling Data
        m_sample = self.re_sample.search(line)
        if m_sample:
            data_packet['type'] = 'sample'
            data_packet['density'] = float(m_sample.group(1))
            self.data_updated.emit(data_packet)

    def stop(self):
        """Stops the worker safely."""
        self.running = False


class StatCard(QFrame):
    """
    A UI widget representing a single metric card.
    """

    def __init__(self, title: str, accent_color: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Internal layout
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)

        # Title
        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 8pt; letter-spacing: 1px;")

        # Main Value
        self.lbl_value = QLabel("—")
        self.lbl_value.setStyleSheet(
            f"color: {accent_color}; font-size: 18pt; font-family: 'JetBrains Mono';"
        )

        # Subtext (e.g., 'epoch 50')
        self.lbl_sub = QLabel("waiting...")
        self.lbl_sub.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 8pt;")

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)
        layout.addWidget(self.lbl_sub)
        self.setLayout(layout)

        # Decorative left border
        self.setStyleSheet(
            f"QFrame {{ border-left: 2px solid {accent_color}; background-color: {COLORS['surface']}; }}"
        )

    def update_value(self, value: str, subtext: str = ""):
        """Updates the card's content."""
        self.lbl_value.setText(str(value))
        if subtext:
            self.lbl_sub.setText(subtext)


class TrainingMonitor(QMainWindow):
    """
    Main Application Window.
    """

    def __init__(self):
        super().__init__()

        # --- Data Attributes Initialization ---
        self.epochs_train: List[int] = []
        self.loss_train: List[float] = []
        self.recon_train: List[float] = []
        self.kl_train: List[float] = []
        self.epochs_val: List[int] = []
        self.loss_val: List[float] = []

        self.latest_log_file: Optional[str] = None
        self.worker: Optional[LogWorker] = None

        # --- UI Attributes Initialization (PEP 8) ---
        self.status_label: Optional[QLabel] = None
        self.log_display: Optional[QTextEdit] = None

        self.card_epoch: Optional[StatCard] = None
        self.card_loss: Optional[StatCard] = None
        self.card_val: Optional[StatCard] = None
        self.card_kl: Optional[StatCard] = None
        self.card_density: Optional[StatCard] = None

        self.plot_loss: Optional[pg.PlotWidget] = None
        self.plot_components: Optional[pg.PlotWidget] = None

        self.curve_train_loss: Optional[pg.PlotDataItem] = None
        self.curve_val_loss: Optional[pg.PlotDataItem] = None
        self.curve_recon: Optional[pg.PlotDataItem] = None
        self.curve_kl: Optional[pg.PlotDataItem] = None

        # --- Setup ---
        self.setWindowTitle("VAE TRAINING MONITOR")
        self.resize(*CONFIG['WINDOW_SIZE'])
        self.setStyleSheet(STYLE_SHEET)

        self.init_ui()

        # Auto-detect log file on startup
        self.latest_log_file = self.find_latest_log()

        if self.latest_log_file:
            self.log_display.append(
                f"<span style='color:{COLORS['accent_1']}'>Found latest log: {self.latest_log_file}</span>"
            )
            self.start_monitoring(self.latest_log_file)
        else:
            self.log_display.append(
                f"<span style='color:{COLORS['text_dim']}'>"
                f"No logs found in {os.path.abspath(CONFIG['LOG_DIR'])}. Waiting for file selection...</span>"
            )

    @staticmethod
    def find_latest_log() -> Optional[str]:
        """
        Finds the most recent log file based on CONFIG parameters.
        """
        search_path = os.path.join(CONFIG['LOG_DIR'], CONFIG['LOG_PATTERN'])
        files = glob.glob(search_path)

        if not files:
            return None

        # Sort by modification time, newest last
        return max(files, key=os.path.getmtime)

    def init_ui(self):
        """Constructs the GUI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 25)

        # 1. HEADER
        header_layout = QHBoxLayout()

        title = QLabel("VAE MONITOR // SYSTEM ACTIVE")
        title.setStyleSheet(f"color: {COLORS['text_main']}; font-size: 14pt; letter-spacing: 2px;")

        self.status_label = QLabel("● NO SIGNAL")
        self.status_label.setStyleSheet(f"color: {COLORS['text_dim']};")

        btn_load = QPushButton("SELECT SOURCE")
        btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_load.clicked.connect(self.select_file)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        header_layout.addWidget(btn_load)
        main_layout.addLayout(header_layout)

        # 2. METRICS ROW
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        self.card_epoch = StatCard("Epoch", COLORS['accent_2'])
        self.card_loss = StatCard("Train Loss", COLORS['accent_1'])
        self.card_val = StatCard("Val Loss", COLORS['accent_2'])
        self.card_kl = StatCard("KL Divergence", COLORS['accent_3'])
        self.card_density = StatCard("Density", COLORS['text_main'])

        stats_layout.addWidget(self.card_epoch)
        stats_layout.addWidget(self.card_loss)
        stats_layout.addWidget(self.card_val)
        stats_layout.addWidget(self.card_kl)
        stats_layout.addWidget(self.card_density)
        main_layout.addLayout(stats_layout)

        # 3. CHARTS AREA
        charts_layout = QGridLayout()

        # Global Chart Styling
        pg.setConfigOption('background', COLORS['surface'])
        pg.setConfigOption('foreground', COLORS['text_dim'])
        pg.setConfigOptions(antialias=True)

        # --- Chart 1: Loss ---
        self.plot_loss = pg.PlotWidget()
        self._style_plot(self.plot_loss, "Total Loss (Log Scale)")
        self.plot_loss.setLogMode(y=True)

        self.curve_train_loss = self.plot_loss.plot(
            pen=pg.mkPen(COLORS['accent_1'], width=1), name='Train'
        )
        self.curve_val_loss = self.plot_loss.plot(
            pen=pg.mkPen(COLORS['accent_2'], width=1, style=Qt.PenStyle.DashLine), name='Val'
        )

        # --- Chart 2: Components ---
        self.plot_components = pg.PlotWidget()
        self._style_plot(self.plot_components, "Recon vs KL (Log Scale)")
        self.plot_components.setLogMode(y=True)
        self.plot_components.addLegend(offset=(10, 10), labelTextColor=COLORS['text_dim'])

        self.curve_recon = self.plot_components.plot(
            pen=pg.mkPen(COLORS['accent_2'], width=1), name='Recon'
        )
        self.curve_kl = self.plot_components.plot(
            pen=pg.mkPen(COLORS['accent_3'], width=1), name='KL'
        )

        charts_layout.addWidget(self.plot_loss, 0, 0)
        charts_layout.addWidget(self.plot_components, 0, 1)
        main_layout.addLayout(charts_layout, stretch=3)

        # 4. LOG CONSOLE
        log_frame = QFrame()
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)

        lbl_log_header = QLabel(" DATA STREAM")
        lbl_log_header.setStyleSheet(
            f"background-color: {COLORS['border']}; color: {COLORS['text_dim']}; "
            f"padding: 5px 10px; font-size: 8pt; letter-spacing: 1px;"
        )

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("border: none; padding: 10px;")

        log_layout.addWidget(lbl_log_header)
        log_layout.addWidget(self.log_display)
        main_layout.addWidget(log_frame, stretch=2)

    def _style_plot(self, plot_widget: pg.PlotWidget, title: str):
        """Applies consistent styling to plot widgets."""
        plot_widget.showGrid(x=True, y=True, alpha=0.1)
        plot_widget.setTitle(title, color=COLORS['text_main'], size='9pt')
        plot_widget.getAxis('left').setPen(COLORS['border'])
        plot_widget.getAxis('bottom').setPen(COLORS['border'])
        # Remove white borders
        plot_widget.setBackground(COLORS['surface'])

    def select_file(self):
        """Opens file dialog for manual selection."""
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Open Log File', CONFIG['LOG_DIR'], "Log files (*.log *.txt)"
        )
        if fname:
            self.start_monitoring(fname)

    def start_monitoring(self, path: str):
        """Initializes the worker thread for the selected file."""
        # Clear previous data
        self.epochs_train.clear()
        self.loss_train.clear()
        self.recon_train.clear()
        self.kl_train.clear()
        self.epochs_val.clear()
        self.loss_val.clear()
        self.log_display.clear()

        # Update UI status
        filename = os.path.basename(path)
        self.status_label.setText(f"● MONITORING: {filename}")
        self.status_label.setStyleSheet(f"color: {COLORS['accent_1']};")

        # Handle Threading
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()

        self.worker = LogWorker(path)
        self.worker.data_updated.connect(self.update_charts)
        self.worker.log_line_added.connect(self.append_log)
        self.worker.start()

    def update_charts(self, data: Dict):
        """Receives parsed data and updates widgets."""
        if data['type'] == 'train':
            epoch = data['epoch']
            loss = data['loss']

            # Update Data Structures
            self.epochs_train.append(epoch)
            self.loss_train.append(loss)
            self.recon_train.append(data['recon'])
            self.kl_train.append(data['kl'])

            # Update UI Cards
            self.card_epoch.update_value(str(epoch))
            self.card_loss.update_value(f"{loss:.1f}", "train loss")
            self.card_kl.update_value(f"{data['kl']:.1f}", "divergence")

            # Update Plots (Efficiently)
            self.curve_train_loss.setData(self.epochs_train, self.loss_train)
            self.curve_recon.setData(self.epochs_train, self.recon_train)
            self.curve_kl.setData(self.epochs_train, self.kl_train)

        elif data['type'] == 'val':
            self.epochs_val.append(data['epoch'])
            self.loss_val.append(data['val_loss'])

            self.card_val.update_value(f"{data['val_loss']:.1f}", f"epoch {data['epoch']}")
            self.curve_val_loss.setData(self.epochs_val, self.loss_val)

        elif data['type'] == 'sample':
            self.card_density.update_value(f"{data['density']}%", "generated")

    def append_log(self, line: str):
        """Formats and appends a line to the log console."""
        # Simple color coding for log levels
        color = COLORS['text_dim']
        if "ERROR" in line:
            color = COLORS['accent_3']
        elif "WARNING" in line:
            color = "#ffd700"
        elif "New best" in line:
            color = COLORS['accent_1']
        elif "Generated sample" in line:
            color = "#d19a66"
        elif "Train Loss" in line:
            color = COLORS['text_main']

        # Split timestamp and message
        parts = line.split(' - ', 1)
        if len(parts) == 2:
            timestamp, message = parts
        else:
            timestamp, message = "", line

        html = (
            f"<span style='color:{COLORS['text_dim']}'>{timestamp}</span> "
            f"<span style='color:{COLORS['border']}'>|</span> "
            f"<span style='color:{color}'>{message}</span>"
        )

        self.log_display.moveCursor(QTextCursor.MoveOperation.End)
        self.log_display.insertHtml(html + "<br>")
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event):
        """Cleanup on application exit."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Enable High DPI support
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Windows specific dark mode title bar
    if os.name == 'nt':
        sys.argv += ['-platform', 'windows:darkmode=1']
        app.setStyle('Fusion')

    window = TrainingMonitor()
    window.show()
    sys.exit(app.exec())