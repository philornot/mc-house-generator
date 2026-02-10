"""
Real-time Training Monitor for VAE Models.

This module provides a professional GUI for monitoring Variational Autoencoder (VAE)
training processes. It handles log parsing in a background thread and visualizes
metrics using high-performance pyqtgraph plotting.
"""

import glob
import os
import re
import sys
from collections import deque
from typing import Optional, Dict, Any

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QTextCursor, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QTextEdit,
    QFileDialog, QPushButton, QGraphicsDropShadowEffect
)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

CONFIG = {
    'LOG_DIR': '../logs/',  # Directory relative to project root
    'LOG_PATTERN': 'run_*.log',
    'REFRESH_INTERVAL_MS': 100,
    'MAX_LOG_HISTORY': 1000,
    'WINDOW_WIDTH': 1400,
    'WINDOW_HEIGHT': 900
}

COLORS = {
    'background': '#0B0E14',
    'surface': '#151921',
    'border': '#252A34',
    'primary': '#7C4DFF',
    'success': '#00E676',
    'info': '#00B0FF',
    'error': '#FF5252',
    'text_primary': '#E1E4E8',
    'text_secondary': '#959DA5'
}

APP_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}
QWidget {{
    color: {COLORS['text_primary']};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
QFrame#MetricCard {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}
QTextEdit#LogConsole {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    color: {COLORS['text_secondary']};
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 10pt;
    line-height: 1.4;
}}
QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: #6C3BEF;
}}
"""


# ==============================================================================
# DATA PROCESSING ENGINE
# ==============================================================================

class LogParserThread(QThread):
    metrics_received = pyqtSignal(dict)
    log_received = pyqtSignal(str)

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self._is_running = True
        self._file_offset = 0

        # Flexible regex for Train and Val lines
        self._re_train = re.compile(
            r"Epoch\s+(?P<epoch>\d+)\s*-\s*Train Loss:\s*(?P<loss>[\d.]+)\s*\(Recon:\s*(?P<recon>[\d.]+),\s*KL:\s*(?P<kl>[\d.]+)"
        )
        self._re_val = re.compile(
            r"Epoch\s+(?P<epoch>\d+)\s*-\s*Val Loss:\s*(?P<loss>[\d.]+)"
        )

    def run(self):
        while self._is_running:
            if not os.path.exists(self.log_path):
                self.msleep(500)
                continue

            try:
                # Always re-check size to handle potential file rotations or truncation
                with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self._file_offset)
                    while True:
                        line = f.readline()
                        if not line:
                            break

                        clean_line = line.strip()
                        if clean_line:
                            self.log_received.emit(clean_line)
                            self._parse_metrics(clean_line)

                    self._file_offset = f.tell()
            except Exception as e:
                print(f"Log parsing error: {e}")

            self.msleep(CONFIG['REFRESH_INTERVAL_MS'])

    def _parse_metrics(self, line: str):
        if match := self._re_train.search(line):
            data = match.groupdict()
            self.metrics_received.emit({
                'type': 'train',
                'epoch': int(data['epoch']),
                'loss': float(data['loss']),
                'recon': float(data['recon']),
                'kl': float(data['kl'])
            })
        elif match := self._re_val.search(line):
            data = match.groupdict()
            self.metrics_received.emit({
                'type': 'val',
                'epoch': int(data['epoch']),
                'loss': float(data['loss'])
            })

    def stop(self):
        self._is_running = False


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

class MetricDisplayCard(QFrame):
    def __init__(self, label: str, accent_color: str):
        super().__init__()
        self.setObjectName("MetricCard")
        self.setMinimumWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel(label.upper())
        title.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")

        self.value_label = QLabel("N/A")
        self.value_label.setStyleSheet(f"color: {accent_color}; font-size: 24px; font-weight: 800;")

        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(accent_color))
        self.shadow.setOffset(0, 0)
        self.value_label.setGraphicsEffect(self.shadow)

        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")

        layout.addWidget(title)
        layout.addWidget(self.value_label)
        layout.addWidget(self.status_label)

    def update_metric(self, value: str, status: Optional[str] = None):
        self.value_label.setText(value)
        if status:
            self.status_label.setText(status)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAE Training Monitor")
        self.resize(CONFIG['WINDOW_WIDTH'], CONFIG['WINDOW_HEIGHT'])
        self.setStyleSheet(APP_STYLESHEET)

        self.metrics_history = {
            'train_epochs': [], 'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_epochs': [], 'val_loss': []
        }
        self.worker: Optional[LogParserThread] = None
        self.log_buffer = deque()

        self._setup_layout()

        # UI Refresh Timer for log console
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._flush_log_buffer)
        self.refresh_timer.start(50)

        # 1. Automatic detection of latest log on startup
        latest_log = self._get_latest_log_path()
        if latest_log:
            self._start_monitoring(latest_log)
        else:
            self.status_info.setText("Status: No logs found in " + CONFIG['LOG_DIR'])

    def _setup_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # Header Section
        header = QHBoxLayout()
        title_container = QVBoxLayout()
        main_title = QLabel("VAE TRAINING DASHBOARD")
        main_title.setStyleSheet("font-size: 20px; font-weight: 800; letter-spacing: 2px;")
        self.status_info = QLabel("Status: Idle")
        self.status_info.setStyleSheet(f"color: {COLORS['text_secondary']};")

        title_container.addWidget(main_title)
        title_container.addWidget(self.status_info)

        btn_open = QPushButton("Open Log File")
        btn_open.clicked.connect(self._on_select_file)

        header.addLayout(title_container)
        header.addStretch()
        header.addWidget(btn_open)
        main_layout.addLayout(header)

        # Metrics Cards
        cards_layout = QHBoxLayout()
        self.card_epoch = MetricDisplayCard("Epoch", COLORS['info'])
        self.card_loss = MetricDisplayCard("Total Loss", COLORS['success'])
        self.card_recon = MetricDisplayCard("Reconstruction", COLORS['info'])
        self.card_kl = MetricDisplayCard("KL Divergence", COLORS['error'])

        for card in [self.card_epoch, self.card_loss, self.card_recon, self.card_kl]:
            cards_layout.addWidget(card)
        main_layout.addLayout(cards_layout)

        # Plotting Area
        pg.setConfigOption('background', COLORS['surface'])
        pg.setConfigOption('foreground', COLORS['text_secondary'])
        pg.setConfigOptions(antialias=True)

        plots_container = QHBoxLayout()

        self.loss_plot = pg.PlotWidget(title="Loss Trends (Log Scale)")
        self.loss_plot.setLogMode(y=True)
        self.loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve_train_loss = self.loss_plot.plot(pen=pg.mkPen(COLORS['success'], width=2), name="Train")
        self.curve_val_loss = self.loss_plot.plot(pen=pg.mkPen(COLORS['info'], width=2, style=Qt.PenStyle.DashLine),
                                                  name="Val")

        self.comp_plot = pg.PlotWidget(title="VAE Components")
        self.comp_plot.setLogMode(y=True)
        self.comp_plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve_recon = self.comp_plot.plot(pen=pg.mkPen(COLORS['info'], width=2))
        self.curve_kl = self.comp_plot.plot(pen=pg.mkPen(COLORS['error'], width=2))

        plots_container.addWidget(self.loss_plot)
        plots_container.addWidget(self.comp_plot)
        main_layout.addLayout(plots_container, stretch=3)

        # Console Section
        self.console = QTextEdit()
        self.console.setObjectName("LogConsole")
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)  # Prevents messy wrapping
        main_layout.addWidget(self.console, stretch=1)

    def _get_latest_log_path(self) -> Optional[str]:
        if not os.path.exists(CONFIG['LOG_DIR']):
            return None
        pattern = os.path.join(CONFIG['LOG_DIR'], CONFIG['LOG_PATTERN'])
        files = glob.glob(pattern)
        return max(files, key=os.path.getmtime) if files else None

    def _on_select_file(self):
        # 2. Open file dialog in the configured LOG_DIR
        initial_dir = CONFIG['LOG_DIR'] if os.path.exists(CONFIG['LOG_DIR']) else "."
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Log", initial_dir, "Log Files (*.log);;All Files (*)"
        )
        if file_path:
            self._start_monitoring(file_path)

    def _start_monitoring(self, path: str):
        if self.worker:
            self.worker.stop()
            self.worker.wait()

        for key in self.metrics_history:
            self.metrics_history[key] = []
        self.console.clear()

        self.status_info.setText(f"Monitoring: {os.path.basename(path)}")

        self.worker = LogParserThread(path)
        self.worker.metrics_received.connect(self._on_metrics_update)
        self.worker.log_received.connect(self.log_buffer.append)
        self.worker.start()

    def _on_metrics_update(self, data: Dict[str, Any]):
        if data['type'] == 'train':
            self.metrics_history['train_epochs'].append(data['epoch'])
            self.metrics_history['train_loss'].append(data['loss'])
            self.metrics_history['train_recon'].append(data['recon'])
            self.metrics_history['train_kl'].append(data['kl'])

            self.card_epoch.update_metric(str(data['epoch']), "Active Training")
            self.card_loss.update_metric(f"{data['loss']:.2f}")
            self.card_recon.update_metric(f"{data['recon']:.2f}")
            self.card_kl.update_metric(f"{data['kl']:.2f}")

            self.curve_train_loss.setData(self.metrics_history['train_epochs'], self.metrics_history['train_loss'])
            self.curve_recon.setData(self.metrics_history['train_epochs'], self.metrics_history['train_recon'])
            self.curve_kl.setData(self.metrics_history['train_epochs'], self.metrics_history['train_kl'])

        elif data['type'] == 'val':
            self.metrics_history['val_epochs'].append(data['epoch'])
            self.metrics_history['val_loss'].append(data['loss'])
            self.curve_val_loss.setData(self.metrics_history['val_epochs'], self.metrics_history['val_loss'])

    def _flush_log_buffer(self):
        """Fixes 'messy' logs by using clear block inserts and preventing line sticking."""
        if not self.log_buffer:
            return

        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        while self.log_buffer:
            line = self.log_buffer.popleft()
            color = COLORS['text_secondary']

            # Simple color highlighting for visibility
            if "Val Loss" in line or "New best" in line:
                color = COLORS['success']
            elif "Epoch" in line:
                color = COLORS['info']
            elif "WARNING" in line or "ERROR" in line:
                color = COLORS['error']

            # Use append() or insertBlock() to ensure new lines are distinct
            self.console.append(f"<span style='color:{color};'>{line}</span>")

        # Memory management: truncate old lines
        doc = self.console.document()
        if doc.blockCount() > CONFIG['MAX_LOG_HISTORY']:
            diff = doc.blockCount() - CONFIG['MAX_LOG_HISTORY']
            cleanup_cursor = QTextCursor(doc.findBlockByNumber(0))
            for _ in range(diff):
                cleanup_cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cleanup_cursor.removeSelectedText()
                cleanup_cursor.deleteChar()

        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
