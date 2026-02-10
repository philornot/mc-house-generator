"""
Real-time Training Monitor for VAE Models.

Features:
- Tooltip snapping to the nearest data point.
- Interactive tooltips with data lookup.
- Custom context menus (Removed "Link Axis").
- Real-time density tracking.
"""

import glob
import os
import re
import sys
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
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
    'WINDOW_WIDTH': 1500,
    'WINDOW_HEIGHT': 950
}

COLORS = {
    'background': '#0B0E14',
    'surface': '#151921',
    'border': '#252A34',
    'primary': '#7C4DFF',
    'success': '#00E676',
    'info': '#00B0FF',
    'error': '#FF5252',
    'warning': '#FFAB40',
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
# CUSTOM INTERACTIVE PLOT WIDGET
# ==============================================================================

class CustomPlotWidget(pg.PlotWidget):
    """
    Customized PlotWidget that handles:
    1. Tooltips snapping to the closest data point.
    2. Removal of 'Link Axis' from the context menu.
    """

    def __init__(self, title=None, **kwargs):
        super().__init__(**kwargs)
        if title:
            self.setTitle(title)

        self.showGrid(x=True, y=True, alpha=0.2)

        # Tooltip elements
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFFFFF55', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#FFFFFF55', width=1))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        self.vLine.hide()
        self.hLine.hide()

        self.label = pg.TextItem(anchor=(0, 1), color=COLORS['text_primary'], fill=(20, 20, 25, 220))
        self.addItem(self.label, ignoreBounds=True)
        self.label.hide()

        # Connect mouse events
        self.scene().sigMouseMoved.connect(self.mouseMoved)

        # Remove "Link Axis" from Context Menu
        vb = self.getViewBox()
        actions = vb.menu.actions()
        for action in actions:
            if "Link axis" in action.text():
                vb.menu.removeAction(action)

        # Data references for snapping
        self.stored_x = []
        self.stored_y = []

    def set_snap_data(self, x_data, y_data):
        """Update the data used for tooltip snapping."""
        self.stored_x = x_data
        self.stored_y = y_data

    def mouseMoved(self, evt):
        if not self.stored_x:
            return

        pos = evt
        if self.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.getViewBox().mapSceneToView(pos)
            x_mouse = mousePoint.x()

            # Find closest index in stored_x
            idx = np.searchsorted(self.stored_x, x_mouse)
            if idx > 0 and (idx == len(self.stored_x) or abs(x_mouse - self.stored_x[idx - 1]) < abs(
                    x_mouse - self.stored_x[idx])):
                idx = idx - 1

            if 0 <= idx < len(self.stored_x):
                target_x = self.stored_x[idx]
                target_y = self.stored_y[idx]

                # Update lines to snap to point
                self.vLine.setPos(target_x)
                self.hLine.setPos(target_y)
                self.vLine.show()
                self.hLine.show()

                # Format label
                is_log_y = self.getPlotItem().getViewBox().state.get('logMode', [False, False])[1]
                val_display = 10 ** target_y if is_log_y else target_y

                self.label.setText(f"Epoch: {target_x}\nValue: {val_display:.4f}")
                self.label.setPos(target_x, target_y)
                self.label.show()
        else:
            self.vLine.hide()
            self.hLine.hide()
            self.label.hide()


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

        self._re_train = re.compile(
            r"Epoch\s+(?P<epoch>\d+)\s*-\s*Train Loss:\s*(?P<loss>[\d.]+)\s*\(Recon:\s*(?P<recon>[\d.]+),\s*KL:\s*(?P<kl>[\d.]+)"
        )
        self._re_val = re.compile(
            r"Epoch\s+(?P<epoch>\d+)\s*-\s*Val Loss:\s*(?P<loss>[\d.]+)"
        )
        self._re_sample = re.compile(
            r"Density:\s*(?P<density>[\d.]+)%"
        )

    def run(self):
        while self._is_running:
            if not os.path.exists(self.log_path):
                self.msleep(500)
                continue

            try:
                with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self._file_offset)
                    while True:
                        line = f.readline()
                        if not line: break

                        clean_line = line.strip()
                        if clean_line:
                            self.log_received.emit(clean_line)
                            self._parse_metrics(clean_line)
                    self._file_offset = f.tell()
            except Exception as e:
                print(f"Error: {e}")

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
        elif match := self._re_sample.search(line):
            data = match.groupdict()
            self.metrics_received.emit({
                'type': 'sample',
                'density': float(data['density'])
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
        self.setMinimumWidth(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel(label.upper())
        title.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")

        self.value_label = QLabel("N/A")
        self.value_label.setStyleSheet(f"color: {accent_color}; font-size: 22px; font-weight: 800;")

        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(12)
        self.shadow.setColor(QColor(accent_color))
        self.shadow.setOffset(0, 0)
        self.value_label.setGraphicsEffect(self.shadow)

        layout.addWidget(title)
        layout.addWidget(self.value_label)

    def update_metric(self, value: str):
        self.value_label.setText(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional VAE Training Monitor")
        self.resize(CONFIG['WINDOW_WIDTH'], CONFIG['WINDOW_HEIGHT'])
        self.setStyleSheet(APP_STYLESHEET)

        self.metrics_history = {
            'train_epochs': [], 'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_epochs': [], 'val_loss': [],
            'density_steps': [], 'density_vals': []
        }
        self.current_epoch = 0
        self.worker: Optional[LogParserThread] = None
        self.log_buffer = deque()

        self._setup_layout()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._flush_log_buffer)
        self.refresh_timer.start(50)

        latest_log = self._get_latest_log_path()
        if latest_log:
            self._start_monitoring(latest_log)

    def _setup_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Header
        header = QHBoxLayout()
        title_container = QVBoxLayout()
        main_title = QLabel("VAE TRAINING DASHBOARD")
        main_title.setStyleSheet("font-size: 18px; font-weight: 800; letter-spacing: 2px;")
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

        # Metrics
        cards_layout = QHBoxLayout()
        self.card_epoch = MetricDisplayCard("Epoch", COLORS['info'])
        self.card_loss = MetricDisplayCard("Total Loss", COLORS['success'])
        self.card_recon = MetricDisplayCard("Recon", COLORS['info'])
        self.card_kl = MetricDisplayCard("KL", COLORS['error'])
        self.card_density = MetricDisplayCard("Density", COLORS['warning'])

        for card in [self.card_epoch, self.card_loss, self.card_recon, self.card_kl, self.card_density]:
            cards_layout.addWidget(card)
        main_layout.addLayout(cards_layout)

        # Plots
        pg.setConfigOption('background', COLORS['surface'])
        pg.setConfigOption('foreground', COLORS['text_secondary'])
        pg.setConfigOptions(antialias=True)

        plots_layout = QHBoxLayout()

        # Plot 1: Loss
        self.loss_plot = CustomPlotWidget(title="Loss Trends (Log Scale)")
        self.loss_plot.setLogMode(y=True)
        self.curve_train_loss = self.loss_plot.plot(pen=pg.mkPen(COLORS['success'], width=2))
        self.curve_val_loss = self.loss_plot.plot(pen=pg.mkPen(COLORS['info'], width=2, style=Qt.PenStyle.DashLine))

        # Plot 2: Components
        self.comp_plot = CustomPlotWidget(title="VAE Components")
        self.comp_plot.setLogMode(y=True)
        self.curve_recon = self.comp_plot.plot(pen=pg.mkPen(COLORS['info'], width=2))
        self.curve_kl = self.comp_plot.plot(pen=pg.mkPen(COLORS['error'], width=2))

        # Plot 3: Density
        self.density_plot = CustomPlotWidget(title="Generated Sample Density (%)")
        self.curve_density = self.density_plot.plot(pen=pg.mkPen(COLORS['warning'], width=2), symbol='o', symbolSize=5)

        plots_layout.addWidget(self.loss_plot)
        plots_layout.addWidget(self.comp_plot)
        plots_layout.addWidget(self.density_plot)
        main_layout.addLayout(plots_layout, stretch=3)

        # Console
        self.console = QTextEdit()
        self.console.setObjectName("LogConsole")
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        main_layout.addWidget(self.console, stretch=1)

    def _get_latest_log_path(self) -> Optional[str]:
        if not os.path.exists(CONFIG['LOG_DIR']): return None
        pattern = os.path.join(CONFIG['LOG_DIR'], CONFIG['LOG_PATTERN'])
        files = glob.glob(pattern)
        return max(files, key=os.path.getmtime) if files else None

    def _on_select_file(self):
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
            self.current_epoch = data['epoch']
            self.metrics_history['train_epochs'].append(data['epoch'])
            self.metrics_history['train_loss'].append(data['loss'])
            self.metrics_history['train_recon'].append(data['recon'])
            self.metrics_history['train_kl'].append(data['kl'])

            self.card_epoch.update_metric(str(data['epoch']))
            self.card_loss.update_metric(f"{data['loss']:.2f}")
            self.card_recon.update_metric(f"{data['recon']:.2f}")
            self.card_kl.update_metric(f"{data['kl']:.2f}")

            # Convert to log scale for snap data if needed
            y_loss = np.log10(self.metrics_history['train_loss'])
            y_recon = np.log10(self.metrics_history['train_recon'])

            self.curve_train_loss.setData(self.metrics_history['train_epochs'], self.metrics_history['train_loss'])
            self.curve_recon.setData(self.metrics_history['train_epochs'], self.metrics_history['train_recon'])
            self.curve_kl.setData(self.metrics_history['train_epochs'], self.metrics_history['train_kl'])

            # Update Snap Data for tooltips
            self.loss_plot.set_snap_data(self.metrics_history['train_epochs'], y_loss)
            self.comp_plot.set_snap_data(self.metrics_history['train_epochs'], y_recon)

        elif data['type'] == 'val':
            self.metrics_history['val_epochs'].append(data['epoch'])
            self.metrics_history['val_loss'].append(data['loss'])
            self.curve_val_loss.setData(self.metrics_history['val_epochs'], self.metrics_history['val_loss'])

        elif data['type'] == 'sample':
            self.metrics_history['density_steps'].append(self.current_epoch)
            self.metrics_history['density_vals'].append(data['density'])
            self.card_density.update_metric(f"{data['density']:.4f}%")
            self.curve_density.setData(self.metrics_history['density_steps'], self.metrics_history['density_vals'])

            # Snap data for density plot
            self.density_plot.set_snap_data(self.metrics_history['density_steps'], self.metrics_history['density_vals'])

    def _flush_log_buffer(self):
        if not self.log_buffer: return
        while self.log_buffer:
            line = self.log_buffer.popleft()
            color = COLORS['text_secondary']

            if "Val Loss" in line:
                color = COLORS['success']
            elif "Epoch" in line:
                color = COLORS['info']
            elif "Physics" in line or "Connect" in line:
                color = COLORS['warning']
            elif "New best" in line:
                color = COLORS['primary']

            self.console.append(f"<span style='color:{color};'>{line}</span>")

        doc = self.console.document()
        if doc.blockCount() > CONFIG['MAX_LOG_HISTORY']:
            cursor = QTextCursor(doc.findBlockByNumber(0))
            for _ in range(doc.blockCount() - CONFIG['MAX_LOG_HISTORY']):
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
