"""
Desktop Application Module (PyQt5)
Professional dashboard-style GUI for regression model training and evaluation.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

import qtawesome as qta
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QLineEdit, QTextEdit,
    QFileDialog, QGroupBox, QGridLayout, QTabWidget, QMessageBox,
    QSplitter, QFrame, QStatusBar, QAction, QMenuBar, QSizePolicy,
    QProgressBar, QScrollArea, QPlainTextEdit, QDialog, QCheckBox,
    QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QLinearGradient,
    QBrush, QPainter, QPixmap
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ml_pipeline import MLPipeline
from ai_predictor import AIPredictor

logger = logging.getLogger(__name__)

# ──────────────── Color Constants ────────────────
DEEP_GREEN = "#1B5E20"
PRIMARY_GREEN = "#2E7D32"
LIGHT_GREEN = "#4CAF50"
ACCENT_GREEN = "#66BB6A"
SURFACE_GREEN = "#E8F5E9"
BG_COLOR = "#F5F7FA"
CARD_BG = "#FFFFFF"
TEXT_PRIMARY = "#212121"
TEXT_SECONDARY = "#616161"
BORDER_COLOR = "#E0E0E0"


# ──────────── Stylesheet ────────────
STYLESHEET = f"""
QMainWindow {{
    background-color: {BG_COLOR};
}}
QGroupBox {{
    font-weight: bold;
    font-size: 13px;
    color: {DEEP_GREEN};
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 18px;
    background-color: {CARD_BG};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}}
QPushButton {{
    background-color: {PRIMARY_GREEN};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 13px;
    font-weight: bold;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {LIGHT_GREEN};
}}
QPushButton:pressed {{
    background-color: {DEEP_GREEN};
}}
QPushButton:disabled {{
    background-color: #B0BEC5;
    color: #78909C;
}}
QPushButton#dangerBtn {{
    background-color: #E53935;
}}
QPushButton#dangerBtn:hover {{
    background-color: #EF5350;
}}
QComboBox {{
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    background-color: white;
    min-height: 18px;
}}
QComboBox:focus {{
    border-color: {LIGHT_GREEN};
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QSpinBox {{
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    background-color: white;
    min-height: 18px;
}}
QSpinBox:focus {{
    border-color: {LIGHT_GREEN};
}}
QLineEdit {{
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    background-color: white;
    min-height: 18px;
}}
QLineEdit:focus {{
    border-color: {LIGHT_GREEN};
}}
QTextEdit, QPlainTextEdit {{
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 10px;
    font-size: 12px;
    background-color: white;
    font-family: 'Consolas', 'Courier New', monospace;
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {LIGHT_GREEN};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    background-color: {CARD_BG};
    padding: 6px;
}}
QTabBar::tab {{
    background-color: {SURFACE_GREEN};
    color: {TEXT_SECONDARY};
    border: none;
    padding: 12px 24px;
    margin-right: 3px;
    font-size: 13px;
    font-weight: 600;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background-color: {PRIMARY_GREEN};
    color: white;
}}
QTabBar::tab:hover:!selected {{
    background-color: {ACCENT_GREEN};
    color: white;
}}
QLabel#headerLabel {{
    font-size: 22px;
    font-weight: bold;
    color: {DEEP_GREEN};
    padding: 6px;
}}
QLabel#subHeaderLabel {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
    padding: 2px 6px;
}}
QLabel#metricLabel {{
    font-size: 28px;
    font-weight: bold;
    color: {PRIMARY_GREEN};
}}
QLabel#metricTitle {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
    font-weight: 600;
    text-transform: uppercase;
}}
QProgressBar {{
    border: none;
    border-radius: 6px;
    background-color: {SURFACE_GREEN};
    text-align: center;
    font-size: 11px;
    min-height: 12px;
    max-height: 12px;
}}
QProgressBar::chunk {{
    background-color: {LIGHT_GREEN};
    border-radius: 6px;
}}
QStatusBar {{
    background-color: {DEEP_GREEN};
    color: white;
    font-size: 12px;
    padding: 4px;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
"""


class ExperimentWorker(QThread):
    """Worker thread for running ML experiments."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, pipeline, filepath, target_col, history,
                 regressor, n_folds, output_path):
        super().__init__()
        self.pipeline = pipeline
        self.filepath = filepath
        self.target_col = target_col
        self.history = history
        self.regressor = regressor
        self.n_folds = n_folds
        self.output_path = output_path

    def run(self):
        try:
            self.progress.emit("Loading dataset...")
            self.pipeline.load_dataset(self.filepath)

            self.progress.emit("Generating features...")
            self.pipeline.generate_features(self.target_col, self.history)

            self.progress.emit(f"Training {self.regressor}...")
            metrics = self.pipeline.train_and_evaluate(self.regressor, self.n_folds)

            self.progress.emit("Saving results...")
            filename = os.path.basename(self.filepath)
            self.pipeline.save_results(self.output_path, filename,
                                       self.history, self.regressor)

            self.progress.emit("Done!")
            self.finished.emit(metrics)
        except Exception as e:
            self.error.emit(str(e))


class MultiExperimentWorker(QThread):
    """Worker thread for running multiple ML experiments in parallel."""
    finished_one = pyqtSignal(dict, int, int)  # metrics, completed, total
    finished_all = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, filepath, target_col, output_path, combos):
        super().__init__()
        self.filepath = filepath
        self.target_col = target_col
        self.output_path = output_path
        self.combos = combos 

    def run(self):
        import concurrent.futures
        total = len(self.combos)
        completed = 0
        
        def do_work(combo):
            reg, hist, fld = combo
            # Fresh pipeline instance per thread to avoid race conditions
            pipe = MLPipeline()
            pipe.load_dataset(self.filepath)
            pipe.generate_features(self.target_col, hist)
            metrics = pipe.train_and_evaluate(reg, fld)
            filename = os.path.basename(self.filepath)
            pipe.save_results(self.output_path, filename, hist, reg)
            return metrics
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_combo = {executor.submit(do_work, c): c for c in self.combos}
            for future in concurrent.futures.as_completed(future_to_combo):
                combo = future_to_combo[future]
                try:
                    metrics = future.result()
                    completed += 1
                    self.progress.emit(f"Completed {combo[0]} (H={combo[1]}, F={combo[2]})")
                    self.finished_one.emit(metrics, completed, total)
                except Exception as exc:
                    self.error.emit(f"Error in {combo[0]}: {exc}")
                    
        self.finished_all.emit()


class MultiRunDialog(QDialog):
    """Popup dialog to configure multiple parallel experiments."""
    def __init__(self, parent=None, regressors=None):
        super().__init__(parent)
        self.setWindowTitle("Multiple Run Experiment")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.regressors_group = QGroupBox("Select Regressors")
        vbox = QVBoxLayout()
        self.checkboxes = {}
        for reg in regressors:
            chk = QCheckBox(reg)
            chk.setChecked(True)
            self.checkboxes[reg] = chk
            vbox.addWidget(chk)
        self.regressors_group.setLayout(vbox)
        
        self.history_edit = QLineEdit("5, 10, 15")
        form_layout.addRow("History Sizes (comma-separated):", self.history_edit)
        
        self.folds_edit = QLineEdit("5")
        form_layout.addRow("CV Folds (comma-separated):", self.folds_edit)
        
        layout.addWidget(self.regressors_group)
        layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def get_values(self):
        selected_regs = [name for name, chk in self.checkboxes.items() if chk.isChecked()]
        try:
            histories = [int(x.strip()) for x in self.history_edit.text().split(",") if x.strip()]
            folds = [int(x.strip()) for x in self.folds_edit.text().split(",") if x.strip()]
        except ValueError:
            return None, None, None
        return selected_regs, histories, folds



class MetricCard(QFrame):
    """Custom widget for displaying a metric card."""

    def __init__(self, title, value="—", parent=None):
        super().__init__(parent)
        self.setFixedHeight(110)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
                padding: 10px;
            }}
            QFrame:hover {{
                border-color: {LIGHT_GREEN};
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(16, 12, 16, 12)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.title_label.setAlignment(Qt.AlignLeft)

        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("metricLabel")
        self.value_label.setAlignment(Qt.AlignLeft)

        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.value_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding plots."""

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor='white', edgecolor='none')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.subplots_adjust(bottom=0.18, left=0.12, right=0.95, top=0.90)


class DesktopApp(QMainWindow):
    """Main Desktop Application Window."""

    def __init__(self):
        super().__init__()
        self.pipeline = MLPipeline()
        self.ai_predictor = AIPredictor()
        self.results_file = os.path.join(os.path.dirname(__file__),
                                          "results", "experiments.csv")
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        self.setWindowTitle("Regression Dashboard")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._setup_statusbar()
        self.showMaximized()

    # ──────────── Build UI ────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 14, 18, 10)
        main_layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Regression Analysis with Machine Learning")
        title.setObjectName("headerLabel")
        sub = QLabel("Train • Evaluate • Visualize • Predict")
        sub.setObjectName("subHeaderLabel")
        header_layout.addWidget(title)
        header_layout.addWidget(sub)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_experiment_tab(), "Experiment")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.tabs.addTab(self._build_visualization_tab(), "Graphs")
        self.tabs.addTab(self._build_ai_tab(), "AI Predictor")
        main_layout.addWidget(self.tabs, 1)

    # ──────────── Experiment Tab ────────────
    def _build_experiment_tab(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(14)

        # Left: Configuration
        config_group = QGroupBox("  Experiment Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        config_layout.setContentsMargins(16, 24, 16, 16)
        row = 0

        # Input CSV
        config_layout.addWidget(QLabel("Input CSV File"), row, 0)
        file_row = QHBoxLayout()
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("Select or enter CSV file path...")
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.csv_path_edit, 1)
        file_row.addWidget(browse_btn)
        config_layout.addLayout(file_row, row, 1)
        row += 1

        # Target Variable
        config_layout.addWidget(QLabel("Target Variable"), row, 0)
        self.target_combo = QComboBox()
        self.target_combo.setEditable(True)
        self.target_combo.setPlaceholderText("Select after loading dataset...")
        config_layout.addWidget(self.target_combo, row, 1)
        row += 1

        # History Size
        config_layout.addWidget(QLabel("History Window Size"), row, 0)
        self.history_spin = QSpinBox()
        self.history_spin.setRange(1, 1000)
        self.history_spin.setValue(5)
        config_layout.addWidget(self.history_spin, row, 1)
        row += 1

        # Regressor
        config_layout.addWidget(QLabel("Regression Algorithm"), row, 0)
        self.regressor_combo = QComboBox()
        self.regressor_combo.addItems(list(MLPipeline.REGRESSORS.keys()))
        config_layout.addWidget(self.regressor_combo, row, 1)
        row += 1

        # CV Folds
        config_layout.addWidget(QLabel("Cross-Validation Folds"), row, 0)
        self.folds_spin = QSpinBox()
        self.folds_spin.setRange(2, 20)
        self.folds_spin.setValue(5)
        config_layout.addWidget(self.folds_spin, row, 1)
        row += 1

        # Output file
        config_layout.addWidget(QLabel("Output CSV Filename"), row, 0)
        self.output_edit = QLineEdit("results/experiments.csv")
        config_layout.addWidget(self.output_edit, row, 1)
        row += 1

        config_layout.setRowStretch(row, 1)
        layout.addWidget(config_group, 1)

        # Right: Controls & Log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        ctrl_group = QGroupBox("  Training Controls")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setSpacing(10)
        ctrl_layout.setContentsMargins(16, 24, 16, 16)

        self.load_btn = QPushButton(" Load Dataset")
        self.load_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        self.load_btn.clicked.connect(self._load_dataset)
        ctrl_layout.addWidget(self.load_btn)

        self.run_btn = QPushButton(" Run Experiment")
        self.run_btn.setIcon(qta.icon("fa5s.rocket", color="white"))
        self.run_btn.clicked.connect(self._run_experiment)
        self.run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {DEEP_GREEN}; font-size: 15px; padding: 14px; }}"
            f"QPushButton:hover {{ background-color: {PRIMARY_GREEN}; }}"
        )
        ctrl_layout.addWidget(self.run_btn)

        self.multi_run_btn = QPushButton(" Multiple Run Experiment")
        self.multi_run_btn.setIcon(qta.icon("fa5s.running", color="white"))
        self.multi_run_btn.clicked.connect(self._open_multi_run_dialog)
        self.multi_run_btn.setStyleSheet(
            f"QPushButton {{ background-color: {DEEP_GREEN}; font-size: 15px; padding: 14px; }}"
            f"QPushButton:hover {{ background-color: {LIGHT_GREEN}; }}"
        )
        ctrl_layout.addWidget(self.multi_run_btn)

        self.metrics_btn = QPushButton(" Show Metrics")
        self.metrics_btn.setIcon(qta.icon("fa5s.clipboard-list", color="white"))
        self.metrics_btn.clicked.connect(self._show_metrics)
        ctrl_layout.addWidget(self.metrics_btn)

        self.plot_btn = QPushButton(" Plot Results")
        self.plot_btn.setIcon(qta.icon("fa5s.chart-pie", color="white"))
        self.plot_btn.clicked.connect(self._plot_results)
        ctrl_layout.addWidget(self.plot_btn)

        self.save_btn = QPushButton(" Save Results")
        self.save_btn.setIcon(qta.icon("fa5s.save", color="white"))
        self.save_btn.clicked.connect(self._save_results_dialog)
        ctrl_layout.addWidget(self.save_btn)

        self.predict_btn = QPushButton(" Forecast Next Hour")
        self.predict_btn.setIcon(qta.icon("fa5s.forward", color="white"))
        self.predict_btn.clicked.connect(self._predict_next)
        self.predict_btn.setStyleSheet(
            f"QPushButton {{ background-color: {PRIMARY_GREEN}; font-size: 15px; padding: 14px; }}"
            f"QPushButton:hover {{ background-color: {LIGHT_GREEN}; }}"
        )
        ctrl_layout.addWidget(self.predict_btn)

        right_layout.addWidget(ctrl_group)

        log_group = QGroupBox("  Activity Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(10, 22, 10, 10)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(500)
        self.log_text.setPlaceholderText("Experiment activity will appear here...")
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, 1)

        layout.addWidget(right_widget, 1)
        return widget

    # ──────────── Results Tab ────────────
    def _build_results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)

        # Controls
        ctrl_row = QHBoxLayout()
        refresh_btn = QPushButton(" Refresh Results")
        refresh_btn.setIcon(qta.icon("fa5s.sync-alt", color="white"))
        refresh_btn.clicked.connect(self._load_results_table)
        ctrl_row.addWidget(refresh_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText(
            "No results yet. Run an experiment first.")
        layout.addWidget(self.results_text)
        return widget

    # ──────────── Visualization Tab ────────────
    def _build_visualization_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        ctrl_row = QHBoxLayout()
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "MAE Comparison",
            "MSE Comparison",
            "R² Score Comparison",
            "All Metrics Overview",
            "Actual vs Predicted"
        ])
        ctrl_row.addWidget(QLabel("Chart Type:"))
        ctrl_row.addWidget(self.chart_type_combo)
        plot_btn = QPushButton(" Generate Chart")
        plot_btn.setIcon(qta.icon("fa5s.chart-bar", color="white"))
        plot_btn.clicked.connect(self._generate_chart)
        ctrl_row.addWidget(plot_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self.canvas = MatplotlibCanvas(self, width=10, height=6)
        layout.addWidget(self.canvas, 1)
        return widget

    # ──────────── AI Predictor Tab ────────────
    def _build_ai_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Query
        query_group = QGroupBox("  Analysis Query")
        query_layout = QVBoxLayout(query_group)
        query_layout.setContentsMargins(14, 22, 14, 14)
        self.ai_query_edit = QTextEdit()
        self.ai_query_edit.setMaximumHeight(100)
        self.ai_query_edit.setPlaceholderText(
            "Ask the AI about your experiments...\n"
            "e.g. 'Why does Random Forest perform better than Linear Regression?'")
        query_layout.addWidget(self.ai_query_edit)

        btn_row = QHBoxLayout()
        analyze_btn = QPushButton("Analyze with AI")
        analyze_btn.clicked.connect(self._run_ai_analysis)
        btn_row.addWidget(analyze_btn)

        auto_btn = QPushButton(" Auto-Analyze Results")
        auto_btn.setIcon(qta.icon("fa5s.bolt", color="white"))
        auto_btn.clicked.connect(self._auto_ai_analysis)
        btn_row.addWidget(auto_btn)
        btn_row.addStretch()
        query_layout.addLayout(btn_row)
        layout.addWidget(query_group)

        # AI Response
        response_group = QGroupBox("  AI Response")
        response_layout = QVBoxLayout(response_group)
        response_layout.setContentsMargins(14, 22, 14, 14)
        self.ai_response_text = QTextEdit()
        self.ai_response_text.setReadOnly(True)
        self.ai_response_text.setPlaceholderText(
            "AI analysis results will appear here...")
        response_layout.addWidget(self.ai_response_text)
        layout.addWidget(response_group, 1)

        return widget

    # ──────────── Status Bar ────────────
    def _setup_statusbar(self):
        self.statusBar().showMessage("  Ready — Load a dataset to begin")

    # ──────────── Slots / Actions ────────────
    def _log(self, msg):
        self.log_text.appendPlainText(f"► {msg}")
        self.statusBar().showMessage(f"  {msg}")

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "CSV Files (*.csv);;All Files (*)")
        if path:
            self.csv_path_edit.setText(path)
            self._log(f"Selected file: {os.path.basename(path)}")
            self._load_dataset()

    def _load_dataset(self):
        path = self.csv_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first.")
            return
        try:
            df = self.pipeline.load_dataset(path)
            numeric_cols = self.pipeline.get_numeric_columns()
            self.target_combo.clear()
            self.target_combo.addItems(numeric_cols)
            self._log(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{e}")
            self._log(f"Error: {e}")

    def _run_experiment(self):
        filepath = self.csv_path_edit.text().strip()
        target_col = self.target_combo.currentText().strip()
        history = self.history_spin.value()
        regressor = self.regressor_combo.currentText()
        n_folds = self.folds_spin.value()
        output_path = self.output_edit.text().strip()

        if not filepath:
            QMessageBox.warning(self, "Warning", "Please select a CSV file.")
            return
        if not target_col:
            QMessageBox.warning(self, "Warning", "Please select a target variable.")
            return

        # Resolve output path relative to project
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(__file__), output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.results_file = output_path
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._log(f"Starting experiment: {regressor} | History={history} | "
                  f"Folds={n_folds}")

        self.worker = ExperimentWorker(
            self.pipeline, filepath, target_col, history,
            regressor, n_folds, output_path
        )
        self.worker.finished.connect(self._on_experiment_done)
        self.worker.error.connect(self._on_experiment_error)
        self.worker.progress.connect(lambda m: self._log(m))
        self.worker.start()

    def _open_multi_run_dialog(self):
        filepath = self.csv_path_edit.text().strip()
        target_col = self.target_combo.currentText().strip()
        output_path = self.output_edit.text().strip()
        
        if not filepath:
            QMessageBox.warning(self, "Warning", "Please select a CSV file first.")
            return
        if not target_col:
            QMessageBox.warning(self, "Warning", "Please select a target variable.")
            return

        dialog = MultiRunDialog(self, list(MLPipeline.REGRESSORS.keys()))
        dialog.setStyleSheet(STYLESHEET)
        if dialog.exec_():
            regs, histories, folds = dialog.get_values()
            if not regs or not histories or not folds:
                QMessageBox.warning(self, "Warning", "Invalid input. Please enter numbers separated by commas.")
                return
            
            # Resolve output path
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.path.dirname(__file__), output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.results_file = output_path
            
            combos = []
            for r in regs:
                for h in histories:
                    for f in folds:
                        combos.append((r, h, f))
                        
            if not combos:
                return

            self.run_btn.setEnabled(False)
            self.multi_run_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(combos))
            self.progress_bar.setValue(0)
            self._log(f"Starting multiple runs: {len(combos)} total experiments")

            self.multi_worker = MultiExperimentWorker(filepath, target_col, output_path, combos)
            self.multi_worker.finished_one.connect(self._on_multi_one_done)
            self.multi_worker.finished_all.connect(self._on_multi_all_done)
            self.multi_worker.error.connect(self._on_experiment_error)
            self.multi_worker.progress.connect(lambda m: self._log(m))
            self.multi_worker.start()

    def _on_multi_one_done(self, metrics, completed, total):
        self.progress_bar.setValue(completed)
        self._log(f"-> Batch Progress: {completed}/{total} completed")

    def _on_multi_all_done(self):
        self.run_btn.setEnabled(True)
        self.multi_run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)
        self._log("✅ All multiple parallel experiments completed!")
        QMessageBox.information(self, "Success", "All parallel experiments have completed successfully!\nYou can now see them in the Results tab.")

    def _on_experiment_done(self, metrics):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._log("✅ Experiment completed successfully!")
        QMessageBox.information(self, "Success",
                                f"Experiment completed!\n\n"
                                f"MAE: {metrics['MAE']:.4f}\n"
                                f"MSE: {metrics['MSE']:.4f}\n"
                                f"R² Score: {metrics['R2_Score']:.4f}")

    def _on_experiment_error(self, error_msg):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Experiment Error", error_msg)

    def _show_metrics(self):
        if not self.pipeline.metrics:
            QMessageBox.information(self, "Info",
                                    "No metrics yet. Run an experiment first.")
            return
        m = self.pipeline.metrics
        msg = (f"═══ Experiment Metrics ═══\n\n"
               f"MAE:  {m['MAE']:.4f}  (±{m.get('MAE_std', 0):.4f})\n"
               f"MSE:  {m['MSE']:.4f}  (±{m.get('MSE_std', 0):.4f})\n"
               f"R²:   {m['R2_Score']:.4f}  (±{m.get('R2_std', 0):.4f})\n")
        QMessageBox.information(self, "Experiment Metrics", msg)

    def _plot_results(self):
        self.tabs.setCurrentIndex(2)
        self._generate_chart()

    def _save_results_dialog(self):
        if not self.pipeline.metrics:
            QMessageBox.information(self, "Info",
                                    "No results to save. Run an experiment first.")
            return
        self._log("Results already saved to: " + self.results_file)
        QMessageBox.information(self, "Saved",
                                f"Results appended to:\n{self.results_file}")

    def _predict_next(self):
        if self.pipeline.model is None:
            QMessageBox.warning(self, "Warning", "Please run an experiment first to train a model.")
            return

        target_col = self.target_combo.currentText().strip()
        history = self.history_spin.value()
        
        try:
            next_val = self.pipeline.predict_next_value(target_col, history)
            self._log(f"🔮 Predicted next {target_col}: {next_val:.2f} MW")
            
            QMessageBox.information(
                self, 
                "Future Forecast", 
                f"Based on the trained {self.pipeline.model.__class__.__name__} "
                f"and the last {history} hours of real data,\n\n"
                f"The forecasted next {target_col} is:\n"
                f"{next_val:.2f} MW"
            )
        except Exception as e:
            self._log(f"Error predicting next value: {e}")
            QMessageBox.critical(self, "Error", f"Could not predict next value:\n{e}")

    def _load_results_table(self):
        try:
            df = self.pipeline.load_results(self.results_file)
            if df.empty:
                self.results_text.setHtml(
                    "<p style='color:#999;'>No experiment results found.</p>")
                return
            # Build HTML table
            html = """
            <style>
                table { border-collapse: collapse; width: 100%; font-family: 'Segoe UI', sans-serif; }
                th { background-color: #2E7D32; color: white; padding: 12px 16px; text-align: left; }
                td { padding: 10px 16px; border-bottom: 1px solid #E0E0E0; }
                tr:nth-child(even) { background-color: #F5F5F5; }
                tr:hover { background-color: #E8F5E9; }
            </style>
            <table>
            <tr>"""
            for col in df.columns:
                html += f"<th>{col}</th>"
            html += "</tr>"
            for _, row in df.iterrows():
                html += "<tr>"
                for val in row:
                    html += f"<td>{val}</td>"
                html += "</tr>"
            html += "</table>"
            self.results_text.setHtml(html)
            self._log(f"Results loaded: {len(df)} experiments")
        except Exception as e:
            self._log(f"Error loading results: {e}")

    def _generate_chart(self):
        # Disconnect any old hover events
        if hasattr(self, '_hover_cid') and self._hover_cid:
            self.canvas.mpl_disconnect(self._hover_cid)
            self._hover_cid = None

        chart_type = self.chart_type_combo.currentText()
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        # Tooltip annotation setup
        self.annot = ax.annotate("", xy=(0,0), xytext=(15, 15),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=DEEP_GREEN, lw=1.5, alpha=0.95),
                                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color=DEEP_GREEN))
        self.annot.set_visible(False)
        self.annot.set_zorder(100)

        if chart_type == "Actual vs Predicted":
            actual, predicted = self.pipeline.get_predictions()
            if actual is None:
                self._log("No predictions available. Run an experiment.")
                return
            
            sc = ax.scatter(actual, predicted, alpha=0.6, s=25,
                       color=LIGHT_GREEN, edgecolors=DEEP_GREEN, linewidth=0.5)
            
            lims = [min(min(actual), min(predicted)),
                    max(max(actual), max(predicted))]
            ax.plot(lims, lims, '--', color='#E53935', linewidth=1.5,
                    label='Perfect Prediction')
            ax.set_xlabel('Actual Values', fontsize=11)
            ax.set_ylabel('Predicted Values', fontsize=11)
            ax.set_title('Actual vs Predicted', fontsize=14, fontweight='bold',
                         color=DEEP_GREEN)
            ax.legend()
            ax.grid(True, alpha=0.3)

            def hover(event):
                vis = self.annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        pos = sc.get_offsets()[idx]
                        self.annot.xy = pos
                        self.annot.set_text(f"Actual: {actual[idx]:.2f}\nPred: {predicted[idx]:.2f}")
                        self.annot.set_visible(True)
                        self.canvas.draw_idle()
                        return
                if vis:
                    self.annot.set_visible(False)
                    self.canvas.draw_idle()
            
            self._hover_cid = self.canvas.mpl_connect("motion_notify_event", hover)

        else:
            df = self.pipeline.load_results(self.results_file)
            if df.empty:
                self._log("No experiment results to chart.")
                return

            colors = ['#1B5E20', '#4CAF50', '#81C784', '#A5D6A7',
                      '#C8E6C9', '#388E3C', '#66BB6A']
            bar_colors = [colors[i % len(colors)] for i in range(len(df))]
            labels = [f"{r}\n(H={h})" for r, h in zip(df['Regressor'], df['History'])]
            
            bar_data = [] # List of (bar_patch, hover_text)

            if chart_type == "MAE Comparison":
                bars = ax.bar(range(len(df)), df['MAE'], color=bar_colors, edgecolor='white', linewidth=1.5)
                ax.set_ylabel('MAE', fontsize=12)
                ax.set_title('MAE Comparison', fontsize=14, fontweight='bold', color=DEEP_GREEN)
                for bar, val in zip(bars, df['MAE']):
                    bar_data.append((bar, f"MAE: {val:.4f}"))
                    
            elif chart_type == "MSE Comparison":
                bars = ax.bar(range(len(df)), df['MSE'], color=bar_colors, edgecolor='white', linewidth=1.5)
                ax.set_ylabel('MSE', fontsize=12)
                ax.set_title('MSE Comparison', fontsize=14, fontweight='bold', color=DEEP_GREEN)
                for bar, val in zip(bars, df['MSE']):
                    bar_data.append((bar, f"MSE: {val:.4f}"))
                    
            elif chart_type == "R² Score Comparison":
                bars = ax.bar(range(len(df)), df['R2_Score'], color=bar_colors, edgecolor='white', linewidth=1.5)
                ax.set_ylabel('R² Score', fontsize=12)
                ax.set_title('R² Score Comparison', fontsize=14, fontweight='bold', color=DEEP_GREEN)
                for bar, val in zip(bars, df['R2_Score']):
                    bar_data.append((bar, f"R²: {val:.4f}"))
                    
            else:  # All Metrics overview
                x = list(range(len(df)))
                w = 0.25
                mae_max = df['MAE'].max()
                mse_max = df['MSE'].max()
                mae_norm = df['MAE'] / mae_max if mae_max != 0 else df['MAE']
                mse_norm = df['MSE'] / mse_max if mse_max != 0 else df['MSE']
                
                bars_mae = ax.bar([i - w for i in x], mae_norm, w, label='MAE (normalized)', color='#1B5E20', edgecolor='white')
                bars_r2 = ax.bar(x, df['R2_Score'], w, label='R² Score', color='#4CAF50', edgecolor='white')
                bars_mse = ax.bar([i + w for i in x], mse_norm, w, label='MSE (normalized)', color='#81C784', edgecolor='white')
                
                ax.set_ylabel('Normalized Score', fontsize=12)
                ax.set_title('All Metrics Overview', fontsize=14, fontweight='bold', color=DEEP_GREEN)
                ax.legend()
                
                for i in range(len(df)):
                    bar_data.append((bars_mae[i], f"Real MAE: {df['MAE'].iloc[i]:.4f}"))
                    bar_data.append((bars_r2[i], f"Real R²: {df['R2_Score'].iloc[i]:.4f}"))
                    bar_data.append((bars_mse[i], f"Real MSE: {df['MSE'].iloc[i]:.4f}"))

            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

            def hover_bar(event):
                vis = self.annot.get_visible()
                if event.inaxes == ax:
                    for bar, text in bar_data:
                        cont, _ = bar.contains(event)
                        if cont:
                            x = bar.get_x() + bar.get_width() / 2.
                            y = bar.get_y() + bar.get_height()
                            self.annot.xy = (x, y)
                            self.annot.set_text(text)
                            self.annot.set_visible(True)
                            self.canvas.draw_idle()
                            return
                if vis:
                    self.annot.set_visible(False)
                    self.canvas.draw_idle()

            self._hover_cid = self.canvas.mpl_connect("motion_notify_event", hover_bar)

        self.canvas.fig.tight_layout()
        self.canvas.draw()
        self._log(f"Chart generated: {chart_type}")

    def _run_ai_analysis(self):
        query = self.ai_query_edit.toPlainText().strip()
        experiment_data = self._get_experiment_data_for_ai()
        self.ai_response_text.setPlainText("⏳ Analyzing...")
        QApplication.processEvents()
        response = self.ai_predictor.analyze_results(experiment_data, query)
        self.ai_response_text.setPlainText(response)
        self._log("AI analysis completed")

    def _auto_ai_analysis(self):
        experiment_data = self._get_experiment_data_for_ai()
        self.ai_response_text.setPlainText("⏳ Running auto-analysis...")
        QApplication.processEvents()
        response = self.ai_predictor.analyze_results(experiment_data)
        self.ai_response_text.setPlainText(response)
        self._log("Auto AI analysis completed")

    def _get_experiment_data_for_ai(self):
        df = self.pipeline.load_results(self.results_file)
        if df.empty and self.pipeline.metrics:
            return self.pipeline.metrics
        if not df.empty:
            return df.to_dict('records')
        return {"message": "No experiment data available yet. Please run an experiment first."}


def launch_desktop():
    """Launch the PyQt5 desktop application."""
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = DesktopApp()
    window.show()
    sys.exit(app.exec_())
