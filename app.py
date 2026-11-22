import sys
import random
import json
import numpy as np
from collections import deque
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QFrame, QSlider, QMenu,
                               QFileDialog, QInputDialog, QGroupBox,
                               QGridLayout, QCheckBox, QComboBox,
                               QTabWidget, QSplitter, QStatusBar,
                               QScrollArea, QSizePolicy, QToolButton,
                               QButtonGroup, QRadioButton, QProgressBar,
                               QMessageBox)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QFont, QColor, QPalette, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.gridspec import GridSpec


# ------------------------------------------------------------------ presets
PRESETS = {
    "Heart":      [0.0, 0.8, 0.0, -0.5, 0.0, 0.3, 0.0, -0.15],
    "Circle":     [1.0, 0.0, -0.5, 0.0, 0.0, 1.0, 0.0, -0.5],
    "Figure-8":   [0.0, 1.0, 0.0, -0.8, 0.0, 0.0, 0.5, 0.0],
    "Star":       [0.0, 1.2, 0.0, -1.5, 0.0, 0.8, 0.0, -0.3],
    "Spiral":     [0.0, 0.3, 0.0, 0.5, 0.0, 0.7, 0.0, 0.9],
    "Wave":       [0.0, 1.0, 0.0, -0.33, 0.0, 0.2, 0.0, -0.14],
    "Butterfly":  [0.0, 0.6, 0.3, -0.4, 0.2, 0.5, -0.3, -0.2],
    "Knot":       [0.5, 0.7, -0.3, 0.9, -0.5, 0.4, 0.2, -0.6],
    "Lissajous":  [0.0, 1.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0],
    "Rose":       [0.0, 0.7, 0.0, -0.9, 0.0, 0.4, 0.0, -0.2],
    "Trefoil":    [0.2, 0.8, -0.4, 0.6, 0.3, -0.7, 0.5, -0.3],
    "Infinity":   [0.0, 1.0, 0.0, -1.0, 0.0, 0.5, 0.0, -0.25],
}

WORDS = [
    "hello", "world", "python", "math", "curve", "plot",
    "fractal", "chaos", "beauty", "code", "art", "magic",
    "dream", "spark", "glow", "flow", "wave", "star",
]

AUTO_MODES = ["Random", "Word", "Pattern", "Emoji Storm"]


# ----------------------------------------------------------------- helpers
class StatusLED(QLabel):
    """A small colored circle that indicates on/off state."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._on = False
        self._on_color = "#4caf50"
        self._off_color = "#888888"
        self._update()

    def set_on(self, on):
        self._on = on
        self._update()

    def set_colors(self, on_color, off_color):
        self._on_color = on_color
        self._off_color = off_color
        self._update()

    def _update(self):
        color = self._on_color if self._on else self._off_color
        self.setStyleSheet(
            f"background-color: {color}; border-radius: 6px; "
            f"border: 1px solid rgba(0,0,0,0.2);"
        )


class CoefficientBar(QFrame):
    """A small horizontal bar chart showing current coefficients."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setMinimumWidth(200)
        self.coeffs = []
        self._bars = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        for _ in range(12):
            bar = QFrame(self)
            bar.setFrameShape(QFrame.StyledPanel)
            bar.setMinimumWidth(8)
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(bar)
            self._bars.append(bar)
        self.setStyleSheet("CoefficientBar { background: rgba(0,0,0,0.05); border-radius: 4px; }")

    def set_coefficients(self, coeffs, max_coeffs=12):
        self.coeffs = coeffs
        for i, bar in enumerate(self._bars):
            if i < len(coeffs):
                c = coeffs[i]
                if c >= 0:
                    intensity = int(min(c, 1.0) * 200)
                    color = f"rgb({intensity}, {int(intensity*0.3)}, {int(intensity*0.3)})"
                else:
                    intensity = int(min(abs(c), 1.0) * 200)
                    color = f"rgb({int(intensity*0.3)}, {int(intensity*0.3)}, {intensity})"
                bar.setStyleSheet(
                    f"background-color: {color}; border-radius: 2px; "
                    f"min-height: {int(abs(c) * 30) + 4}px;"
                )
                bar.setToolTip(f"c_{i} = {c:+.2f}")
            else:
                bar.setStyleSheet("background-color: transparent;")
                bar.setToolTip("")


class ActionButton(QPushButton):
    """Styled action button with consistent sizing."""
    def __init__(self, text, tooltip="", parent=None):
        super().__init__(text, parent)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(32)


class ToggleButton(QPushButton):
    """A toggle button with LED indicator."""
    def __init__(self, text, led_color="#4caf50", parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(32)
        self.led = StatusLED(self)
        self.led.set_colors(led_color, "#888888")
        self.toggled.connect(self.led.set_on)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(10, 4, 10, 4)
        self._layout.addStretch()
        self._layout.addWidget(self.led)


# -------------------------------------------------------------------- main
class PolynomialVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keypress Parametric Polynomial Visualizer")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        self.setFocusPolicy(Qt.StrongFocus)

        # State
        self.coeffs = []
        self.max_coeffs = 12
        self.total_chars_typed = 0
        self.auto_typing = False
        self.auto_mode = "Random"
        self.dark_mode = False
        self.show_derivative = False
        self.show_fft = False
        self.show_trails = True
        self.lock_axes = True
        self.trail_length = 6
        self.trail_history = deque(maxlen=self.trail_length)
        self.word_index = 0
        self.char_index = 0
        self.pattern_phase = 0.0
        self.line_width = 3.0

        # Character pools
        self.basic_chars = list("abcdefghijklmnopqrstuvwxyz0123456789!@#$%&*+-=<>?")
        self.emoji_chars = [
            "😀", "😎", "🤔", "😴", "🥳", "😱", "🤯", "😂",
            "🐶", "🐱", "🐼", "🦊", "🐸", "🦄", "🐙", "🦋",
            "🚀", "🌍", "🔥", "⭐", "🎨", "🎵", "🎮", "💻",
            "🍕", "🍎", "🍩", "🍔", "🍦", "🌮", "🍣", "🍪",
            "❤", "⚡", "✨", "🌈", "🌸", "🌊", "☀", "🌙",
        ]

        # Timers
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(100)

        self.setup_ui()
        self.update_plots()

    # ------------------------------------------------------------------ UI
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self._setup_menu()

        # Main splitter: canvas (left) + side panel (right)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(2)

        # --- LEFT: Canvas + info strip ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Canvas with GridSpec for 3 subplots
        self.figure = Figure(figsize=(12, 8), facecolor="#fafafa")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        gs = GridSpec(2, 2, figure=self.figure, height_ratios=[1, 1])
        self.ax1d = self.figure.add_subplot(gs[0, 0])
        self.ax2d = self.figure.add_subplot(gs[0, 1])
        self.ax_fft = self.figure.add_subplot(gs[1, :])

        # Info strip below canvas
        info_strip = self._build_info_strip()

        left_layout.addWidget(self.canvas, stretch=1)
        left_layout.addWidget(info_strip)

        # --- RIGHT: Control side panel ---
        right_widget = self._build_side_panel()

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1000, 360])

        main_layout.addWidget(self.splitter, stretch=1)

        # Status bar
        self.statusBar().showMessage(
            "Press any key to add a coefficient  |  "
            "Backspace = remove  |  Esc = clear  |  "
            "Click 2D plot to add coefficient"
        )

        self._apply_theme()

    def _build_info_strip(self):
        """Compact info strip showing current state."""
        strip = QFrame()
        strip.setObjectName("infoStrip")
        strip.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # Last key
        key_box = QVBoxLayout()
        key_box.setSpacing(2)
        key_box.addWidget(self._make_field_label("Last Key"))
        self.key_label = QLabel("-")
        self.key_label.setObjectName("fieldValue")
        key_box.addWidget(self.key_label)
        layout.addLayout(key_box)

        layout.addWidget(self._make_separator())

        # Count
        count_box = QVBoxLayout()
        count_box.setSpacing(2)
        count_box.addWidget(self._make_field_label("Count"))
        self.count_label = QLabel("0")
        self.count_label.setObjectName("fieldValue")
        count_box.addWidget(self.count_label)
        layout.addLayout(count_box)

        layout.addWidget(self._make_separator())

        # Coefficient bars
        bar_box = QVBoxLayout()
        bar_box.setSpacing(2)
        bar_box.addWidget(self._make_field_label("Coefficients"))
        self.coeff_bar = CoefficientBar()
        bar_box.addWidget(self.coeff_bar)
        layout.addLayout(bar_box, stretch=1)

        layout.addWidget(self._make_separator())

        # Polynomial
        poly_box = QVBoxLayout()
        poly_box.setSpacing(2)
        poly_box.addWidget(self._make_field_label("Polynomial"))
        self.poly_label = QLabel("P(t) = 0")
        self.poly_label.setObjectName("fieldValue")
        poly_box.addWidget(self.poly_label)
        layout.addLayout(poly_box, stretch=1)

        layout.addWidget(self._make_separator())

        # Stats
        stats_box = QVBoxLayout()
        stats_box.setSpacing(2)
        stats_box.addWidget(self._make_field_label("Arc Length"))
        self.stats_label = QLabel("0.00")
        self.stats_label.setObjectName("fieldValue")
        stats_box.addWidget(self.stats_label)
        layout.addLayout(stats_box)

        layout.addWidget(self._make_separator())

        # Clear button
        clear_btn = ActionButton("Clear", "Clear all coefficients (Esc)")
        clear_btn.clicked.connect(self.clear)
        layout.addWidget(clear_btn)

        return strip

    def _make_field_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("fieldLabel")
        lbl.setStyleSheet("font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;")
        return lbl

    def _make_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedWidth(1)
        return sep

    def _build_side_panel(self):
        """Right side panel with tabbed controls."""
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title
        title = QLabel("Controls")
        title.setObjectName("panelTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        auto_tab = self._build_auto_tab()
        self.tabs.addTab(auto_tab, "Auto-Type")

        display_tab = self._build_display_tab()
        self.tabs.addTab(display_tab, "Display")

        presets_tab = self._build_presets_tab()
        self.tabs.addTab(presets_tab, "Presets")

        actions_tab = self._build_actions_tab()
        self.tabs.addTab(actions_tab, "Actions")

        layout.addWidget(self.tabs, stretch=1)

        footer = self._build_footer()
        layout.addWidget(footer)

        return panel

    def _build_auto_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.auto_btn = ToggleButton("Auto-Type: OFF", "#4caf50")
        self.auto_btn.toggled.connect(self.toggle_auto_type)
        layout.addWidget(self.auto_btn)

        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(AUTO_MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        self.mode_combo.setToolTip("Select auto-typing mode")
        mode_layout.addWidget(self.mode_combo)

        mode_desc = QLabel(
            "Random: Mix of all characters\n"
            "Word: Type from dictionary\n"
            "Pattern: Smooth sine waves\n"
            "Emoji Storm: Emojis only"
        )
        mode_desc.setObjectName("helpText")
        mode_desc.setWordWrap(True)
        mode_layout.addWidget(mode_desc)
        layout.addWidget(mode_group)

        speed_group = QGroupBox("Speed")
        speed_layout = QVBoxLayout(speed_group)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Interval:"))
        self.speed_value_label = QLabel("200 ms")
        self.speed_value_label.setObjectName("monoLabel")
        self.speed_value_label.setAlignment(Qt.AlignRight)
        speed_row.addWidget(self.speed_value_label)
        speed_layout.addLayout(speed_row)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(30, 1000)
        self.speed_slider.setValue(200)
        self.speed_slider.valueChanged.connect(self._update_speed)
        self.speed_slider.setToolTip("Time between auto-typed keys (ms)")
        speed_layout.addWidget(self.speed_slider)

        preset_row = QHBoxLayout()
        for label, ms in [("Fast", 60), ("Normal", 200), ("Slow", 500)]:
            btn = QPushButton(label)
            btn.setCheckable(False)
            btn.clicked.connect(lambda checked, m=ms: self.speed_slider.setValue(m))
            preset_row.addWidget(btn)
        speed_layout.addLayout(preset_row)

        layout.addWidget(speed_group)
        layout.addStretch()

        return tab

    def _build_display_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.deriv_check = QCheckBox("Show derivative P'(t)")
        self.deriv_check.setToolTip("Overlay the derivative curve on the 1D plot")
        self.deriv_check.stateChanged.connect(self._toggle_derivative)
        layout.addWidget(self.deriv_check)

        self.fft_check = QCheckBox("Show Fourier Transform")
        self.fft_check.setToolTip("Show the FFT magnitude spectrum of the 1D polynomial")
        self.fft_check.stateChanged.connect(self._toggle_fft)
        layout.addWidget(self.fft_check)

        self.trail_check = QCheckBox("Show ghost trails")
        self.trail_check.setChecked(True)
        self.trail_check.setToolTip("Show fading history of previous curves")
        self.trail_check.stateChanged.connect(self._toggle_trails)
        layout.addWidget(self.trail_check)

        self.lock_axes_check = QCheckBox("Lock axis limits")
        self.lock_axes_check.setChecked(True)
        self.lock_axes_check.setToolTip("Keep fixed axis limits instead of auto-scaling")
        self.lock_axes_check.stateChanged.connect(self._toggle_lock_axes)
        layout.addWidget(self.lock_axes_check)

        self.dark_check = QCheckBox("Dark mode")
        self.dark_check.setToolTip("Toggle dark theme for UI and plots")
        self.dark_check.stateChanged.connect(self._toggle_dark_mode)
        layout.addWidget(self.dark_check)

        trail_group = QGroupBox("Trail Length")
        trail_layout = QVBoxLayout(trail_group)
        trail_row = QHBoxLayout()
        trail_row.addWidget(QLabel("Frames:"))
        self.trail_value_label = QLabel("6")
        self.trail_value_label.setObjectName("monoLabel")
        self.trail_value_label.setAlignment(Qt.AlignRight)
        trail_row.addWidget(self.trail_value_label)
        trail_layout.addLayout(trail_row)

        self.trail_slider = QSlider(Qt.Horizontal)
        self.trail_slider.setRange(1, 15)
        self.trail_slider.setValue(6)
        self.trail_slider.valueChanged.connect(self._update_trail_length)
        trail_layout.addWidget(self.trail_slider)
        layout.addWidget(trail_group)

        lw_group = QGroupBox("Line Width")
        lw_layout = QVBoxLayout(lw_group)
        lw_row = QHBoxLayout()
        lw_row.addWidget(QLabel("Thickness:"))
        self.lw_value_label = QLabel("3.0")
        self.lw_value_label.setObjectName("monoLabel")
        self.lw_value_label.setAlignment(Qt.AlignRight)
        lw_row.addWidget(self.lw_value_label)
        lw_layout.addLayout(lw_row)

        self.lw_slider = QSlider(Qt.Horizontal)
        self.lw_slider.setRange(10, 60)
        self.lw_slider.setValue(30)
        self.lw_slider.valueChanged.connect(self._update_line_width)
        lw_layout.addWidget(self.lw_slider)
        layout.addWidget(lw_group)

        layout.addStretch()
        return tab

    def _build_presets_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        desc = QLabel("Load a preset curve shape:")
        desc.setObjectName("helpText")
        layout.addWidget(desc)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        grid = QGridLayout(scroll_widget)
        grid.setSpacing(6)

        cols = 2
        for i, (name, _) in enumerate(PRESETS.items()):
            btn = QPushButton(name)
            btn.setMinimumHeight(36)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Load the '{name}' preset curve")
            btn.clicked.connect(lambda checked, n=name: self._load_preset(n))
            grid.addWidget(btn, i // cols, i % cols)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, stretch=1)

        rand_btn = ActionButton("Randomize", "Generate random coefficients")
        rand_btn.clicked.connect(self._randomize)
        layout.addWidget(rand_btn)

        return tab

    def _build_actions_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        edit_group = QGroupBox("Edit")
        edit_layout = QVBoxLayout(edit_group)

        rev_btn = ActionButton("Reverse", "Reverse coefficient order (Ctrl+R)")
        rev_btn.clicked.connect(self._reverse_coeffs)
        edit_layout.addWidget(rev_btn)

        scale_btn = ActionButton("Scale...", "Multiply all coefficients by a factor (Ctrl+K)")
        scale_btn.clicked.connect(self._scale_coeffs)
        edit_layout.addWidget(scale_btn)

        neg_btn = ActionButton("Negate", "Flip sign of all coefficients (Ctrl+N)")
        neg_btn.clicked.connect(self._negate_coeffs)
        edit_layout.addWidget(neg_btn)

        layout.addWidget(edit_group)

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        save_btn = ActionButton("Save Coefficients...", "Save to JSON file (Ctrl+S)")
        save_btn.clicked.connect(self._save_coefficients)
        file_layout.addWidget(save_btn)

        load_btn = ActionButton("Load Coefficients...", "Load from JSON file (Ctrl+O)")
        load_btn.clicked.connect(self._load_coefficients)
        file_layout.addWidget(load_btn)

        export_btn = ActionButton("Export PNG...", "Save plot as image (Ctrl+E)")
        export_btn.clicked.connect(self._export_png)
        file_layout.addWidget(export_btn)

        layout.addWidget(file_group)

        layout.addStretch()
        return tab

    def _build_footer(self):
        """Footer with status LEDs."""
        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(10, 6, 10, 6)

        layout.addWidget(QLabel("Status:"))

        self.led_auto = StatusLED()
        self.led_auto.set_colors("#4caf50", "#555555")
        layout.addWidget(self.led_auto)
        layout.addWidget(QLabel("Auto"))

        layout.addSpacing(10)

        self.led_trail = StatusLED()
        self.led_trail.set_on(True)
        self.led_trail.set_colors("#2196f3", "#555555")
        layout.addWidget(self.led_trail)
        layout.addWidget(QLabel("Trails"))

        layout.addSpacing(10)

        self.led_deriv = StatusLED()
        self.led_deriv.set_colors("#ff9800", "#555555")
        layout.addWidget(self.led_deriv)
        layout.addWidget(QLabel("P'(t)"))

        layout.addSpacing(10)

        self.led_fft = StatusLED()
        self.led_fft.set_colors("#9c27b0", "#555555")
        layout.addWidget(self.led_fft)
        layout.addWidget(QLabel("FFT"))

        layout.addSpacing(10)

        self.led_dark = StatusLED()
        self.led_dark.set_colors("#9c27b0", "#555555")
        layout.addWidget(self.led_dark)
        layout.addWidget(QLabel("Dark"))

        layout.addStretch()

        ver = QLabel("v2.2")
        ver.setObjectName("versionLabel")
        layout.addWidget(ver)

        return footer

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        save_act = QAction("Save Coefficients...", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self._save_coefficients)
        file_menu.addAction(save_act)

        load_act = QAction("Load Coefficients...", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self._load_coefficients)
        file_menu.addAction(load_act)

        file_menu.addSeparator()
        export_act = QAction("Export PNG...", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._export_png)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = menubar.addMenu("Edit")
        clear_act = QAction("Clear", self)
        clear_act.setShortcut("Esc")
        clear_act.triggered.connect(self.clear)
        edit_menu.addAction(clear_act)

        reverse_act = QAction("Reverse Coefficients", self)
        reverse_act.setShortcut("Ctrl+R")
        reverse_act.triggered.connect(self._reverse_coeffs)
        edit_menu.addAction(reverse_act)

        scale_act = QAction("Scale Coefficients...", self)
        scale_act.setShortcut("Ctrl+K")
        scale_act.triggered.connect(self._scale_coeffs)
        edit_menu.addAction(scale_act)

        negate_act = QAction("Negate Coefficients", self)
        negate_act.setShortcut("Ctrl+N")
        negate_act.triggered.connect(self._negate_coeffs)
        edit_menu.addAction(negate_act)

        rand_act = QAction("Randomize", self)
        rand_act.setShortcut("Ctrl+D")
        rand_act.triggered.connect(self._randomize)
        edit_menu.addAction(rand_act)

        view_menu = menubar.addMenu("View")
        dark_act = QAction("Toggle Dark Mode", self)
        dark_act.setShortcut("Ctrl+Shift+D")
        dark_act.triggered.connect(lambda: self.dark_check.toggle())
        view_menu.addAction(dark_act)

        help_menu = menubar.addMenu("Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

        shortcuts_act = QAction("Keyboard Shortcuts", self)
        shortcuts_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_act)

    # ----------------------------------------------------------- key input
    def keyPressEvent(self, event):
        text = event.text()
        key  = event.key()
        self._process_key(text, key, source="manual")

    def _process_key(self, text, key, source="manual"):
        if key == Qt.Key_Backspace:
            if self.coeffs:
                self.coeffs.pop()
            self._save_to_trail()
            self.refresh_labels()
            self.update_plots()
            return
        elif key == Qt.Key_Escape:
            self.clear()
            return

        if text:
            # Use first character to handle multi-char emojis safely
            code = ord(text[0])
            value = ((code % 200) - 100) / 100.0

            if len(self.coeffs) >= self.max_coeffs:
                self.coeffs.pop(0)
            self.coeffs.append(value)
            self.total_chars_typed += 1

            display_char = text if text.isprintable() else f"U+{code:04X}"
            prefix = "[AUTO] " if source == "auto" else ""
            self.key_label.setText(
                f"{prefix}'{display_char}' ({value:+.2f})"
            )

            self._save_to_trail()
            self.refresh_labels()
            self.update_plots()

    def _save_to_trail(self):
        if self.coeffs and self.show_trails:
            self.trail_history.append(list(self.coeffs))

    def refresh_labels(self):
        # Show total chars typed + currently active count
        self.count_label.setText(f"{self.total_chars_typed} ({len(self.coeffs)})")
        self.coeff_bar.set_coefficients(self.coeffs, self.max_coeffs)
        if self.coeffs:
            self.poly_label.setText(self.format_polynomial())
        else:
            self.poly_label.setText("P(t) = 0")

    def format_polynomial(self):
        if not self.coeffs:
            return "P(t) = 0"
        terms = []
        for i, c in enumerate(self.coeffs):
            c_str = f"{c:.2f}"
            if i == 0:
                terms.append(c_str)
            elif i == 1:
                terms.append(f"{c_str}*t")
            else:
                terms.append(f"{c_str}*t^{i}")
        poly = " + ".join(terms)
        poly = poly.replace("+ -", "- ")
        return f"P(t) = {poly}"

    # -------------------------------------------------------- math helpers
    @staticmethod
    def evaluate(coeffs, t):
        result = np.zeros_like(t, dtype=float)
        for i, c in enumerate(coeffs):
            result = result + c * (t ** i)
        return result

    @staticmethod
    def evaluate_derivative(coeffs, t):
        result = np.zeros_like(t, dtype=float)
        for i, c in enumerate(coeffs):
            if i >= 1:
                result = result + i * c * (t ** (i - 1))
        return result

    @staticmethod
    def compute_arc_length(x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt(dx**2 + dy**2))

    # ---------------------------------------------------------- rendering
    def update_plots(self):
        self.ax1d.clear()
        self.ax2d.clear()
        self.ax_fft.clear()

        t = np.linspace(-1.0, 1.0, 600)

        if self.dark_mode:
            bg_color = "#1e1e1e"
            text_color = "#e0e0e0"
            grid_color = "#444444"
            axis_color = "#888888"
            main_color = "#4fc3f7"
            fill_color = "#4fc3f7"
            deriv_color = "#ff8a65"
            self.figure.set_facecolor(bg_color)
        else:
            bg_color = "#fafafa"
            text_color = "#333333"
            grid_color = "#dddddd"
            axis_color = "#666666"
            main_color = "#2ca02c"
            fill_color = "#2ca02c"
            deriv_color = "#d62728"
            self.figure.set_facecolor(bg_color)

        # --- 1D plot ---
        self.ax1d.set_facecolor(bg_color)
        self.ax1d.set_title("1D: P(t)", fontsize=13, fontweight="bold", color=text_color)
        self.ax1d.set_xlabel("t in [-1, 1]", color=text_color)
        self.ax1d.set_ylabel("P(t)", color=text_color)
        self.ax1d.tick_params(colors=text_color)
        self.ax1d.grid(True, alpha=0.3, linestyle="--", color=grid_color)
        self.ax1d.axhline(0, color=axis_color, lw=0.8)
        self.ax1d.axvline(0, color=axis_color, lw=0.8)

        if self.coeffs:
            y = self.evaluate(self.coeffs, t)
            self.ax1d.plot(t, y, color=main_color, lw=2.5, label="P(t)")
            self.ax1d.fill_between(t, 0, y, color=fill_color, alpha=0.15)

            if self.show_derivative:
                dy = self.evaluate_derivative(self.coeffs, t)
                self.ax1d.plot(t, dy, color=deriv_color, lw=1.8,
                              linestyle="--", label="P'(t)")
                self.ax1d.legend(loc="best", fontsize=9, facecolor=bg_color,
                                edgecolor=axis_color, labelcolor=text_color)

            if self.lock_axes:
                self.ax1d.set_ylim(-2.0, 2.0)
            else:
                ymin, ymax = y.min(), y.max()
                if self.show_derivative:
                    dy = self.evaluate_derivative(self.coeffs, t)
                    ymin = min(ymin, dy.min())
                    ymax = max(ymax, dy.max())
                margin = max((ymax - ymin) * 0.15, 0.1)
                self.ax1d.set_ylim(ymin - margin, ymax + margin)
        else:
            self.ax1d.set_xlim(-1, 1)
            self.ax1d.set_ylim(-2.0, 2.0)
            self.ax1d.text(0, 0, "press any key...",
                           ha="center", va="center", color=axis_color, fontsize=14)

        # --- 2D parametric plot ---
        self.ax2d.set_facecolor(bg_color)
        self.ax2d.set_title("2D: (x(t), y(t))", fontsize=13, fontweight="bold", color=text_color)
        self.ax2d.set_xlabel("x(t)", color=text_color)
        self.ax2d.set_ylabel("y(t)", color=text_color)
        self.ax2d.tick_params(colors=text_color)
        self.ax2d.set_aspect("equal")
        self.ax2d.grid(True, alpha=0.3, linestyle="--", color=grid_color)
        self.ax2d.axhline(0, color=axis_color, lw=0.8)
        self.ax2d.axvline(0, color=axis_color, lw=0.8)

        if self.coeffs:
            cx = self.coeffs[0::2]
            cy = self.coeffs[1::2] or [0.0]

            x  = self.evaluate(cx, t)
            y2 = self.evaluate(cy, t)

            # Trails (convert deque to list for slicing)
            if self.show_trails and len(self.trail_history) > 1:
                trail_list = list(self.trail_history)
                for i, trail_coeffs in enumerate(trail_list[:-1]):
                    alpha = 0.1 + 0.3 * (i / len(trail_list))
                    tcx = trail_coeffs[0::2]
                    tcy = trail_coeffs[1::2] or [0.0]
                    tx = self.evaluate(tcx, t)
                    ty = self.evaluate(tcy, t)
                    self.ax2d.plot(tx, ty, color=main_color, alpha=alpha, lw=1.5)

            # Current curve
            points = np.column_stack([x, y2]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = cm.plasma if not self.dark_mode else cm.viridis
            lc = LineCollection(segments, cmap=cmap, linewidth=self.line_width)
            lc.set_array(t)
            self.ax2d.add_collection(lc)

            self.ax2d.plot(x[0],  y2[0],  "o", color="cyan",  ms=10, mec='k',
                           label="t = -1.0", zorder=5)
            self.ax2d.plot(x[-1], y2[-1], "s", color="magenta", ms=10, mec='k',
                           label="t = +1.0", zorder=5)

            if self.lock_axes:
                self.ax2d.set_xlim(-2.0, 2.0)
                self.ax2d.set_ylim(-2.0, 2.0)
            else:
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y2.min(), y2.max()
                pad = max((xmax - xmin) * 0.15, 0.1)
                pady = max((ymax - ymin) * 0.15, 0.1)
                self.ax2d.set_xlim(xmin - pad, xmax + pad)
                self.ax2d.set_ylim(ymin - pady, ymax + pady)

            self.ax2d.legend(loc="best", fontsize=9, facecolor=bg_color,
                            edgecolor=axis_color, labelcolor=text_color)
        else:
            self.ax2d.set_xlim(-2.0, 2.0)
            self.ax2d.set_ylim(-2.0, 2.0)
            self.ax2d.text(0, 0, "press any key...",
                           ha="center", va="center", color=axis_color, fontsize=14)

        # --- Fourier Transform plot ---
        self.ax_fft.set_facecolor(bg_color)
        self.ax_fft.set_title("Fourier Transform Magnitude of P(t)", fontsize=11, fontweight="bold", color=text_color)
        self.ax_fft.set_xlabel("Frequency", color=text_color)
        self.ax_fft.set_ylabel("Magnitude", color=text_color)
        self.ax_fft.tick_params(colors=text_color)
        self.ax_fft.grid(True, alpha=0.3, linestyle="--", color=grid_color)

        if self.show_fft:
            if self.coeffs:
                y = self.evaluate(self.coeffs, t)
                N = len(t)
                dt = t[1] - t[0]
                yf = np.fft.fft(y)
                xf = np.fft.fftfreq(N, d=dt)
                
                half_N = N // 2
                xf_pos = xf[:half_N]
                yf_pos = 2.0 / N * np.abs(yf[:half_N])
                
                self.ax_fft.plot(xf_pos, yf_pos, color=main_color, lw=1.5)
                max_freq = min(100.0, xf_pos[-1])
                self.ax_fft.set_xlim(0, max_freq)
            else:
                self.ax_fft.text(0.5, 0.5, "Type keys to generate\nFourier Transform",
                                 ha="center", va="center", color=axis_color, fontsize=12)
                self.ax_fft.set_xticks([])
                self.ax_fft.set_yticks([])
        else:
            self.ax_fft.text(0.5, 0.5, "Fourier Transform hidden\n(Enable in Display tab)",
                             ha="center", va="center", color=axis_color, fontsize=12)
            self.ax_fft.set_xticks([])
            self.ax_fft.set_yticks([])

        self.figure.tight_layout()
        self.canvas.draw()

    # --------------------------------------------------------- interactions
    def _on_canvas_click(self, event):
        if event.inaxes == self.ax2d and event.button == 1:
            x_val = event.xdata
            y_val = event.ydata
            value = np.clip(np.sqrt(x_val**2 + y_val**2) * 0.5, -1.0, 1.0)
            if len(self.coeffs) >= self.max_coeffs:
                self.coeffs.pop(0)
            self.coeffs.append(value)
            self.total_chars_typed += 1
            self.key_label.setText(f"Click ({value:+.2f})")
            self._save_to_trail()
            self.refresh_labels()
            self.update_plots()

    # --------------------------------------------------------------- misc
    def clear(self):
        self.coeffs = []
        self.total_chars_typed = 0
        self.trail_history.clear()
        self.key_label.setText("-")
        self.refresh_labels()
        self.update_plots()

    def _reverse_coeffs(self):
        if self.coeffs:
            self.coeffs.reverse()
            self._save_to_trail()
            self.refresh_labels()
            self.update_plots()

    def _scale_coeffs(self):
        if not self.coeffs:
            return
        factor, ok = QInputDialog.getDouble(self, "Scale Coefficients",
                                            "Enter scale factor:", 1.0, -10.0, 10.0, 2)
        if ok:
            self.coeffs = [c * factor for c in self.coeffs]
            self._save_to_trail()
            self.refresh_labels()
            self.update_plots()

    def _negate_coeffs(self):
        if self.coeffs:
            self.coeffs = [-c for c in self.coeffs]
            self._save_to_trail()
            self.refresh_labels()
            self.update_plots()

    def _randomize(self):
        n = random.randint(4, 12)
        self.coeffs = [random.uniform(-1, 1) for _ in range(n)]
        self.trail_history.clear()
        self.key_label.setText("Random")
        self.refresh_labels()
        self.update_plots()

    def _load_preset(self, name):
        if name in PRESETS:
            self.coeffs = list(PRESETS[name])
            self.trail_history.clear()
            self.key_label.setText(f"Preset: {name}")
            self.refresh_labels()
            self.update_plots()

    def _save_coefficients(self):
        if not self.coeffs:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Coefficients",
                                                "coefficients.json", "JSON (*.json)")
        if fname:
            data = {"coefficients": self.coeffs, "timestamp": datetime.now().isoformat()}
            with open(fname, 'w') as f:
                json.dump(data, f, indent=2)
            self.statusBar().showMessage(f"Saved to {fname}", 3000)

    def _load_coefficients(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Coefficients",
                                                "", "JSON (*.json)")
        if fname:
            try:
                with open(fname, 'r') as f:
                    data = json.load(f)
                self.coeffs = data.get("coefficients", [])
                self.trail_history.clear()
                self.refresh_labels()
                self.update_plots()
                self.statusBar().showMessage(f"Loaded from {fname}", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"Error loading: {e}", 3000)

    def _export_png(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Export PNG",
                                                "plot.png", "PNG (*.png)")
        if fname:
            self.figure.savefig(fname, dpi=150, bbox_inches='tight')
            self.statusBar().showMessage(f"Exported to {fname}", 3000)

    def _show_about(self):
        QMessageBox.about(self, "About",
            "Keypress Parametric Polynomial Visualizer v2.2\n\n"
            "Type keys to build polynomial coefficients.\n"
            "Each key maps to a value in [-1, 1].\n\n"
            "Features:\n"
            "- 1D polynomial plot P(t)\n"
            "- 2D parametric plot (x(t), y(t))\n"
            "- Fourier Transform magnitude of P(t)\n"
            "- Auto-typing with multiple modes\n"
            "- Ghost trails showing history\n"
            "- Derivative overlay P'(t)\n"
            "- 12 preset patterns\n"
            "- Dark mode\n"
            "- Lock axis limits (no jumping)\n"
            "- Export to PNG\n"
            "- Save/load coefficients\n"
            "- Total character counter\n"
            "- Coefficient visualization bars")

    def _show_shortcuts(self):
        QMessageBox.information(self, "Keyboard Shortcuts",
            "Ctrl+S           Save coefficients\n"
            "Ctrl+O           Load coefficients\n"
            "Ctrl+E           Export PNG\n"
            "Ctrl+R           Reverse coefficients\n"
            "Ctrl+K           Scale coefficients\n"
            "Ctrl+N           Negate coefficients\n"
            "Ctrl+D           Randomize\n"
            "Ctrl+Shift+D     Toggle dark mode\n"
            "Ctrl+Q           Quit\n"
            "Esc              Clear all\n"
            "Backspace        Remove last\n"
            "Click on 2D plot to add coefficient")

    # --------------------------------------------------------- auto-typing
    def toggle_auto_type(self, checked):
        self.auto_typing = checked
        self.led_auto.set_on(checked)
        if checked:
            self.auto_btn.setText("Auto-Type: ON")
            self._update_speed(self.speed_slider.value())
            self.auto_timer.start()
        else:
            self.auto_btn.setText("Auto-Type: OFF")
            self.auto_timer.stop()

    def _on_mode_change(self, mode):
        self.auto_mode = mode
        self.word_index = 0
        self.char_index = 0

    def _update_speed(self, value):
        self.speed_value_label.setText(f"{value} ms")
        if self.auto_typing:
            self.auto_timer.setInterval(value)

    def _update_trail_length(self, value):
        self.trail_value_label.setText(str(value))
        self.trail_length = value
        self.trail_history = deque(maxlen=value)

    def _update_line_width(self, value):
        lw = value / 10.0
        self.lw_value_label.setText(f"{lw:.1f}")
        self.line_width = lw
        self.update_plots()

    def _auto_tick(self):
        char = self._get_next_auto_char()
        self._process_key(char, 0, source="auto")

    def _get_next_auto_char(self):
        if self.auto_mode == "Random":
            pool = self.basic_chars + self.emoji_chars
            return random.choice(pool)
        elif self.auto_mode == "Word":
            word = WORDS[self.word_index % len(WORDS)]
            char = word[self.char_index % len(word)]
            self.char_index += 1
            if self.char_index >= len(word):
                self.char_index = 0
                self.word_index += 1
            return char
        elif self.auto_mode == "Pattern":
            self.pattern_phase += 0.1
            value = np.sin(self.pattern_phase)
            code = int((value + 1) * 100) % 200 + 50
            return chr(code)
        elif self.auto_mode == "Emoji Storm":
            return random.choice(self.emoji_chars)
        return random.choice(self.basic_chars)

    # ----------------------------------------------------------- toggles
    def _toggle_derivative(self, state):
        self.show_derivative = (state == Qt.Checked)
        self.led_deriv.set_on(self.show_derivative)
        self.update_plots()

    def _toggle_fft(self, state):
        self.show_fft = (state == Qt.Checked)
        self.led_fft.set_on(self.show_fft)
        self.update_plots()

    def _toggle_trails(self, state):
        self.show_trails = (state == Qt.Checked)
        self.led_trail.set_on(self.show_trails)
        if not self.show_trails:
            self.trail_history.clear()
        self.update_plots()

    def _toggle_lock_axes(self, state):
        self.lock_axes = (state == Qt.Checked)
        self.update_plots()

    def _toggle_dark_mode(self, state):
        self.dark_mode = (state == Qt.Checked)
        self.led_dark.set_on(self.dark_mode)
        self._apply_theme()
        self.update_plots()

    def _apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow { background: #1e1e1e; }
                QFrame#infoStrip {
                    background: #252525;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                }
                QFrame#sidePanel {
                    background: #252525;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                }
                QFrame#footer {
                    background: #1a1a1a;
                    border: 1px solid #3a3a3a;
                    border-radius: 6px;
                }
                QLabel#panelTitle {
                    font-size: 14px;
                    font-weight: bold;
                    color: #e0e0e0;
                    padding: 4px;
                }
                QLabel#fieldLabel {
                    color: #888888;
                    font-size: 10px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                QLabel#fieldValue {
                    color: #e0e0e0;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 13px;
                    font-weight: bold;
                }
                QLabel#monoLabel {
                    font-family: 'Consolas', 'Courier New', monospace;
                    color: #4fc3f7;
                }
                QLabel#helpText {
                    color: #888888;
                    font-size: 11px;
                }
                QLabel#versionLabel {
                    color: #555555;
                    font-size: 10px;
                }
                QGroupBox {
                    background: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 14px;
                    color: #e0e0e0;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #4fc3f7;
                }
                QLabel, QCheckBox, QComboBox, QPushButton {
                    color: #e0e0e0;
                }
                QPushButton {
                    background: #333333;
                    border: 1px solid #444444;
                    padding: 6px 14px;
                    border-radius: 4px;
                }
                QPushButton:hover { background: #3d3d3d; border: 1px solid #4fc3f7; }
                QPushButton:pressed { background: #2a2a2a; }
                QPushButton:checked { background: #1b5e20; color: white; border: 1px solid #4caf50; }
                QSlider::groove:horizontal { background: #3a3a3a; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #4fc3f7; width: 16px; margin: -5px 0; border-radius: 8px; }
                QSlider::handle:horizontal:hover { background: #81d4fa; }
                QComboBox { background: #333333; border: 1px solid #444444; padding: 5px; border-radius: 4px; }
                QComboBox:hover { border: 1px solid #4fc3f7; }
                QTabWidget::pane { border: 1px solid #3a3a3a; background: #252525; border-radius: 4px; }
                QTabBar::tab { background: #2a2a2a; color: #888888; padding: 8px 16px; border: 1px solid #3a3a3a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                QTabBar::tab:selected { background: #333333; color: #4fc3f7; border-bottom: 2px solid #4fc3f7; }
                QTabBar::tab:hover { background: #333333; color: #e0e0e0; }
                QMenuBar { background: #1a1a1a; color: #e0e0e0; border-bottom: 1px solid #3a3a3a; }
                QMenuBar::item:selected { background: #333333; }
                QMenu { background: #252525; color: #e0e0e0; border: 1px solid #3a3a3a; }
                QMenu::item:selected { background: #333333; color: #4fc3f7; }
                QStatusBar { background: #1a1a1a; color: #888888; border-top: 1px solid #3a3a3a; }
                QScrollArea { border: none; background: transparent; }
                QScrollBar:vertical { background: #1a1a1a; width: 10px; border-radius: 5px; }
                QScrollBar::handle:vertical { background: #444444; border-radius: 5px; min-height: 20px; }
                QScrollBar::handle:vertical:hover { background: #4fc3f7; }
                QSplitter::handle { background: #3a3a3a; }
                QSplitter::handle:hover { background: #4fc3f7; }
                QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #444444; border-radius: 3px; background: #333333; }
                QCheckBox::indicator:checked { background: #4fc3f7; border: 1px solid #4fc3f7; }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow { background: #f5f5f5; }
                QFrame#infoStrip {
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                }
                QFrame#sidePanel {
                    background: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                }
                QFrame#footer {
                    background: #f9f9f9;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                }
                QLabel#panelTitle {
                    font-size: 14px;
                    font-weight: bold;
                    color: #333333;
                    padding: 4px;
                }
                QLabel#fieldLabel {
                    color: #888888;
                    font-size: 10px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                QLabel#fieldValue {
                    color: #333333;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-size: 13px;
                    font-weight: bold;
                }
                QLabel#monoLabel {
                    font-family: 'Consolas', 'Courier New', monospace;
                    color: #1976d2;
                }
                QLabel#helpText {
                    color: #888888;
                    font-size: 11px;
                }
                QLabel#versionLabel {
                    color: #aaaaaa;
                    font-size: 10px;
                }
                QGroupBox {
                    background: #fafafa;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 14px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: #1976d2;
                    font-weight: bold;
                }
                QPushButton {
                    background: white;
                    border: 1px solid #d0d0d0;
                    padding: 6px 14px;
                    border-radius: 4px;
                }
                QPushButton:hover { background: #f0f7ff; border: 1px solid #1976d2; color: #1976d2; }
                QPushButton:pressed { background: #e3f2fd; }
                QPushButton:checked { background: #4caf50; color: white; border: 1px solid #388e3c; }
                QSlider::groove:horizontal { background: #e0e0e0; height: 6px; border-radius: 3px; }
                QSlider::handle:horizontal { background: #1976d2; width: 16px; margin: -5px 0; border-radius: 8px; }
                QSlider::handle:horizontal:hover { background: #42a5f5; }
                QComboBox { background: white; border: 1px solid #d0d0d0; padding: 5px; border-radius: 4px; }
                QComboBox:hover { border: 1px solid #1976d2; }
                QTabWidget::pane { border: 1px solid #e0e0e0; background: white; border-radius: 4px; }
                QTabBar::tab { background: #f5f5f5; color: #666666; padding: 8px 16px; border: 1px solid #e0e0e0; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
                QTabBar::tab:selected { background: white; color: #1976d2; border-bottom: 2px solid #1976d2; }
                QTabBar::tab:hover { background: #f0f7ff; color: #1976d2; }
                QMenuBar { background: white; color: #333333; border-bottom: 1px solid #e0e0e0; }
                QMenuBar::item:selected { background: #f0f7ff; color: #1976d2; }
                QMenu { background: white; color: #333333; border: 1px solid #e0e0e0; }
                QMenu::item:selected { background: #f0f7ff; color: #1976d2; }
                QStatusBar { background: #f9f9f9; color: #666666; border-top: 1px solid #e0e0e0; }
                QScrollArea { border: none; background: transparent; }
                QScrollBar:vertical { background: #f5f5f5; width: 10px; border-radius: 5px; }
                QScrollBar::handle:vertical { background: #d0d0d0; border-radius: 5px; min-height: 20px; }
                QScrollBar::handle:vertical:hover { background: #1976d2; }
                QSplitter::handle { background: #e0e0e0; }
                QSplitter::handle:hover { background: #1976d2; }
                QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #d0d0d0; border-radius: 3px; background: white; }
                QCheckBox::indicator:checked { background: #1976d2; border: 1px solid #1976d2; }
            """)

    # -------------------------------------------------------------- stats
    def _update_stats(self):
        if self.coeffs:
            t = np.linspace(-1.0, 1.0, 600)
            cx = self.coeffs[0::2]
            cy = self.coeffs[1::2] or [0.0]
            x = self.evaluate(cx, t)
            y = self.evaluate(cy, t)
            arc_len = self.compute_arc_length(x, y)
            self.stats_label.setText(f"{arc_len:.2f}")
        else:
            self.stats_label.setText("0.00")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = PolynomialVisualizer()
    win.show()
    sys.exit(app.exec())