import sys
import os
import logging

# ──────── Logging Setup ────────
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def show_startup_dialog():
    """Show PyQt5 startup dialog to choose application mode."""
    from PyQt5.QtWidgets import (
        QApplication, QDialog, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect
    )
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap
    import qtawesome as qta

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    dialog = QDialog()
    dialog.setWindowTitle("Ontario Demand Analysis")
    dialog.setFixedSize(700, 500)
    dialog.setStyleSheet("""
        QDialog {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #E8F5E9, stop:0.5 #FFFFFF, stop:1 #E8F5E9
            );
        }
    """)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(36, 30, 36, 30)
    layout.setSpacing(12)

    # Title Icon
    icon_label = QLabel()
    icon_label.setAlignment(Qt.AlignCenter)
    icon_pixmap = qta.icon("fa5s.flask", color="#1B5E20").pixmap(QSize(48, 48))
    icon_label.setPixmap(icon_pixmap)
    icon_label.setStyleSheet("margin-bottom: 4px;")
    layout.addWidget(icon_label)

    title = QLabel("Ontario Demand Analysis")
    title.setAlignment(Qt.AlignCenter)
    title.setStyleSheet("""
        font-size: 24px; font-weight: bold; color: #1B5E20;
        margin-bottom: 2px;
    """)
    layout.addWidget(title)

    subtitle = QLabel("Hybrid Machine Learning Experimentation Platform")
    subtitle.setAlignment(Qt.AlignCenter)
    subtitle.setStyleSheet("font-size: 12px; color: #616161; margin-bottom: 16px;")
    layout.addWidget(subtitle)

    # Separator
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("background-color: #C8E6C9; max-height: 1px;")
    layout.addWidget(sep)

    prompt = QLabel("Select Application Mode")
    prompt.setAlignment(Qt.AlignCenter)
    prompt.setStyleSheet("""
        font-size: 15px; font-weight: 600; color: #2E7D32;
        margin-top: 12px; margin-bottom: 8px;
    """)
    layout.addWidget(prompt)

    result = {'mode': None}

    btn_style = """
        QPushButton {{
            background-color: {bg};
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px 24px;
            font-size: 15px;
            font-weight: bold;
            text-align: left;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:pressed {{
            background-color: {pressed};
        }}
    """

    # Desktop button
    desktop_btn = QPushButton("   Desktop Application\n            PyQt5 GUI Interface")
    desktop_btn.setIcon(qta.icon("fa5s.desktop", color="white"))
    desktop_btn.setIconSize(QSize(28, 28))
    desktop_btn.setFixedHeight(72)
    desktop_btn.setStyleSheet(btn_style.format(
        bg='#1B5E20', hover='#2E7D32', pressed='#0D3B11'))
    desktop_btn.setCursor(Qt.PointingHandCursor)

    def choose_desktop():
        result['mode'] = 'desktop'
        dialog.accept()

    desktop_btn.clicked.connect(choose_desktop)
    layout.addWidget(desktop_btn)

    # Web button
    web_btn = QPushButton("   Web Application\n            Flask Dashboard in Browser")
    web_btn.setIcon(qta.icon("fa5s.globe", color="white"))
    web_btn.setIconSize(QSize(28, 28))
    web_btn.setFixedHeight(72)
    web_btn.setStyleSheet(btn_style.format(
        bg='#4CAF50', hover='#66BB6A', pressed='#388E3C'))
    web_btn.setCursor(Qt.PointingHandCursor)

    def choose_web():
        result['mode'] = 'web'
        dialog.accept()

    web_btn.clicked.connect(choose_web)
    layout.addWidget(web_btn)

    layout.addStretch()

    # Footer
    footer = QLabel("v21.0  |  Developed by Ashikul Islam")
    footer.setAlignment(Qt.AlignCenter)
    footer.setStyleSheet("font-size: 10px; color: #9E9E9E;")
    layout.addWidget(footer)

    dialog.exec_()
    return result['mode'], app


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("  Ontario Demand ML Lab — Starting")
    logger.info("=" * 60)

    # Ensure required directories exist
    for d in ['data', 'results', 'logs', 'templates', 'static', 'static/css', 'static/js']:
        os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)

    mode, app = show_startup_dialog()

    if mode == 'desktop':
        logger.info("Launching Desktop Application (PyQt5)")
        # We already have a QApplication from the dialog, 
        # so we reuse it instead of creating another.
        from desktop_app import DesktopApp
        window = DesktopApp()
        window.show()
        sys.exit(app.exec_())

    elif mode == 'web':
        logger.info("Launching Web Application (Flask)")
        # Close the Qt event loop so Flask can run
        app.quit()
        from app import launch_web
        launch_web()

    else:
        logger.info("No mode selected — exiting.")
        sys.exit(0)


if __name__ == '__main__':
    main()
