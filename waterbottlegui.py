import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QFont, QMovie
from PyQt5.QtCore import Qt, QTimer, QSize
from waterbottle4 import estimate_water_level

class WaterLevelApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Water Level Detector")
        self.setStyleSheet("background-color: #D9ED92;")
        self.showFullScreen()  

        self.layout = QVBoxLayout()

        # Header
        self.label = QLabel("Bottles spill, and we count it")
        self.label.setFont(QFont("KAGE_DEMO_FONT Black", 40))
        self.label.setStyleSheet("color: #1A759F;")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        # Image preview
        self.image_label = QLabel()
        self.image_label.setFixedSize(500, 500)
        self.image_label.setStyleSheet("""
            border: 2px dashed #76C893;
            background-color: #D9ED92;
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Upload button
        self.button = QPushButton("Upload Image")
        self.button.setFont(QFont("KAGE_DEMO_FONT Black", 20))
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #76C893;
                color: #89376C;
                border: none;
                padding: 15px 30px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background-color: #1A759F;
            }
        """)
        self.button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.button, alignment=Qt.AlignCenter)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("KAGE_DEMO_FONT Black", 24, QFont.Bold))
        self.result_label.setStyleSheet("color: #184E77; padding-top: 20px;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_label.setPixmap(QPixmap(file_path).scaled(500, 500, Qt.KeepAspectRatio))
            result = estimate_water_level(file_path)
            self.result_label.setText(result)

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading...")
        self.setFixedSize(400, 400)
        self.setStyleSheet("background-color: #D9ED92; border: none;")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)

        # Load GIF
        movie = QMovie(r"D:\uselesscat\squirtle final loop.gif")
        movie.setScaledSize(QSize(300, 300))
        label.setMovie(movie)
        movie.start()

        layout.addWidget(label)
        self.setLayout(layout)

        # Timer to load main app
        QTimer.singleShot(3000, self.show_main_app)

    def show_main_app(self):
        self.main_window = WaterLevelApp()
        self.main_window.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec_())