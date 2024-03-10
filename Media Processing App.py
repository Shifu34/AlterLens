import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.uic import loadUi


class MediaUploader(QMainWindow):
    def __init__(self):
        super(MediaUploader, self).__init__()
        loadUi("ui_file.ui", self)  # Load the UI file created with PyQt Designer
        self.btn_upload.clicked.connect(self.upload_media)
        self.apply.clicked.connect(lambda: self.image_transformations(self.options.currentIndex()))
        self.original_image = None  # Initialize original_image attribute
        self.next_btn.clicked.connect(self.next_frame)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.video_frames = []
        self.current_frame_index = 0
        self.count = 0
        self.next_btn.setVisible(False)  # Initially hide next button
        self.prev_btn.setVisible(False)  # Initially hide previous button

    def upload_media(self):

        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mkv)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_path, _ = file_dialog.getOpenFileName(self, "Open Media", "",
                                                   "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mkv)")
        if file_path:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                pixmap = QPixmap(file_path)
                self.lbl_media.setPixmap(pixmap)
                self.lbl_media.setScaledContents(True)
                self.original_image = cv2.imread(file_path)
                self.count = 1
            elif file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
                self.video_frames = []
                cap = cv2.VideoCapture(file_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self.video_frames.append(frame)
                cap.release()
                self.show_frame(0)
                self.count = 0
                self.toggle_arrow_buttons(True)
            else:
                print("Unsupported file format.")

    def show_frame(self, index):
        image = cv2.cvtColor(self.video_frames[index], cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.lbl_media.setPixmap(pixmap)
        self.lbl_media.setScaledContents(True)

    def image_transformations(self, index):
        if index == 0:
            # thresholding
            self.thresholding(self.current_frame_index)
        elif index == 1:
            # histogram equalization
            self.histogram_eq(self.current_frame_index)
        elif index == 2:
            # Average Filter
            self.average_filter(self.current_frame_index)
        elif index == 3:
            # Laplacian
            self.laplican(self.current_frame_index)

    def histogram_eq(self, frame=0):
        if self.original_image is not None and self.count == 1:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            h, w = equalized_image.shape
            bytes_per_line = w
            q_image = QImage(equalized_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            gray_image = cv2.cvtColor(self.video_frames[frame], cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            h, w = equalized_image.shape
            bytes_per_line = w
            q_image = QImage(equalized_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)

    def thresholding(self, frame=0):
        if self.original_image is not None and self.count == 1:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            h, w = thresholded_image.shape
            bytes_per_line = w
            q_image = QImage(thresholded_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            gray_image = cv2.cvtColor(self.video_frames[frame], cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            h, w = thresholded_image.shape
            bytes_per_line = w
            q_image = QImage(thresholded_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)

    def average_filter(self, frame=0):
        if self.original_image is not None and self.count == 1:
            blurred_image = cv2.blur(self.original_image, (5, 5))  # Apply average filter with 5x5 kernel size
            h, w, ch = blurred_image.shape
            bytes_per_line = ch * w
            q_image = QImage(blurred_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            blurred_image = cv2.blur(self.video_frames[frame], (5, 5))  # Apply average filter with 5x5 kernel size
            h, w, ch = blurred_image.shape
            bytes_per_line = ch * w
            q_image = QImage(blurred_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)

    def laplican(self, frame=0):
        if self.original_image is not None and self.count == 1:
            laplacian_image = cv2.Laplacian(self.original_image, cv2.CV_64F)
            laplacian_image = cv2.convertScaleAbs(laplacian_image)
            h, w, ch = laplacian_image.shape
            bytes_per_line = ch * w
            q_image = QImage(laplacian_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            laplacian_image = cv2.Laplacian(self.video_frames[frame], cv2.CV_64F)
            laplacian_image = cv2.convertScaleAbs(laplacian_image)
            h, w, ch = laplacian_image.shape
            bytes_per_line = ch * w
            q_image = QImage(laplacian_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)

    def next_frame(self):
        if self.video_frames:
            self.current_frame_index = (self.current_frame_index + 1)
            self.show_frame(self.current_frame_index)

    def prev_frame(self):
        if self.video_frames:
            self.current_frame_index = (self.current_frame_index - 1)
            self.show_frame(self.current_frame_index)

    def toggle_arrow_buttons(self, visible):
        self.next_btn.setVisible(visible)
        self.prev_btn.setVisible(visible)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MediaUploader()
    window.show()
    sys.exit(app.exec())
