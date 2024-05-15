import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.uic import loadUi
import numpy as np

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
        elif index == 4:
            self.adaptive_thresholding(self.current_frame_index)
        elif index == 5:
            self.clustering(self.current_frame_index)
        elif index == 6:
            self.LoG(self.current_frame_index)
        elif index == 7:
            # Erosion
            self.morphological_ops(self.current_frame_index, 'erosion')
        elif index == 8:
            # Dilation
            self.morphological_ops(self.current_frame_index, 'dilation')
        elif index == 9:
            # Opening
            self.morphological_ops(self.current_frame_index, 'opening')
        elif index == 10:
            # Closing
            self.morphological_ops(self.current_frame_index, 'closing')

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

    def adaptive_thresholding(self, frame=0):
        if self.original_image is not None and self.count == 1:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                    11, 2)
            h, w = adaptive_thresh.shape
            bytes_per_line = w
            q_image = QImage(adaptive_thresh.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            gray_image = cv2.cvtColor(self.video_frames[frame], cv2.COLOR_BGR2GRAY)
            adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                    11, 2)
            h, w = adaptive_thresh.shape
            bytes_per_line = w
            q_image = QImage(adaptive_thresh.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)

    def clustering(self, frame=0):
        if self.original_image is not None and self.count == 1:
            # Convert image to RGB
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            # Flatten the image to 2D array
            pixels = image_rgb.reshape((-1, 3))

            # Perform k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 10  # You can adjust the number of clusters as needed
            _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert centers to 8-bit values
            centers = np.uint8(centers)

            # Map each pixel to its nearest centroid
            segmented_data = centers[labels.flatten()]

            # Reshape segmented data into the original image dimensions
            segmented_image = segmented_data.reshape(image_rgb.shape)

            # Convert segmented image to Grayscale
            segmented_image_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)

            # Display the segmented image
            h, w = segmented_image_gray.shape
            bytes_per_line = w
            q_image = QImage(segmented_image_gray.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        elif self.video_frames:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(self.video_frames[frame], cv2.COLOR_BGR2RGB)
            # Flatten the frame to 2D array
            pixels = frame_rgb.reshape((-1, 3))

            # Perform k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 10  # You can adjust the number of clusters as needed
            _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert centers to 8-bit values
            centers = np.uint8(centers)

            # Map each pixel to its nearest centroid
            segmented_data = centers[labels.flatten()]

            # Reshape segmented data into the original frame dimensions
            segmented_frame = segmented_data.reshape(frame_rgb.shape)

            # Convert segmented frame to Grayscale
            segmented_frame_gray = cv2.cvtColor(segmented_frame, cv2.COLOR_RGB2GRAY)

            # Display the segmented frame
            h, w = segmented_frame_gray.shape
            bytes_per_line = w
            q_image = QImage(segmented_frame_gray.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            print("No media file is loaded.")

    def LoG(self, frame=0):
        if self.original_image is not None and self.count == 1:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            # Apply Laplacian
            laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
            # Convert back to uint8 and scale to 0-255
            laplacian_image = cv2.convertScaleAbs(laplacian)
            h, w = laplacian_image.shape
            bytes_per_line = w
            q_image = QImage(laplacian_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)
        else:
            gray_image = cv2.cvtColor(self.video_frames[frame], cv2.COLOR_BGR2GRAY)
            # Apply Gaussian Blur
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            # Apply Laplacian
            laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
            # Convert back to uint8 and scale to 0-255
            laplacian_image = cv2.convertScaleAbs(laplacian)
            h, w = laplacian_image.shape
            bytes_per_line = w
            q_image = QImage(laplacian_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_media.setPixmap(pixmap)
            self.lbl_media.setScaledContents(True)

    def erosion(self,image, kernel):
        # Assumes a grayscale image and structuring element (kernel)
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape
        half_kh, half_kw = kernel_h // 2, kernel_w // 2

        eroded_image = np.zeros_like(image)

        for y in range(half_kh, img_h - half_kh):
            for x in range(half_kw, img_w - half_kw):
                patch = image[y - half_kh: y + half_kh + 1, x - half_kw: x + half_kw + 1]
                eroded_image[y, x] = np.min(patch * kernel)

        return eroded_image

    def dilation(self,image, kernel):
        # Assumes a grayscale image and structuring element (kernel)
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape
        half_kh, half_kw = kernel_h // 2, kernel_w // 2

        dilated_image = np.zeros_like(image)

        for y in range(half_kh, img_h - half_kh):
            for x in range(half_kw, img_w - half_kw):
                patch = image[y - half_kh: y + half_kh + 1, x - half_kw: x + half_kw + 1]
                dilated_image[y, x] = np.max(patch * kernel)

        return dilated_image

    def opening(self, image, kernel):
        # Perform erosion followed by dilation
        return self.dilation(self.erosion(image, kernel), kernel)

    def closing(self, image, kernel):
        # Perform dilation followed by erosion
        return self.erosion(self.dilation(image, kernel), kernel)

    def morphological_ops(self, frame, operation):
        if self.original_image is not None and self.count == 1:
            gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = cv2.cvtColor(self.video_frames[frame], cv2.COLOR_BGR2GRAY)

        # Define the kernel
        kernel = np.ones((3, 3), np.uint8)

        if operation == 'erosion':
            result_img = self.erosion(gray_img, kernel)
        elif operation == 'dilation':
            result_img = self.dilation(gray_img, kernel)
        elif operation == 'opening':
            result_img = self.opening(gray_img, kernel)
        elif operation == 'closing':
            result_img = self.closing(gray_img, kernel)

        # Display the result image
        h, w = result_img.shape
        bytes_per_line = w
        q_image = QImage(result_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
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
