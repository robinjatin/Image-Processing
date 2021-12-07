import io
from pre_process.doc2text.doc2text import Document as Doc
import pytesseract
from PIL import Image
import math
from scipy import ndimage
from pytesseract import Output
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter, rank_filter
from logger import get_logger
log = get_logger(__name__)


class ImagePreProcessor:
    @staticmethod
    def resize_img(image, x=1050, y=1610):
        return cv2.resize(image, (x, y))

    @staticmethod
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    @staticmethod
    def remove_noise(image):
        return cv2.medianBlur(image, 5)

    # thresholding
    @staticmethod
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    @staticmethod
    def threshold(gray):
        return cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # dilation
    @staticmethod
    def dilate(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    @staticmethod
    def dilated(thresh):
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        return cv2.dilate(thresh, rect_kernel, iterations=1)

    # erosion
    @staticmethod
    def erode(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    @staticmethod
    def opening(image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    @staticmethod
    def canny(image):
        return cv2.Canny(image, 100, 200)

    # skew correction
    @staticmethod
    def de_skew(image):
        coordinates = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coordinates)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    def reduce_noise_raw(im):
        bilat = cv2.bilateralFilter(im, 9, 75, 75)

        blur = cv2.medianBlur(bilat, 5)
        return blur

    def reduce_noise_edges(im):
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, structuring_element)
        maxed_rows = rank_filter(opening, -4, size=(1, 20))
        maxed_cols = rank_filter(opening, -4, size=(20, 1))
        debordered = np.minimum(np.minimum(opening, maxed_rows), maxed_cols)
        return debordered
    @staticmethod
    def convert_to_bounding_box(img):
        h, w, c = img.shape
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            b = b.split(' ')
            img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        return img

    @staticmethod
    def correct_image_angle(img_bytes, filename, page=1):
        image = Image.open(io.BytesIO(img_bytes))
        try:
            osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
            log.info("Orientation for File: %s, page: %s is: %s", filename, page, osd)
            angle = osd['rotate']
            angle_confidence = osd.get('orientation_conf')
            script_confidence = osd.get('script_conf')
            script = osd.get('script').lower()
            if angle != 0:
                if script in ['arabic'] and angle_confidence < 3:
                    angle = 0
                    log.warning("Error identifying the angle for page %s: Low confidence", page)
                elif script in ['greek'] and angle_confidence < 3 and page < 3:
                    angle = 0
                    log.warning("Error identifying the angle for page %s: Low confidence", page)
                elif script in ['greek'] and angle_confidence <= 0.5 and script_confidence <= 0.5:
                    angle = 0
                    log.warning("Error identifying the angle for page %s: Low confidence", page)
                elif script in ['cyrillic'] and script_confidence < 1 and angle_confidence < 1 and page < 3:
                    angle = 0
                    log.warning("Error identifying the angle for page %s: Low confidence", page)
                elif script in ['cyrillic'] and script_confidence == 0.0 and angle_confidence < 1:
                    angle = 0
                    log.warning("Error identifying the angle for page %s: Low confidence", page)
                elif script in ['japanese'] and script_confidence <= 0.5 and angle_confidence < 0.5:
                    angle = 360 - angle
                    log.warning("Error identifying the angle for page %s: Low confidence", page)
        except Exception as e:
            log.warning("Error identifying the angle for page %s: %s", page, str(e), exc_info=1)
            angle = 0
        angle = int(angle)
        if angle in [-90, 270, 90, -270, 180, -180]:
            log.info("Rotating image with angle: {} page: {}".format(360 - angle, page))
            image = image.rotate(360 - angle, expand=True)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return image

    @staticmethod
    def image_to_byte_array(image: Image, image_format="png", **params) -> bytes:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image_format, **params)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr

    @classmethod
    def correct_skew(cls, image, delta=1, limit=5):
        gray = cls.get_grayscale(image)
        thresh = cls.thresholding(gray)
        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = cls.determine_score(thresh, angle)
            scores.append(score)
        best_angle = angles[scores.index(max(scores))]
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    @classmethod
    def image2txt(cls, img):
        # Preprocessing the image starts
        output = ""
        # Convert the image to gray_image scale
        gray_image = cls.get_grayscale(img)

        # Performing OTSU threshold
        thresh_image = cls.thresholding(gray_image)

        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        # Applying dilation on the threshold image
        dilated_image = cls.dilated(thresh_image)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        # Creating a copy of image
        im2 = img.copy()
        # A text file is created and flushed
        # Looping through the identified contours
        # Then rectangular part is cropped and passed on
        # to pytesseract for extracting text from it
        # Extracted text is then written into the text file

        # reversing the contours
        contours = contours[::-1]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Drawing a rectangle on copied image
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]
            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(cropped)
            # Open the file in append mode
            # Close the file
            with open("recognized.txt", "w+") as f:
                f.write(text)
                f.write("\n")
            # Appending the text into file
            output += str(text)
            output += "\n"
        log.info(f'text output after preprocess: ')
        print(output)
        return output

    @staticmethod
    def doc_to_text(path, image_name):
        doc = Doc()

        # You can pass the lang (as 3 letters code) to the class to improve accuracy
        # On ubuntu it requires the package tesseract-ocr-$lang$
        # On other OS, see https://github.com/tesseract-ocr/langdata
        # doc = doc2text.Document(lang="eng")

        # Read the file in. Currently accepts pdf, png, jpg, bmp, tiff.
        # If reading a PDF, doc2text will split the PDF into its component pages.
        doc.read(path, image_name)

        # Crop the pages down to estimated text regions, de_skew, and optimize for OCR.
        doc.process()

        # Extract text from the pages.
        doc.extract_text()
        text = doc.get_text()
        return text

    @staticmethod
    def perform_rotation(img_before):

        img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

        angles = []

        if lines.any() and len(lines) > 1:
            for [[x1, y1, x2, y2]] in lines:
                # cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

            median_angle = np.median(angles)
            img_rotated = ndimage.rotate(img_before, median_angle)
            return img_rotated
        return img_before
