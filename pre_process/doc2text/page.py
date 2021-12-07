import traceback
import sys

from scipy.ndimage import rank_filter
from skimage.filters import threshold_local
import cv2
import numpy as np
import pytesseract
from logger import get_logger
from pre_process import pre_process_code
# from tesserocr import PyTessBaseAPI, RIL
# import pandas as pd
from PIL import Image
import os

# name of the file to save
filename = "img01.png"

# create new image

log = get_logger(__name__)


class Page:
    def __init__(self, im, page_num, lang=None):
        self.image = None
        self.num_tries = None
        self.crop_shape = None
        self.theta_est = None
        self.text = None
        self.healthy = True
        self.err = False
        self.page_num = page_num
        self.orig_im = im
        self.orig_shape = self.orig_im.shape
        self.lang = lang
        self.count = 0
        self.json_output = {}

    @staticmethod
    def process_image(orig_im):
        return orig_im

    # Pre Process method 1
    @staticmethod
    def pre_process_image1(orig_im, page):
        # height, width, channel = image.shape

        gray = cv2.cvtColor(orig_im, cv2.COLOR_BGR2GRAY)

        thresh = threshold_local(gray, 15, offset=6, method="gaussian")  # generic, mean, median, gaussian
        cv2.imwrite("altthreshimage" + str(page) + ".png", thresh)
        thresh = (gray > thresh).astype("uint8") * 255
        thresh = ~thresh

        # Dilation
        kernel = np.ones((1, 1), np.uint8)
        ero = cv2.erode(thresh, kernel, iterations=1)
        cv2.imwrite("alterodedimage" + str(page) + ".png", ero)
        img_dilation = cv2.dilate(ero, kernel, iterations=1)
        cv2.imwrite("altdilatedimage" + str(page) + ".png", img_dilation)
        # Remove noise
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilation, None, None, None, 8,
                                                                              cv2.CV_32S)
        sizes = stats[1:, -1]  # get CC_STAT_AREA component
        final = np.zeros(labels.shape, np.uint8)
        for i in range(0, n_labels - 1):
            if sizes[i] >= 10:  # filter small dotted regions
                final[labels == i + 1] = 255

        # Find contours
        kern = np.ones((5, 15), np.uint8)
        return final, kern

    # Pre Process method 2
    @staticmethod
    def pre_process_image2(orig_im, page):
        # height, width, channel = image.shape
        gray_image = pre_process_code.ImagePreProcessor.get_grayscale(orig_im)

        # Performing OTSU threshold
        thresh_image = pre_process_code.ImagePreProcessor.thresholding(gray_image)
        cv2.imwrite("threshimage" + str(page) + ".png", thresh_image)
        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        # Applying dilation on the threshold image
        eroded_image = pre_process_code.ImagePreProcessor.erode(thresh_image)
        cv2.imwrite("erodedimage" + str(page) + ".png", eroded_image)
        dilated_image = pre_process_code.ImagePreProcessor.dilated(eroded_image)
        cv2.imwrite("dilatedimage" + str(page) + ".png", dilated_image)
        return dilated_image

    def crop(self):
        try:
            self.image = self.process_image(self.orig_im)
            self.crop_shape = self.image.shape
            return self.image
        except Exception as e:
            for frame in traceback.extract_tb(sys.exc_info()[2]):
                f_name, line_no, fn, text = frame
                log.warning("Error in %s on line %d" % (f_name, line_no))
                log.warning(e)
            self.err = e
            self.healthy = False

    def extract_text(self, page):
        log.info("Inside text extraction")
        im_path = 'temp.png'
        cv2.imwrite(im_path, self.image)
        log.info(f"Confidence results before applying preprocessing methods of page {page}")
        text = pytesseract.image_to_data(Image.open(im_path), output_type='data.frame')
        total = text['conf'].sum()
        text['total'] = text[['conf']].sum(axis=1).where(text['conf'] > 0, 0)
        avg_conf_total = text['total'].sum()
        log.info(f"Average Total: {total}")
        text['total2'] = text[['conf']].count(axis=1).where(text['conf'] > 0, 0)
        text['total3'] = text[['conf']].sum(axis=1).where(text['conf'] > 0, 100)
        total = text['total2'].sum()
        log.info(f'Result of average confidence: {avg_conf_total / total}')
        log.info(f"Minimum Confidence: {text['total3'].min()}")
        log.info(f"Maximum Confidence: {text['total'].max()}")
        percent = text['total'].quantile(0.9)
        log.info(f"90 Percentile: {percent}")
        # min_conf_text = text['total3'].min()
        # text.text[text.total3] == min_conf_text]
        # log.info("Complete dataset: ")
        # print(text)
        log.info(f"Text nonpreprocessed{page}.txt before applying Preprocessing created of page {page}")
        alt_text = pytesseract.image_to_string(Image.open(im_path))
        with open("nonpreprocessed" + str(page) + ".txt", "w+") as f:
            f.write(alt_text)
        # print(pytesseract.image_to_string(Image.open(im_path)))
        # avg_conf=0
        # total_text=0
        # with PyTessBaseAPI() as api:
        #     api.SetImage(image)
        #     boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        #     print('Found {} textline image components.'.format(len(boxes)))
        #     for i, (im, box, _, _) in enumerate(boxes):
        #         # im is a PIL image object
        #         # box is a dict with x, y, w and h keys
        #         api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        #         ocrResult = api.GetUTF8Text()
        #         conf = api.MeanTextConf()
        #         print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
        #               "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))
        #         # self.json_output.append()
        #         if conf!=0:
        #             total_text=total_text+1
        #             avg_conf=avg_conf+conf
        # print('resultant confidence')
        # print(avg_conf/total_text)

        image = cv2.imread(im_path)
        os.remove(im_path)
        log.info(f"Confidence results after applying preprocessing methods of page {page}")
        # First Method
        log.info("Applying the first Preprocessing Approach")
        output = ""
        dilated_image = self.pre_process_image2(image, page)

        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Creating a copy of image
        im2 = image.copy()
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

            cv2.imwrite(im_path, cropped)
            cv2.imwrite("croppedimage" + str(page) + ".png", cropped)
            text = pytesseract.image_to_data(Image.open(im_path), output_type='data.frame')
            total = text['conf'].sum()
            text['total'] = text[['conf']].sum(axis=1).where(text['conf'] > 0, 0)
            avg_conf_total = text['total'].sum()
            log.info(f"Average Total: {total}")
            text['total2'] = text[['conf']].count(axis=1).where(text['conf'] > 0, 0)
            text['total3'] = text[['conf']].sum(axis=1).where(text['conf'] > 0, 100)
            total = text['total2'].sum()
            log.info(f'Result of average confidence: {avg_conf_total / total}')
            log.info(f"Minimum Confidence: {text['total3'].min()}")
            log.info(f"Maximum Confidence: {text['total'].max()}")
            percent = text['total'].quantile(0.9)
            log.info(f"90 Percentile: {percent}")
            # min_conf_text = text['total3'].min()
            # text.text[text.total3] == min_conf_text]
            # log.info("Complete dataset: ")
            # print(text)
            os.remove(im_path)
            # Open the file in append mode
            # Close the file
            text = pytesseract.image_to_string(cropped)
            with open("recognized" + str(page) + ".txt", "w+") as f:
                f.write(text)
            # Appending the text into file
            output += str(text)
        # output = " ".join([word for word in output.split()])
        self.text = output

        # Second Method
        # log.info("Applying the second Preprocessing Approach")
        # # Threshold
        # final, kern = self.pre_process_image1(image, page)
        #
        # img_dilation = cv2.dilate(final, kern, iterations=1)
        # cv2.imwrite("alt1dilatedimage" + str(page) + ".png", img_dilation)
        # contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # # Map contours to bounding rectangles, using bounding_rect property
        # rectangles = map(lambda c: cv2.boundingRect(c), contours)
        # # Sort rectangles by top-left x (rect.x == rect.tl.x)
        # sorted_rectangles = sorted(rectangles, key=lambda r: r[0])
        # sorted_rectangles = sorted(sorted_rectangles, key=lambda r: r[1])
        # output = ''
        # for rect in sorted_rectangles:
        #     x, y, w, h = rect
        #     if w < 20 or h < 20:
        #         continue
        #     temp = image[y:y + h, x:x + w]
        #     temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        #     hi = pytesseract.image_to_data(temp, config=r'--psm 6')
        #     hi = hi.split()
        #     ind = 22
        #     while True:
        #         if ind > len(hi):
        #             break
        #         if int(hi[ind]) == -1:
        #             ind += 11
        #         else:
        #             output = output + hi[ind + 1]
        #             output = output + " "
        #             x += len(hi[ind + 1]) * 20
        #             ind += 12
        #     output = output + '\n'
        # with open("altrecognized" + str(page) + ".txt", "w+") as f:
        #     f.write(output)
        # log.info(f"Output altrecognized{page}.txt of Secondary preprocessing method created of page {page}")
        # print(output)

        log.info(f'text output recognized{page}.txt created after preprocess of page {page}')
        # print(self.text)
        return self.text

    def save(self, out_path):
        if not self.healthy:
            log.warning("There was an error when cropping")
            raise Exception(self.err)
        else:
            cv2.imwrite(out_path, self.image)
