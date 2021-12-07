import os
import mimetypes
from PIL import Image
import PyPDF2 as pyPdf
import cv2
import img2pdf
from pdf2image import convert_from_path
from pre_process.doc2text.page import Page
from logger import get_logger
from pre_process import pre_process_code

log = get_logger(__name__)

acceptable_mime = ["image/bmp", "image/png", "image/jpeg",
                   "image/jpg", "video/JPEG", "video/jpeg2000",
                   "image/tiff", "image/tif", "image/TIFF"]

FileNotAcceptedException = Exception(
    'The filetype is not acceptable. We accept bmp, png, tiff, jpg, jpeg, jpeg2000, and PDF.'
)


class Document:
    def __init__(self, lang=None):
        self.filename = None
        self.file_basename, self.file_extension = None, None
        self.num_pages = None
        self.path = None
        self.mime_type = None
        self.file_basepath = None
        self.lang = lang
        self.pages = []
        self.processed_pages = []
        self.page_content = []
        self.prepared = False
        self.error = None

    def read(self, path, image_name):
        self.filename = os.path.basename(path)
        self.file_basename, self.file_extension = os.path.splitext(self.filename)
        self.path = path
        self.mime_type = mimetypes.guess_type(path)
        self.file_basepath = os.path.dirname(path)

        # If the file is a pdf, split the pdf and prep the pages else it converts the file to pdf and performs the same.
        if self.mime_type[0] == "application/pdf" or self.mime_type[0] in acceptable_mime:
            try:
                if self.mime_type[0] in acceptable_mime:
                    im = Image.open(self.path)
                    if im.mode == "RGBA":
                        rgb_im = im.convert('RGB')
                        rgb_im.save(self.path)
                    with open("name.pdf", "wb") as f:
                        f.write(img2pdf.convert(self.path))
                    self.path = 'name.pdf'
                file_temp = open(self.path, 'rb')
                pdf_reader = pyPdf.PdfFileReader(file_temp, strict=False)
                self.num_pages = pdf_reader.numPages
                images = convert_from_path(self.path)
                i = 0
                for image in images:
                    output = pyPdf.PdfFileWriter()
                    output.addPage(pdf_reader.getPage(i))
                    path = 'temp.pdf'
                    image_type = image_name[-4:]
                    image_name = image_name.replace(image_type, ".png")
                    im_path = image_name
                    with open(path, 'wb') as f:
                        output.write(f)

                    # Convert RGB to BGR
                    image.save(im_path)
                    i = i + 1
                    byte_img = pre_process_code.ImagePreProcessor.image_to_byte_array(
                        image=Image.open(im_path))
                    rotate_img = pre_process_code.ImagePreProcessor.correct_image_angle(byte_img, im_path, i)
                    rotate_img.save(im_path)
                    rotate_img.save("rotatedimage" + str(i) + ".png")
                    image = cv2.imread(im_path)
                    de_skew_img = pre_process_code.ImagePreProcessor.correct_skew(image)
                    cv2.imwrite(im_path, de_skew_img)
                    cv2.imwrite("deskewedimage" + str(i) + ".png", de_skew_img)
                    orig_im = cv2.imread(im_path, 0)
                    current_page = Page(orig_im, i, self.lang)
                    self.pages.append(current_page)
                    os.remove(path)
                    os.remove(im_path)
                self.prepared = True
            except Exception as e:
                self.error = e
                raise
        # Otherwise, out of luck.
        else:
            log.warn(self.mime_type[0])
            raise FileNotAcceptedException

    def process(self):
        for current_page in self.pages:
            new = current_page
            # new.de_skew()
            new.crop()
            self.processed_pages.append(new)

    def extract_text(self):
        if len(self.processed_pages) > 0:
            for current_page in self.processed_pages:
                new = current_page
                text = new.extract_text(self.processed_pages.index(current_page) + 1)
                self.page_content.append(text)
        else:
            raise Exception('You must run `process()` first.')

    def get_text(self):
        if len(self.page_content) > 0:
            log.info("Final Document finalrecognized.txt created after applying preprocessing for all pages.")
            with open("finalrecognized.txt", "w+") as f:
                f.write("\n".join(self.page_content))
            # print("\n".join(self.page_content))
            return "\n".join(self.page_content)
        else:
            raise Exception('You must run `extract_text()` first.')
