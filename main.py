import os
from pre_process.pre_process_code import ImagePreProcessor as PreProcess
from post_process.post_process_code import PostProcess as Post
from process.spacy_code import Spacy as SpacyCode
from logger import get_logger
# from os import listdir
# from os.path import isfile, join
# from PIL import Image
# from pdf2image import convert_from_path
log = get_logger(__name__)

if __name__ == '__main__':
    # Code for Converting one type of document to another
    # directory = os.getcwd() + '/JPEG_images'
    # os.chdir(directory)
    # path = "../pre_process/Colosseum Group - Fraud OCR Documentation/"
    # images = [f for f in listdir(path) if isfile(join(path, f))]
    # log.info(f"Names of all PDFs: {images}")
    #
    # # For converting all PDFs to JPEG
    # # for image in images:
    # #     new_path = path + str(image)
    # #     log.info(f"Image processing: {image}")
    # #     f_image = convert_from_path(new_path)
    # #     for i, imag in enumerate(f_image):
    # #         fname = "image" + str(i) + ".jpg"
    # #         img = directory + "/" + str(image[:-4]) + "-" + str(i) + ".jpg"
    # #         imag.save(img, "JPEG")
    #
    # # For converting all JPEGs to PNG
    # for image in images:
    #     new_path = path + str(image)
    #     log.info(f"Image processing: {image}")
    #     im1 = Image.open(new_path)
    #     im1.save(directory + "/" + str(image[:-4]) + ".png")

    # Main Code
    directory = os.getcwd() + '/output_images'
    os.chdir(directory)
    log.info(f"Before saving images: {os.listdir(directory)}")

    # Pre processing
    log.info("Pre Processing Initiated")
    path = "../pre_process/realpage_input_images/"
    image = "CentralHudson-GasElectric.pdf"
    doc2text_text = PreProcess.doc_to_text(path + image, image)

    # Processing
    log.info("Processing Initiated")
    processed_data = SpacyCode.perform_spacy(doc2text_text)

    # Post Processing
    log.info("Post Processing Initiated")
    Post.post_process(processed_data, image)

    log.info(f"After saving images: {os.listdir(directory)}")

