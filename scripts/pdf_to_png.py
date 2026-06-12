from data_process.utils import pdf_to_image, get_lastest
from .config import RAW_DATA_DIR, PROCESSED_IMAGES_DIR

def main(batch: int):
    pdf_dir = RAW_DATA_DIR + f"batch-{batch}/"
    png_dir = PROCESSED_IMAGES_DIR + "png-images/"
    pdf_to_image(pdf_dir, png_dir, start=get_lastest(png_dir)+1)
    return

if __name__ == "__main__":
    main(batch=3)
