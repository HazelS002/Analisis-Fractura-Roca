import fitz
import os

def pdf_to_image(pdfs_dir, images_dir, format="png", quality=2):
    os.makedirs(images_dir, exist_ok=True)

    for file in os.listdir(pdfs_dir):
        if file.lower().endswith(".pdf"):
            pdf_file = os.path.join(pdfs_dir, file)

            doc = fitz.open(pdf_file)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(quality, quality))

            base_name = os.path.splitext(file)[0]
            output_name = f"{base_name}.{format}"
            output_path = os.path.join(images_dir, output_name)

            temp = 1
            while os.path.exists(output_path):
                output_name = f"{base_name}({temp}).{format}"
                output_path = os.path.join(images_dir, output_name)
                temp += 1

            pix.save(output_path)
            doc.close()

            print(f"{file} -> image")


if __name__ == "__main__":
    raw_root = "./data/raw/"
    output_dir = "./data/processed/png-images/"

    for batch_dir in os.listdir(raw_root):
        batch_path = os.path.join(raw_root, batch_dir)
        if os.path.isdir(batch_path):
            pdf_to_image(batch_path, output_dir)