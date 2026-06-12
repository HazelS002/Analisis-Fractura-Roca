import fitz
import os

def pdf_to_image(input_dir, output_dir, format="png", start=1, quality=2):
    os.makedirs(output_dir, exist_ok=True)

    base_name = start
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            pdf_file = os.path.join(input_dir, file)

            doc = fitz.open(pdf_file)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(quality, quality))

            output_name = f"{base_name}.{format}"
            output_path = os.path.join(output_dir, output_name)

            pix.save(output_path)
            doc.close()
            print(f"{file} -> {base_name}.{format}")
            base_name += 1


if __name__ == "__main__": pass