import fitz # pip install PyMuPDF
import os

def pdf_to_image(pdfs_dir, images_dir, format="png", quality=2):
    """ 
    Convertir PDFs a imagenes

    Args:
        pdfs_dir (str): Carpeta de PDFs
        images_dir (str): Carpeta de imagenes
        format (str, optional): Formato de la imagen. Defaults to "png".
        quality (int, optional): Calidad de la imagen. Defaults to 2.
    """
    for file in os.listdir(pdfs_dir):    # leer archivos en la carpeta
        if file.lower().endswith(".pdf"):    # si es pdf
            pdf_file = os.path.join(pdfs_dir, file)
            
            doc = fitz.open(pdf_file)    # abrir el pdf
            page = doc.load_page(0) # suponemos una unica pagina
            pix = page.get_pixmap(matrix=fitz.Matrix(quality, quality))
            
            output_name = f"{os.path.splitext(file)[0]}.{format}"
            pix.save(os.path.join(images_dir, output_name))
            doc.close()

            print(f"{file} -> image")

if __name__ == "__main__":
    pass
