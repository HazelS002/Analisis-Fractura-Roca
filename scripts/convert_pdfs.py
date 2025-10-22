import fitz # pip install PyMuPDF
import os

def convert(dir, quality=2, format="png", delete_pdf=False):
    """ 
    Convertir archivos pdf a imagenes

    Args:
        dir (str): ruta de la carpeta
        quality (int, optional): calidad de la imagen. Defaults to 2.
        format (str, optional): formato de la imagen. Defaults to "png".
        delete_pdf (bool, optional): si se borra el pdf. Defaults to False.
    """
    for file in os.listdir(dir):    # leer archivos en la carpeta
        if file.lower().endswith(".pdf"):    # si es pdf

            pdf_file = os.path.join(dir, file)
            
            doc = fitz.open(pdf_file)    # abrir el pdf
            page = doc.load_page(0) # suponemos una unica pagina
            pix = page.get_pixmap(matrix=fitz.Matrix(quality, quality))
            
            output_name = f"{os.path.splitext(file)[0]}.{format}"
            pix.save(os.path.join(dir, output_name))
            doc.close()

            if delete_pdf: os.remove(pdf_file) # borrar el pdf

            print(f"{file} -> image")

    return

if __name__ == "__main__":
    convert("./sample-images", 2, "png", delete_pdf=True)

