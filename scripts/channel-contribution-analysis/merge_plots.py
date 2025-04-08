from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def merge_pdf_and_png(pdf_path, png_path, output_path):
    # Read the original PDF
    pdf_reader = PdfReader(pdf_path)
    pdf_writer = PdfWriter()
    
    # Copy original PDF pages
    for page in pdf_reader.pages:
        pdf_writer.add_page(page)
    
    # Get the size of the PDF page
    pdf_page = pdf_reader.pages[0]
    pdf_width = float(pdf_page.mediabox.width)
    pdf_height = float(pdf_page.mediabox.height)
    
    # Open and resize PNG
    img = Image.open(png_path)
    # Calculate height maintaining aspect ratio
    aspect_ratio = img.height / img.width
    new_width = pdf_width
    new_height = pdf_width * aspect_ratio
    img = img.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
    
    # Create a new PDF with the image
    img_temp = BytesIO()
    img_pdf = canvas.Canvas(img_temp, pagesize=(pdf_width, new_height))
    img_pdf.drawImage(png_path, 0, 0, width=new_width, height=new_height)
    img_pdf.save()
    
    # Add the image page
    img_temp.seek(0)
    img_pdf_reader = PdfReader(img_temp)
    img_page = img_pdf_reader.pages[0]
    pdf_writer.add_page(img_page)
    
    # Save the result
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

def process_directories(pdf_dir, png_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        base_name = pdf_file[:-4]  # Remove .pdf extension
        png_file = f'{base_name}.png'
        
        # Check if corresponding PNG exists
        if png_file in os.listdir(png_dir):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            png_path = os.path.join(png_dir, png_file)
            output_path = os.path.join(output_dir, f'merged_{base_name}.pdf')
            
            print(f'Merging {pdf_file} with {png_file}...')
            merge_pdf_and_png(pdf_path, png_path, output_path)
            print(f'Created {output_path}')
        else:
            print(f'Warning: No matching PNG file found for {pdf_file}')

# Usage example
if __name__ == "__main__":
    pdf_directory = "umap-plots"
    png_directory = "csp_patterns"
    output_directory = "umap-csp-patterns"
    process_directories(pdf_directory, png_directory, output_directory)
