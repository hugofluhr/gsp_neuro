import os
import argparse
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

def create_pdf(directory_path):
    # Get the list of PNG files in the directory
    png_files = [file for file in os.listdir(directory_path) if file.endswith('.png')]
    # Sort the PNG files alphabetically
    png_files.sort()

    # Create a new PDF file
    pdf_file_name = directory_path + "/" + get_last_directory(directory_path) + '.pdf'
    c = canvas.Canvas(pdf_file_name, pagesize=landscape(A4))

    # Iterate over the PNG files and add them to the PDF
    for png_file in png_files:
        # Get the full path of the PNG file
        png_path = os.path.join(directory_path, png_file)

        # Assuming you want the image to fill the page, adjust these values as needed
        width, height = landscape(A4)
        c.drawImage(png_path, 0, 0, width=width, height=height, preserveAspectRatio=True)

        # Add the filename as a text above the figure
        c.setFont("Helvetica", 12)
        c.drawString(10, height - 20, png_file)

        # Go to the next page
        c.showPage()

    # Save and close the PDF file
    c.save()

def get_last_directory(full_path):
    # Use os.path.basename to get the last directory in the path
    last_directory = os.path.basename(os.path.normpath(full_path))
    return last_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a PDF from PNG files in a directory.")
    parser.add_argument("--dir", help="Path to the directory containing PNG files", required=True)
    args = parser.parse_args()

    if os.path.isdir(args.dir):
        create_pdf(args.dir)
    else:
        print("The specified directory does not exist.")
