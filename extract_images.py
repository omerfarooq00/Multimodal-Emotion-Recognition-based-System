import os
import json
import base64
from pathlib import Path

def extract_images_from_ipynb(notebook_path, output_dir):
    """Extract images from Jupyter notebook and save them to output directory.
    
    Args:
        notebook_path (str): Path to the Jupyter notebook (.ipynb file)
        output_dir (str): Directory to save the extracted images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    image_count = 0
    
    # Iterate through all cells
    for cell_num, cell in enumerate(notebook.get('cells', [])):
        # Check if cell has outputs
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                # Check if output contains image data
                if 'data' in output and 'image/png' in output['data']:
                    # Get image data
                    image_data = output['data']['image/png']
                    
                    # Save image to file
                    image_path = os.path.join(output_dir, f'output_{image_count}.png')
                    with open(image_path, 'wb') as img_file:
                        img_file.write(base64.b64decode(image_data))
                    
                    print(f'Saved image: {image_path}')
                    image_count += 1
    
    print(f'\nExtracted {image_count} images to {output_dir}')

if __name__ == "__main__":
    # Define paths
    notebook_path = r"d:\Omer_Project\FInal_code.ipynb"
    output_dir = r"d:\Omer_Project\results\images"
    
    # Extract images
    extract_images_from_ipynb(notebook_path, output_dir)
