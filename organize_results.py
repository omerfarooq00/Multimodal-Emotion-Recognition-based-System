import os
import shutil
from pathlib import Path

def organize_images(src_dir, dest_dir):
    """Organize extracted images into categories based on content.
    
    Args:
        src_dir (str): Source directory containing extracted images
        dest_dir (str): Destination directory for organized images
    """
    # Create category directories
    categories = {
        'eda': 'exploratory_data_analysis',
        'models': 'model_performance',
        'results': 'final_results',
        'misc': 'miscellaneous'
    }
    
    # Create destination directories
    for cat_dir in categories.values():
        os.makedirs(os.path.join(dest_dir, cat_dir), exist_ok=True)
    
    # Map filenames to categories
    file_mapping = {
        # EDA plots
        'output_0.png': 'eda',
        'output_1.png': 'eda',
        'output_2.png': 'eda',
        'output_3.png': 'eda',
        'output_4.png': 'eda',
        'output_5.png': 'eda',
        'output_6.png': 'eda',
        'output_7.png': 'eda',
        'output_8.png': 'eda',
        'output_9.png': 'eda',
        
        # Model performance
        'output_10.png': 'models',
        'output_11.png': 'models',
        'output_12.png': 'models',
        'output_13.png': 'models',
        'output_14.png': 'models',
        'output_15.png': 'models',
        'output_16.png': 'models',
        'output_17.png': 'models',
        'output_18.png': 'models',
        'output_19.png': 'models',
        
        # Results
        'output_20.png': 'results',
        'output_21.png': 'results',
        'output_22.png': 'results',
        'output_23.png': 'results',
        'output_24.png': 'results',
        'output_25.png': 'results',
        'output_26.png': 'results',
        'output_27.png': 'results',
        'output_28.png': 'results',
        'output_29.png': 'results',
        'output_30.png': 'results',
        'output_31.png': 'results',
        'output_32.png': 'results',
        'output_33.png': 'results',
        'output_34.png': 'results',
        'output_35.png': 'results',
        'output_36.png': 'results',
        'output_37.png': 'results',
        'output_38.png': 'results',
        'output_39.png': 'results',
        'output_40.png': 'results',
        'output_41.png': 'results',
        'output_42.png': 'results',
    }
    
    # Copy files to appropriate directories
    for filename, category in file_mapping.items():
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, categories[category], filename)
        
        try:
            shutil.copy2(src_path, dest_path)
            print(f'Copied {filename} to {categories[category]}')
        except FileNotFoundError:
            print(f'Warning: {filename} not found in source directory')
    
    print('\nOrganization complete!')

if __name__ == "__main__":
    # Define paths
    src_dir = r"d:\Omer_Project\results\images"
    dest_dir = r"d:\Omer_Project\results\figures"
    
    # Organize images
    organize_images(src_dir, dest_dir)
