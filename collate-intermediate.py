import os
import random
from PIL import Image, ImageDraw, ImageFont
import re

def create_image_grid(input_folder, output_path='grid_output.png', grid_size=5, image_size=300):
    """
    Create a grid of images from a folder of square images.
    
    Args:
        input_folder: Path to folder containing images
        output_path: Path to save the output grid image
        grid_size: Number of images per row/column in the grid
        image_size: Size to resize each individual image to
    """
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # List all image files in the folder
    image_files = [(f, float(f.split('.png')[0].split('_')[-1])) for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # sort the iamge files by name
    image_files.sort(key=lambda x: x[1])
    image_files = [f[0] for f in image_files]
    
    # Randomly sample if we have more than grid_size^2 images
    max_images = grid_size * grid_size
    if len(image_files) > max_images:
        # Calculate the skip interval
        skip_interval = len(image_files) // max_images
        # Use every skip_interval-th image
        sampled_files = [image_files[i] for i in range(0, len(image_files), skip_interval)][:max_images]
    else:
        sampled_files = image_files
    
    # Create a blank canvas for the grid
    canvas_size = image_size * grid_size
    grid_image = Image.new('RGB', (canvas_size, canvas_size), color='white')
    
    # Try to load a font for the image titles
    try:
        # Try to use a common font
        font = ImageFont.truetype("Arial.ttf", 20)
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Place each image in the grid
    for i, filename in enumerate(sampled_files):
        # Calculate position in grid
        row = i // grid_size
        col = i % grid_size
        
        # Load and resize image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img = img.resize((image_size, image_size))
        
        # Extract t value from filename (e.g., image_t_681.0.png -> 681)
        t_value = None
        match = re.search(r'image_t_(\d+(\.\d+)?)', filename)
        if match:
            t_value = match.group(1)
            if t_value.endswith('.0'):
                t_value = t_value[:-2]  # Remove .0 suffix for clean integer display
        
        # Add t value as text overlay if found
        if t_value:
            draw = ImageDraw.Draw(img)
            # Add text at the top with a small margin
            draw.text((10, 10), f"t = {t_value}", fill="white", font=font, stroke_width=2, stroke_fill="black")
        
        # Paste the image onto the grid canvas
        grid_image.paste(img, (col * image_size, row * image_size))
    
    # Save the final grid image
    grid_image.save(output_path)
    print(f"Grid image created at {output_path}")
    return output_path

if __name__ == "__main__":
    # Replace with your actual folder path
    input_folder = "/global/homes/c/chrislai/tt-scale-flux/output/sdxl-base/qwen/overall_score/20250420_223857/latent_images"
    # input_folder = "/global/homes/c/chrislai/tt-scale-flux/output/sdxl-base/qwen/overall_score/20250420_224806/latent_images"
    input_folder = "/global/homes/c/chrislai/tt-scale-flux/output/flux.1-dev/qwen/overall_score/20250421_181911/latent_images"
    input_folder = "/global/homes/c/chrislai/tt-scale-flux/output/flux.1-dev/qwen/overall_score/20250422_152237/latent_images/1672527486"
    create_image_grid(input_folder, output_path=os.path.join(input_folder, 'grid_output.png'))