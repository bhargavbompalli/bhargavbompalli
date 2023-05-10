from PIL import Image
import numpy as np
import cv2

'''
def replace_pixels (image_path, mask_path, output_path):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    #RGBA can handle transparent pixels in the images
    image = image.convert("RGBA")
    mask = image.convert("RGBA")

    image_data = image.load()
    mask_data = mask.load()

    print(mask_data)

    for y in range(image.height):
        for x in range(image.width):
            if mask_data[x, y][0] != 0 or mask_data[x, y][1] != 0 or mask_data[x, y][2] != 0:
                image_data[x, y] = mask_data[x, y]
    
    image.save(output_path)

replace_pixels("/home/bhargav/VC/neuralbody/out_2/KCam1/00000.jpg","/home/bhargav/VC/neuralbody/out_2/mask_cihp/KCam1/00000.png", "/home/bhargav/VC/neuralbody/out_2/00000.png")
'''

def transparent_non_zero_pixels(mask_path, transparent_path):
    # Open the image using PIL
    img = Image.open(mask_path).convert("RGBA")

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Find the indices of non-zero pixels
    non_zero_coords = np.argwhere(np.any(img_array[:, :, :3] != 0, axis=-1))

    # Set the alpha channel to 0 for non-zero pixels (make them transparent)
    img_array[non_zero_coords[:, 0], non_zero_coords[:, 1], 3] = 0

    # Convert the modified NumPy array back to a PIL image
    img_transparent = Image.fromarray(img_array)
    img_transparent.save(transparent_path, "PNG")

def overlay_images(original_path, transparent_path, final_out):

    original_image = Image.open(original_path).convert("RGBA")
    transparent_mask = Image.open(transparent_path).convert("RGBA")

    assert original_image.size == transparent_mask.size, "The image and the mask must be the same size"

    composite = Image.alpha_composite(original_image, transparent_mask)
    composite.save(final_out, "PNG")


if __name__ == "__main__":
    mask_path = "/home/bhargav/VC/Human-Body-Measurements-using-Computer-Vision/input_data/steven_body/steven_body.png"  # Replace with your image path
    original_path = "/home/bhargav/VC/Human-Body-Measurements-using-Computer-Vision/input_data/steven_body/steven_body.jpg"
    transparent_path = "/home/bhargav/VC/Human-Body-Measurements-using-Computer-Vision/input_data/steven_body/transparent.png"  # This is the output of the transparent mask
    final_out = "/home/bhargav/VC/Human-Body-Measurements-using-Computer-Vision/input_data/steven_body/final.png"

    transparent_non_zero_pixels(mask_path, transparent_path)
    overlay_images(original_path, transparent_path, final_out)

    #add comment for git testing
    #add comment for git-test-branch