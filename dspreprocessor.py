#resize the dataset images and labels to 512 x 256
import os
from PIL import Image
from tqdm import tqdm

from utils import find_images

if __name__ == "__main__":

    images_path = 'rightImg8bit_trainvaltest/rightImg8bit'
    labels_path = 'gtFine_trainvaltest/gtFine'

    input_img_paths = find_images(images_path)

    print("Resizing images...")
    for img_path in tqdm(input_img_paths):
        rel_path = os.path.join(images_path, img_path)
        try:
            img = Image.open(rel_path)
            img = img.resize((512, 256))
            img.save(rel_path)
        except:
            print(f"Error while processing {rel_path}")

    print("Resizing labels...")
    for img_path in tqdm(input_img_paths):
        lbl_path = img_path.replace("rightImg8bit", "gtFine_labelIds")
        rel_path = os.path.join(labels_path, lbl_path)
        try:
            img = Image.open(rel_path)
            img = img.resize((512, 256))
            img.save(rel_path)
        except:
            print(f"Error while processing {rel_path}")

    print("Done!")

    
                
