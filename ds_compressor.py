#non comprime un tubo anche cambiando il valore di compression_quality
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing

from utils import find_images

def compress_image(input_path, output_path, compression_quality):
    with Image.open(input_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(output_path, optimize=True, quality=compression_quality)

if __name__ == "__main__":

    input_dir = '/Users/simone/Desktop/unit/autoencoder/rightImg8bit_trainvaltest/rightImg8bit/train/'
    output_dir = '/Users/simone/Desktop/unit/autoencoder/dataset/train/'
    compression_quality = 5  # adjust this value as needed

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pool = multiprocessing.Pool()

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.png'):
                input_path = os.path.join(root, filename)
                output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))
                output_path = os.path.join(output_subdir, filename)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                pool.apply_async(compress_image, args=(input_path, output_path, compression_quality))

    pool.close()
    pool.join()
