import os
from PIL import Image
import numpy as np

def is_white(pixel, threshold=245):
    return all(channel > threshold for channel in pixel)

def crop_image(image_path, save_path, threshold=245, padding=5):
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)

    # Create a mask of non-white pixels
    mask = np.any(np_img < threshold, axis=-1)

    if not mask.any():
        print(f"[SKIP] All white: {image_path}")
        return

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Apply padding
    y0 = max(y0 - padding, 0)
    x0 = max(x0 - padding, 0)
    y1 = min(y1 + padding, np_img.shape[0])
    x1 = min(x1 + padding, np_img.shape[1])

    cropped_img = img.crop((x0, y0, x1, y1))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cropped_img.save(save_path)

def process_dataset(input_dir, output_dir, extensions=(".jpg", ".png", ".jpeg")):
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(extensions):
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
                crop_image(in_path, out_path)
    print("[DONE] Finished cropping and saving all images.")

# ======= RUN ========
if __name__ == "__main__":
    input_folder = "data/train"       # <- change to your dataset path
    output_folder = "data/train_cropped"

    process_dataset(input_folder, output_folder)
