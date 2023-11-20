import os
import sys
import segmentation_models_pytorch as smp

from PIL import Image
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
import model


class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform):
        super(UNetTestDataClass, self).__init__()

        images_list = os.listdir(images_path)
        images_list = [images_path + i for i in images_list]

        self.images_list = images_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data) / 255
        return data, img_path, h, w

    def __len__(self):
        return len(self.images_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PolypSegment Inference')
parser.add_argument('--path', type=str, help='Path to model checkpoint')
parser.add_argument('--test_dir', type=str, help='Path to test data')
parser.add_argument('--mask_dir', type=str, help='Directory path to save predicted masks')
args = parser.parse_args()

checkpoint = torch.load(args.path, map_location=device)
new_state_dict = {}
for key, value in checkpoint['model'].items():
    new_key = key.replace('module.', '')  # remove prefix 'module.'
    new_state_dict[new_key] = value
model = smp.UnetPlusPlus(
    encoder_name="resnet18",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)
model.to(device)
model.load_state_dict(new_state_dict)

color_mapping = {
    0: (0, 0, 0),  # Background
    1: (255, 0, 0),  # Neoplastic polyp
    2: (0, 255, 0)  # Non-neoplastic polyp
}


def mask_to_rgb(mask, color_mapping):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for key in color_mapping.keys():
        output[mask == key] = color_mapping[key]

    return np.uint8(output)

model.eval()

for idx, img_name in enumerate(os.listdir(path)):
    print(f'Predicted {idx + 1}/200 ...\r', end='')
    test_img_path = os.path.join(path, img_name)

    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    width, height = img.shape[1], img.shape[0]

    img = cv2.resize(img, (512, 512))

    # Transform the image
    transformed_img = transform(image=img)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        out_mask = model.forward(transformed_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)

    # Resize the mask to the original size
    out_mask = cv2.resize(out_mask, (width, height))
    out_mask = np.argmax(out_mask, axis=2)

    # Convert the mask to RGB
    rgb_mask = mask_to_rgb(out_mask, color_mapping)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)

    # Save the mask
    save_dir = '/kaggle/working/predict_mask'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, rgb_mask)

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:, :, channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = '/kaggle/working/predicted_masks'  # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)
