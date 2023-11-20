import os
import sys
import segmentation_models_pytorch as smp

from PIL import Image
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import model
import argparse

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
test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
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
testsize = 384
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

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)    

model.eval()
for i in os.listdir(args.test_dir):
    img_path = os.path.join(args.test_dir, i)
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (testsize, testsize))
    transformed = test_transform(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("predicted_masks/{}".format(i), mask_rgb) 
    
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 225] = 255
    pixels[pixels <= 225] = 0
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

def rle2mask(mask_rle, shape=(3,3)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2string(dir):
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = args.mask_dir # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)
