# Instruction to run on kaggle notebook


```python
import requests
import os

# Replace url with link Google Drive link
drive_url = 'url'
'''
https://drive.google.com/uc?id=116OGkSfEFxcoAfcmAJ4Kj83oFANh7FgA&export=download&confirm=t&uuid=2b4102a9\-5972-416b-97eb-88ba28ee326d&at=AB6BwCAGuaEHjfdCyfAwGaV0E-O9:1700047389408
'''
# Directory where the downloaded file will be saved
save_dir = '/kaggle/working/'

# Send a get request to the drive_url
response = requests.get(drive_url)

# Write the content of the response to a file in the save_dir
with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)
```
Then install the lib
```python
!pip install segmentation-models-pytorch
```
Then clone my git repo and run
```python
!git clone https://github.com/WValleyy/DL3.git 
```
```python
!mkdir predicted_masks # make dir for mask prediction
```
```python
!python /kaggle/working/DL3/infer.py --path '/kaggle/working/model.pth' --test_dir '/kaggle/input/bkai-igh-neopolyp/test/test' --mask_dir '/kaggle/working/predicted_mask'

# parse args checkpoint, test_dir (please add data of competition), mask_dir

