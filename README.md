# Greyscale_Image_Colorization

This model, at its core, leverages **PyTorch** for computer vision applications, specifically focusing on automatic image colorization. The model uses the **Microsoft Common Objects in Context (COCO)** dataset, which is the gold standard benchmark for evaluating the performance of state-of-the-art computer vision models.

## GAN and ResNet Architecture

This model utilizes a Generative Adversarial Network (GAN) with a ResNet18 architecture for the generative component (Goodfellow et al., 2014). The GAN has two neural networks: a generator and a discriminator, trained simultaneously via adversarial training. The generator creates images to fool the discriminator, which learns to distinguish between real and fake images (Isola et al., 2018).

The model employed a pre-trained ResNet18 model from PyTorch for the generator due to its effectiveness in image-based tasks (He et al., 2015). ResNet's “skip connector” architecture mitigates the vanishing and exploding gradient problems, allowing the model to access low-level features throughout the training process.

The discriminator has five layers (64, 128, 256, 512, output), with model weights initialized using Xavier, Kaiming, or normal methods to enhance performance (Xavier, 2010; He et al., 2015).

Training employed 10,000 images from the COCO test2017 dataset — 8,000 for training and 2,000 for validation.

Performance was measured using Mean Absolute Error (L1) for the generator and a combination of L1 and Binary Cross Entropy (BCE) for both networks. Qualitative assessments involved sampling predictions every two epochs for manual review against visual expectations. The abridged version, in the code section, sampled predictions after epochs 5 & 10 and then generated 'ground truth' vs 'greyscale' vs 'predicted' images. Training loops for this abridged version ran for 10 epochs. 


## Key Steps Include:

---

### 1. Library Installation & Imports:
Install necessary libraries and import required modules for building the image colorization model.

```python
# Example command to install libraries
!pip install torch torchvision matplotlib
```

---

### 2. Device Configuration:
Code to set the device for computations—Metal Performance Shaders (MPS), CUDA, and/or CPU, and print the selected device.

```python
import torch

# Check and set device
device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
```

---

### 3. Data Handling:
This involves mounting Google Drive to store and facilitate access to datasets stored there.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

### 4. Dataset Download and Preparation:
This step sets up the directory for the COCO dataset, downloading required images, and unzipping the dataset.

```python
# Example steps to download and unzip the COCO dataset
!wget -O coco_test2017.zip http://images.cocodataset.org/zips/test2017.zip
!unzip coco_test2017.zip -d ./coco_test2017
!rm coco_test2017.zip
```

---

### 5. Image Path Loading:
Load image paths for training and validation from the COCO dataset. Randomly select a sample of images and split them into training and validation sets.

```python
import os
import numpy as np

# Load image paths
coco_image_dir = './coco_test2017/test2017'
paths = [os.path.join(coco_image_dir, f) for f in os.listdir(coco_image_dir) if f.endswith('.jpg')]

# Randomly select and shuffle the dataset
np.random.seed(42)
paths_subset = np.random.choice(paths, 10000, replace=False)
rand_idxs = np.random.permutation(10000)
train_idxs = rand_idxs[:8000]
val_idxs = rand_idxs[8000:]
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
```

---

### 6. Dataset Class Definition:
Define a custom Dataset using the Microsoft COCO dataset. This includes initializing the Image Colorization Data class with image paths and size, loading and processing images, and providing the length of the dataset.

```python
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab

class ImageColorizationData(Dataset):
    def __init__(self, paths, size):
        self.transforms = transforms.Resize((size, size), Image.BICUBIC)
        self.size = size
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return L, ab

    def __len__(self):
        return len(self.paths)
```

---

### 7. DataLoader Creation:
Create DataLoader instances for both training and validation datasets with specified batch sizes and shuffling options.

```python
from torch.utils.data import DataLoader

# Create DataLoader instances
train_data = ImageColorizationData(train_paths, size=256)
val_data = ImageColorizationData(val_paths, size=256)

train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(val_data, batch_size=8, shuffle=False)
```

---

### 8. Generator Model Definition:
Build a ResNet-based UNet generator model for colorization.

```python
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from fastai.vision.learner import create_body

def build_res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18(pretrained=True), n_in
```


## Results

Here are some sample results from the image colorization model, using a 10,000-images dataset and training loops of 10 epochs:

### Image 1
<img width="1196" alt="Image 1" src="https://github.com/user-attachments/assets/d156e82f-8f47-4548-9dbf-ab52356dfa4a">


### Image 2
<img width="1192" alt="Image 2" src="https://github.com/user-attachments/assets/52851230-ee74-4a5a-9801-13c323e9bf14">


### Image 3
<img width="1196" alt="Image 3" src="https://github.com/user-attachments/assets/5053e811-09d6-4211-974b-d338f0c44408">



These images showcase the performance of the model on various samples from the COCO dataset, albeit on a very limited sample size and a short training loop of 10 epochs, due to limited local laptop processing power.
