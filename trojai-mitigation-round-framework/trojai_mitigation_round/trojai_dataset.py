import json
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset


class Round11SampleDataset(Dataset):
    def __init__(self, root, img_exts=["jpg", "png"], class_info_ext="json", split='test', require_label=False):
        root = Path(root)
        train_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )

        test_augmentation_transforms = torchvision.transforms.Compose(
            [
                v2.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )
        if split == 'test':
            self.transform = test_augmentation_transforms
        else:
            self.transform = train_augmentation_transforms

        self._img_directory_contents = sorted([path for path in root.glob("*.*") if path.suffix[1:] in img_exts])

        self.data = []
        self.fnames = []
        for img_fname in self._img_directory_contents:
            full_path = root / img_fname
            
            if require_label:
                json_path = root / Path(img_fname.stem + f".{class_info_ext}")
                assert json_path.exists(), f"No {class_info_ext} found for {img_fname}"
                with open(json_path, 'r') as f:
                    label = json.load(f)
                    if isinstance(label, dict):
                        if 'clean' in str(root):
                            label = label['clean_label']
                        else:
                            label = label['poisoned_label']
            else:
                label = -1

            pil_img = Image.open(full_path)

            try:
                self.data.append((self.transform(pil_img), label))
                self.fnames.append(img_fname.name)
            except:
                continue


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        fname = self.fnames[idx]

        # img = self.transform(img)
        return img, label, fname