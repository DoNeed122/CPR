import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset_Creator:
    def __init__(self, dataset_path, batch_size=128, num_workers=4, img_resolution=256, crop_resolution=224):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        simple_transform = transforms.Compose([
            transforms.Resize(img_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_resolution),
            transforms.ToTensor(),
            normalize,
        ])

        self.transforms = {
            "train": simple_transform,
            "val": simple_transform,
            "test": simple_transform
        }

    def build_dataset(self, split_dataset, selected_subsets="all"):
        if split_dataset == 'tta':
            split_dataset = 'test'

        assert split_dataset in ['train', 'val', 'test']

        if selected_subsets == "all":
            if split_dataset == "train":
                selected_subsets = ['SDv14']
            else:
                selected_subsets = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong']

        if isinstance(selected_subsets, str):
            selected_subsets = [selected_subsets]

        sub_datasets = []
        for subset in selected_subsets:
            subset_path = os.path.join(self.dataset_path, split_dataset, subset)

            if not os.path.exists(subset_path):
                print(f"[Warning] Path does not exist, skipped: {subset_path}")
                continue

            available_dirs = os.listdir(subset_path)

            if "0_real" in available_dirs and "1_fake" in available_dirs:
                sub_datasets.append(ImageFolder(subset_path, self.transforms[split_dataset]))

            else:
                tmp_datasets = []
                for sub_class in available_dirs:
                    sub_class_path = os.path.join(subset_path, sub_class)
                    if os.path.isdir(sub_class_path) and ("0_real" in os.listdir(sub_class_path) or "1_fake" in os.listdir(sub_class_path)):
                        tmp_datasets.append(ImageFolder(sub_class_path, self.transforms[split_dataset]))
                if tmp_datasets:
                    sub_datasets.append(ConcatDataset(tmp_datasets))

        if split_dataset == "test":
            return sub_datasets, selected_subsets

        if len(sub_datasets) == 0:
            raise FileNotFoundError(f"No valid 0_real/1_fake data found under {self.dataset_path}, please check the path!")

        return ConcatDataset(sub_datasets)

class Dataset_Creator_GenImage(Dataset_Creator): pass
class Dataset_Creator_Chameleon(Dataset_Creator): pass
class Dataset_Creator_Chameleon_SD(Dataset_Creator): pass
