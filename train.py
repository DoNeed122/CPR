import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel

from dataset import Dataset_Creator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_fft_filter(images, keep_low=True, ratio=0.3):
    original_dtype = images.dtype
    images_f32 = images.float()

    B, C, H, W = images_f32.shape
    fft_x = torch.fft.fftshift(torch.fft.fft2(images_f32))

    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(
        torch.arange(H, device=images_f32.device),
        torch.arange(W, device=images_f32.device),
        indexing="ij"
    )

    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    r = int(min(H, W) * ratio)

    mask = (dist_sq <= r ** 2).float()
    if not keep_low:
        mask = 1.0 - mask

    mask = mask.view(1, 1, H, W)
    fft_x = fft_x * mask

    filtered_images = torch.fft.ifft2(torch.fft.ifftshift(fft_x)).real
    return filtered_images.to(original_dtype), torch.ones(B, (H // 14) * (W // 14), device=images.device)


def build_attn_mask_from_outputs(images, outputs_orig, patch_size=14, mask_ratio=0.2, mask_highest=True):
    B, C, H, W = images.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    num_mask = int(num_patches * mask_ratio)

    last_attn = outputs_orig.attentions[-1].mean(dim=1)
    cls_attn = last_attn[:, 0, last_attn.shape[-1] - num_patches:]

    mask_idx = torch.argsort(cls_attn, dim=1, descending=mask_highest)[:, :num_mask]
    mask = torch.ones(B, num_patches, device=images.device)
    mask.scatter_(1, mask_idx, 0)

    mask_spatial = mask.reshape(B, 1, num_patches_h, num_patches_w)
    mask_spatial = F.interpolate(mask_spatial, size=(H, W), mode="nearest")

    return images * mask_spatial, mask


def apply_patch_noise(images, patch_size=14, noise_ratio=0.2):
    B, C, H, W = images.shape
    num_patches = (H // patch_size) * (W // patch_size)
    num_mask = int(num_patches * noise_ratio)

    noise = torch.rand(B, num_patches, device=images.device)
    mask_idx = torch.argsort(noise, dim=1)[:, :num_mask]

    mask = torch.ones(B, num_patches, device=images.device)
    mask.scatter_(1, mask_idx, 0)

    mask_spatial = F.interpolate(
        mask.reshape(B, 1, H // patch_size, W // patch_size),
        size=(H, W),
        mode="nearest"
    )

    gaussian_noise = torch.randn_like(images) * images.std() + images.mean()
    return (images * mask_spatial) + (gaussian_noise * (1 - mask_spatial)), mask


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_path,
        intensity=0.3,
        response_layer=-1,
        pooling="mean_max",
        keep_low=False,
        mask_highest=True
    ):
        super().__init__()

        print(f"=> Loading DINOv3 from local directory: {model_path}")
        self.backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.hidden_dim = self.backbone.config.hidden_size
        self.patch_size = getattr(self.backbone.config, "patch_size", 14)
        self.intensity = intensity
        self.response_layer = response_layer
        self.pooling = pooling
        self.keep_low = keep_low
        self.mask_highest = mask_highest

        print(f"=> Backbone hidden dim: {self.hidden_dim}")
        print(f"=> Patch size: {self.patch_size}")
        print(f"=> Perturbation intensity: {self.intensity}")
        print(f"=> Response layer: {self.response_layer}")
        print(f"=> Pooling: {self.pooling}")
        print(f"=> FFT keep_low: {self.keep_low}")
        print(f"=> Attention mask_highest: {self.mask_highest}")

    def _extract_spatial_features_from_hidden(self, hidden, images):
        B, C, H, W = images.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        extra_tokens = hidden.shape[1] - num_patches
        spatial = hidden[:, extra_tokens:, :].float()
        return spatial

    def _extract_spatial_features(self, images):
        outputs = self.backbone(images, output_hidden_states=True)
        hidden = outputs.hidden_states[self.response_layer]
        return self._extract_spatial_features_from_hidden(hidden, images)

    def _pool_response(self, response):
        if self.pooling == "mean":
            return response.mean(dim=1)
        if self.pooling == "mean_max":
            return torch.cat([response.mean(dim=1), response.amax(dim=1)], dim=-1)
        raise ValueError(f"Unknown pooling type: {self.pooling}")

    def forward(self, images):
        images = images.to(self.backbone.dtype)

        with torch.no_grad():
            outputs_orig = self.backbone(
                images,
                output_hidden_states=True,
                output_attentions=True
            )

            feat_orig = self._extract_spatial_features_from_hidden(
                outputs_orig.hidden_states[self.response_layer],
                images
            )

            images_fft, _ = apply_fft_filter(images, keep_low=self.keep_low, ratio=self.intensity)

            images_attn, _ = build_attn_mask_from_outputs(
                images=images,
                outputs_orig=outputs_orig,
                patch_size=self.patch_size,
                mask_ratio=self.intensity,
                mask_highest=self.mask_highest
            )

            images_noise, _ = apply_patch_noise(images, patch_size=self.patch_size, noise_ratio=self.intensity)

            feat_fft = self._extract_spatial_features(images_fft)
            feat_attn = self._extract_spatial_features(images_attn)
            feat_noise = self._extract_spatial_features(images_noise)

            r_fft = feat_fft - feat_orig
            r_attn = feat_attn - feat_orig
            r_noise = feat_noise - feat_orig

            pooled_fft = self._pool_response(r_fft)
            pooled_attn = self._pool_response(r_attn)
            pooled_noise = self._pool_response(r_noise)

        return pooled_fft, pooled_attn, pooled_noise


FEATURE_FILES = ["features_fft.npy", "features_attn.npy", "features_noise.npy", "labels.npy"]


def features_already_extracted(features_dir):
    if not os.path.isdir(features_dir):
        return False
    for name in FEATURE_FILES:
        path = os.path.join(features_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
    return True


def is_single_class_dir(path):
    try:
        inner = os.listdir(path)
    except OSError:
        return False
    return "0_real" in inner and "1_fake" in inner


def build_train_dataset(train_path, args):
    if is_single_class_dir(train_path):
        print(f"=> Detected a single dataset directory (0_real/1_fake): {train_path}")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(args.img_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.crop_resolution),
            transforms.ToTensor(),
            normalize,
        ])
        return ImageFolder(train_path, transform=transform)

    creator = Dataset_Creator(
        dataset_path=train_path,
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        img_resolution=args.img_resolution,
        crop_resolution=args.crop_resolution
    )
    return creator.build_dataset(args.split)


def extract_features(args, device):
    features_dir = args.features_path

    print("=" * 60)
    print("=> [Stage 1] Extracting perturbation response vectors from the training set")
    print(f"=> Training path: {args.train_path}")
    print(f"=> Feature output dir: {features_dir}")
    print("=" * 60)

    os.makedirs(features_dir, exist_ok=True)

    with open(os.path.join(features_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    train_dataset = build_train_dataset(args.train_path, args)
    total_samples = len(train_dataset)
    print(f"=> Training set size: {total_samples}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    extractor = DINOv3FeatureExtractor(
        model_path=args.model_path,
        intensity=args.intensity,
        response_layer=args.response_layer,
        pooling=args.pooling,
        keep_low=args.keep_low,
        mask_highest=args.mask_highest
    ).to(device)
    extractor.eval()

    feature_dim = 2 * extractor.hidden_dim if args.pooling == "mean_max" else extractor.hidden_dim
    print(f"=> Feature dim: {feature_dim}")

    estimated_size = total_samples * 3 * feature_dim * 4 / (1024 ** 3)
    print(f"=> Estimated disk usage: {estimated_size:.2f} GB")

    output_fft = os.path.join(features_dir, "features_fft.npy")
    output_attn = os.path.join(features_dir, "features_attn.npy")
    output_noise = os.path.join(features_dir, "features_noise.npy")
    output_labels = os.path.join(features_dir, "labels.npy")
    output_dataset_ids = os.path.join(features_dir, "dataset_ids.npy")

    print("=> Creating memory-mapped files...")
    features_fft_mmap = np.memmap(output_fft, dtype='float32', mode='w+', shape=(total_samples, feature_dim))
    features_attn_mmap = np.memmap(output_attn, dtype='float32', mode='w+', shape=(total_samples, feature_dim))
    features_noise_mmap = np.memmap(output_noise, dtype='float32', mode='w+', shape=(total_samples, feature_dim))
    labels_mmap = np.memmap(output_labels, dtype='int64', mode='w+', shape=(total_samples,))
    dataset_ids_mmap = np.memmap(output_dataset_ids, dtype='int32', mode='w+', shape=(total_samples,))

    print("=> Extracting features...")
    current_idx = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Extracting")):
            batch_n = images.size(0)
            start_idx = current_idx
            end_idx = current_idx + batch_n

            images = images.to(device)
            pooled_fft, pooled_attn, pooled_noise = extractor(images)

            features_fft_mmap[start_idx:end_idx] = pooled_fft.cpu().numpy()
            features_attn_mmap[start_idx:end_idx] = pooled_attn.cpu().numpy()
            features_noise_mmap[start_idx:end_idx] = pooled_noise.cpu().numpy()
            labels_mmap[start_idx:end_idx] = labels.numpy()
            dataset_ids_mmap[start_idx:end_idx] = 0

            del images, pooled_fft, pooled_attn, pooled_noise
            current_idx = end_idx

            if (batch_idx + 1) % 50 == 0:
                features_fft_mmap.flush()
                features_attn_mmap.flush()
                features_noise_mmap.flush()
                labels_mmap.flush()
                dataset_ids_mmap.flush()
                torch.cuda.empty_cache()

    features_fft_mmap.flush()
    features_attn_mmap.flush()
    features_noise_mmap.flush()
    labels_mmap.flush()
    dataset_ids_mmap.flush()

    print("=> Features saved to:")
    total_size = 0.0
    for f in [output_fft, output_attn, output_noise, output_labels, output_dataset_ids]:
        size = os.path.getsize(f) / (1024 ** 3)
        total_size += size
        print(f"   {os.path.basename(f)}: {size:.2f} GB")
    print(f"=> Total file size: {total_size:.2f} GB")
    print(f"=> Total samples: {total_samples}")
    print("=> [Stage 1] Feature extraction done!\n")


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, features_dir, feature_dim=8192):
        print("=> Scanning precomputed features...")

        labels_path = os.path.join(features_dir, "labels.npy")
        labels_file_size = os.path.getsize(labels_path)
        total_samples = labels_file_size // 8

        print(f"=> Total samples: {total_samples}, feature dim: {feature_dim}")

        dataset_ids_path = os.path.join(features_dir, "dataset_ids.npy")
        if os.path.exists(dataset_ids_path):
            dataset_ids = np.memmap(
                dataset_ids_path, dtype='int32', mode='r', shape=(total_samples,)
            )
            self.genimage_indices = np.where(np.asarray(dataset_ids) == 0)[0].copy()
            del dataset_ids
        else:
            self.genimage_indices = np.arange(total_samples)

        print(f"=> Usable samples: {len(self.genimage_indices)}")

        self.features_dir = features_dir
        self.feature_dim = feature_dim
        self.total_samples = total_samples

        self._features_fft = None
        self._features_attn = None
        self._features_noise = None
        self._labels = None

        labels = np.memmap(
            labels_path, dtype='int64', mode='r', shape=(total_samples,)
        )
        genimage_labels = np.asarray(labels[self.genimage_indices])
        n_real = int(np.sum(genimage_labels == 0))
        n_fake = int(np.sum(genimage_labels == 1))
        del labels

        print(f"   - Real: {n_real}")
        print(f"   - Fake: {n_fake}")

    def _lazy_init(self):
        if self._features_fft is None:
            self._features_fft = np.memmap(
                os.path.join(self.features_dir, "features_fft.npy"),
                dtype='float32', mode='r', shape=(self.total_samples, self.feature_dim)
            )
            self._features_attn = np.memmap(
                os.path.join(self.features_dir, "features_attn.npy"),
                dtype='float32', mode='r', shape=(self.total_samples, self.feature_dim)
            )
            self._features_noise = np.memmap(
                os.path.join(self.features_dir, "features_noise.npy"),
                dtype='float32', mode='r', shape=(self.total_samples, self.feature_dim)
            )
            self._labels = np.memmap(
                os.path.join(self.features_dir, "labels.npy"),
                dtype='int64', mode='r', shape=(self.total_samples,)
            )

    def __len__(self):
        return len(self.genimage_indices)

    def __getitem__(self, idx):
        self._lazy_init()
        real_idx = self.genimage_indices[idx]
        return (
            self._features_fft[real_idx].copy(),
            self._features_attn[real_idx].copy(),
            self._features_noise[real_idx].copy(),
            self._labels[real_idx]
        )


class MultiHeadClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes=2):
        super().__init__()

        self.classifier_fft = nn.Linear(feature_dim, num_classes)
        self.classifier_attn = nn.Linear(feature_dim, num_classes)
        self.classifier_noise = nn.Linear(feature_dim, num_classes)

        print(f"=> Classifier input dim: {feature_dim}")

    def forward(self, feat_fft, feat_attn, feat_noise):
        logits_fft = self.classifier_fft(feat_fft)
        logits_attn = self.classifier_attn(feat_attn)
        logits_noise = self.classifier_noise(feat_noise)

        return logits_fft, logits_attn, logits_noise


def train_heads(args, device):
    print("=" * 60)
    print("=> [Stage 2] Training the three classification heads")
    print(f"=> Feature dir: {args.features_path}")
    print(f"=> Checkpoint output dir: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PrecomputedFeatureDataset(args.features_path, feature_dim=args.feature_dim)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = MultiHeadClassifier(feature_dim=args.feature_dim).to(device)

    optimizer = optim.AdamW([
        {"params": model.classifier_fft.parameters()},
        {"params": model.classifier_attn.parameters()},
        {"params": model.classifier_noise.parameters()},
    ], lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    print(f"=> Config: Epochs={args.epochs}, BS={args.batch_size}, LR={args.lr}")

    best_acc = 0.0
    best_path = os.path.join(args.output_dir, "classifier_best.pth")
    last_path = os.path.join(args.output_dir, "classifier_last.pth")

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        correct_fft = correct_attn = correct_noise = correct_ensemble = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for feat_fft, feat_attn, feat_noise, labels in progress_bar:
            feat_fft = feat_fft.float().to(device)
            feat_attn = feat_attn.float().to(device)
            feat_noise = feat_noise.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad(set_to_none=True)

            logits_fft, logits_attn, logits_noise = model(feat_fft, feat_attn, feat_noise)

            loss_fft = criterion(logits_fft, labels)
            loss_attn = criterion(logits_attn, labels)
            loss_noise = criterion(logits_noise, labels)
            loss = loss_fft + loss_attn + loss_noise

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred_fft = logits_fft.argmax(dim=1)
            pred_attn = logits_attn.argmax(dim=1)
            pred_noise = logits_noise.argmax(dim=1)

            pred_ensemble = ((pred_fft + pred_attn + pred_noise) >= 2).long()

            total += labels.size(0)
            correct_fft += (pred_fft == labels).sum().item()
            correct_attn += (pred_attn == labels).sum().item()
            correct_noise += (pred_noise == labels).sum().item()
            correct_ensemble += (pred_ensemble == labels).sum().item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Ens": f"{100 * correct_ensemble / total:.2f}%"
            })

        epoch_acc_fft = 100 * correct_fft / total
        epoch_acc_attn = 100 * correct_attn / total
        epoch_acc_noise = 100 * correct_noise / total
        epoch_acc_ensemble = 100 * correct_ensemble / total

        print(f">> Epoch {epoch + 1} | FFT: {epoch_acc_fft:.2f}% | Attn: {epoch_acc_attn:.2f}% | "
              f"Noise: {epoch_acc_noise:.2f}% | Ensemble: {epoch_acc_ensemble:.2f}%")

        if epoch_acc_ensemble >= best_acc:
            best_acc = epoch_acc_ensemble
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_acc": best_acc,
            }, best_path)
            print(f"=> Saved best checkpoint: {best_path}")

    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "best_acc": best_acc,
    }, last_path)

    print(f"=> Training done! Final checkpoint: {last_path}")
    print(f"=> Best accuracy: {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Extract perturbation response vectors and train the multi-head classifier")

    parser.add_argument("--train_path", type=str, default="/home/don/dev/Datasets/GenImage",
                        help="Training data path. Either a dataset root containing <split>/<subset>/0_real|1_fake, "
                             "or a single directory that directly contains 0_real and 1_fake.")
    parser.add_argument("--split", type=str, default="train",
                        help="Split name used when --train_path is a dataset root")
    parser.add_argument("--features_path", type=str, default="./extracted_features_genimage_train",
                        help="Directory to save the extracted training features")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save the classifier checkpoints")
    parser.add_argument("--model_path", type=str,
                        default="/home/don/dev/code/Dinov3/dinov3-vit7b16-pretrain-lvd1689m",
                        help="DINOv3 backbone directory")

    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip extraction and train on existing features")
    parser.add_argument("--force_extract", action="store_true",
                        help="Re-extract features even if they already exist")
    parser.add_argument("--skip_train", action="store_true",
                        help="Only extract features, do not train")

    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--img_resolution", type=int, default=256)
    parser.add_argument("--crop_resolution", type=int, default=224)
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument("--response_layer", type=int, default=-1,
                        help="Index of the hidden state used as feature, must match the evaluation script")
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "mean_max"])
    parser.add_argument("--keep_low", type=str2bool, default=False)
    parser.add_argument("--mask_highest", type=str2bool, default=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--feature_dim", type=int, default=8192,
                        help="Feature dimension, 2*hidden_dim when pooling is mean_max")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Device: {device}")

    if args.skip_extract:
        print("=> --skip_extract given, skipping feature extraction")
    elif (not args.force_extract) and features_already_extracted(args.features_path):
        print("=" * 60)
        print("=> Feature directory already exists, skipping feature extraction")
        print(f"=> Dir: {args.features_path}")
        for name in FEATURE_FILES:
            path = os.path.join(args.features_path, name)
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"   {name}: {size_gb:.2f} GB")
        print("=> (Use --force_extract to re-extract)")
        print("=" * 60)
    else:
        extract_features(args, device)

    if args.skip_train:
        print("=> --skip_train given, exiting")
        return

    train_heads(args, device)


if __name__ == "__main__":
    main()
