import os
import argparse
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import AutoModel
from sklearn.metrics import average_precision_score

from train_genimage import MultiHeadClassifier, str2bool, set_seed


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
        return hidden[:, extra_tokens:, :].float()

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


def subset_features_ready(subset_dir):
    if not os.path.isdir(subset_dir):
        return False
    for name in FEATURE_FILES:
        path = os.path.join(subset_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
    return True


def is_single_class_dir(path):
    try:
        inner = os.listdir(path)
    except OSError:
        return False
    return "0_real" in inner and "1_fake" in inner


def discover_subsets(test_path):
    if is_single_class_dir(test_path):
        print(f"=> Detected a single dataset directory (0_real/1_fake): {test_path}")
        return [(os.path.basename(os.path.normpath(test_path)), test_path)]

    subdirs = sorted(os.listdir(test_path))
    print(f"=> Found {len(subdirs)} subdirectories: {subdirs}")

    subsets = []
    for subset_name in subdirs:
        subset_path = os.path.join(test_path, subset_name)
        if not os.path.isdir(subset_path):
            print(f"[Warning] Skipped non-directory: {subset_name}")
            continue
        if not is_single_class_dir(subset_path):
            print(f"[Warning] Skipped {subset_name}: no 0_real/1_fake inside")
            continue
        subsets.append((subset_name, subset_path))
    return subsets


def extract_features_for_subset(subset_name, subset_path, extractor, args, device):
    print(f"\n{'=' * 60}")
    print(f"=> Processing subset: {subset_name}")
    print(f"{'=' * 60}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(args.img_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.crop_resolution),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = ImageFolder(subset_path, transform=transform)
    num_samples = len(dataset)
    print(f"=> Samples: {num_samples}")

    if num_samples == 0:
        print(f"[Warning] Subset {subset_name} is empty, skipped")
        return

    subset_output_dir = os.path.join(args.output_dir, subset_name)
    os.makedirs(subset_output_dir, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=args.extract_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    feature_dim = 2 * extractor.hidden_dim if args.pooling == "mean_max" else extractor.hidden_dim
    estimated_size = num_samples * 3 * feature_dim * 4 / (1024 ** 3)
    print(f"=> Feature dim: {feature_dim}")
    print(f"=> Estimated disk usage: {estimated_size:.2f} GB")

    output_fft = os.path.join(subset_output_dir, "features_fft.npy")
    output_attn = os.path.join(subset_output_dir, "features_attn.npy")
    output_noise = os.path.join(subset_output_dir, "features_noise.npy")
    output_labels = os.path.join(subset_output_dir, "labels.npy")

    features_fft_mmap = np.memmap(output_fft, dtype='float32', mode='w+', shape=(num_samples, feature_dim))
    features_attn_mmap = np.memmap(output_attn, dtype='float32', mode='w+', shape=(num_samples, feature_dim))
    features_noise_mmap = np.memmap(output_noise, dtype='float32', mode='w+', shape=(num_samples, feature_dim))
    labels_mmap = np.memmap(output_labels, dtype='int64', mode='w+', shape=(num_samples,))

    current_idx = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Extracting {subset_name}")):
            batch_n = images.size(0)
            start_idx = current_idx
            end_idx = current_idx + batch_n

            images = images.to(device)
            pooled_fft, pooled_attn, pooled_noise = extractor(images)

            features_fft_mmap[start_idx:end_idx] = pooled_fft.cpu().numpy()
            features_attn_mmap[start_idx:end_idx] = pooled_attn.cpu().numpy()
            features_noise_mmap[start_idx:end_idx] = pooled_noise.cpu().numpy()
            labels_mmap[start_idx:end_idx] = labels.numpy()

            del images, pooled_fft, pooled_attn, pooled_noise
            current_idx = end_idx

            if (batch_idx + 1) % 50 == 0:
                features_fft_mmap.flush()
                features_attn_mmap.flush()
                features_noise_mmap.flush()
                labels_mmap.flush()
                torch.cuda.empty_cache()

    features_fft_mmap.flush()
    features_attn_mmap.flush()
    features_noise_mmap.flush()
    labels_mmap.flush()

    total_size = 0.0
    for f in [output_fft, output_attn, output_noise, output_labels]:
        total_size += os.path.getsize(f) / (1024 ** 3)

    print(f"=> Saved to: {subset_output_dir}")
    print(f"=> Total file size: {total_size:.2f} GB")


def extract_features(args, device):
    print("=" * 60)
    print("=> [Stage 1] Extracting perturbation response vectors from the test set")
    print(f"=> Test path: {args.genimage_test_path}")
    print(f"=> Feature output dir: {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    subsets = discover_subsets(args.genimage_test_path)

    todo = []
    for subset_name, subset_path in subsets:
        subset_output_dir = os.path.join(args.output_dir, subset_name)
        if (not args.force_extract) and subset_features_ready(subset_output_dir):
            print(f"=> Already exists, skipped: {subset_name}")
            continue
        todo.append((subset_name, subset_path))

    if not todo:
        print("=> All subsets already have features, skipping extraction")
        return

    extractor = DINOv3FeatureExtractor(
        model_path=args.model_path,
        intensity=args.intensity,
        response_layer=args.response_layer,
        pooling=args.pooling,
        keep_low=args.keep_low,
        mask_highest=args.mask_highest
    ).to(device)
    extractor.eval()

    for subset_name, subset_path in todo:
        extract_features_for_subset(subset_name, subset_path, extractor, args, device)
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("=> [Stage 1] All subsets extracted!")
    print(f"=> Output dir: {args.output_dir}")
    print("=" * 60)


def load_features(subset_dir, feature_dim=8192):
    labels_path = os.path.join(subset_dir, "labels.npy")
    labels_file_size = os.path.getsize(labels_path)
    num_samples = labels_file_size // 8

    features_fft = np.memmap(
        os.path.join(subset_dir, "features_fft.npy"),
        dtype='float32', mode='r', shape=(num_samples, feature_dim)
    )
    features_attn = np.memmap(
        os.path.join(subset_dir, "features_attn.npy"),
        dtype='float32', mode='r', shape=(num_samples, feature_dim)
    )
    features_noise = np.memmap(
        os.path.join(subset_dir, "features_noise.npy"),
        dtype='float32', mode='r', shape=(num_samples, feature_dim)
    )
    labels = np.memmap(
        labels_path, dtype='int64', mode='r', shape=(num_samples,)
    )

    return features_fft, features_attn, features_noise, labels


def evaluate_single_subset(model, features_fft, features_attn, features_noise, labels,
                           batch_size, device, subset_name):
    num_samples = len(labels)

    model.eval()

    all_labels = []
    all_probs_fft = []
    all_probs_attn = []
    all_probs_noise = []

    all_preds_fft = []
    all_preds_attn = []
    all_preds_noise = []
    all_preds_ensemble = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc=f"Eval {subset_name}", leave=False):
            end_idx = min(i + batch_size, num_samples)

            batch_fft = torch.from_numpy(np.asarray(features_fft[i:end_idx])).to(device)
            batch_attn = torch.from_numpy(np.asarray(features_attn[i:end_idx])).to(device)
            batch_noise = torch.from_numpy(np.asarray(features_noise[i:end_idx])).to(device)
            batch_labels = torch.from_numpy(np.asarray(labels[i:end_idx])).to(device)

            logits_fft, logits_attn, logits_noise = model(batch_fft, batch_attn, batch_noise)

            probs_fft = F.softmax(logits_fft, dim=1)[:, 1]
            probs_attn = F.softmax(logits_attn, dim=1)[:, 1]
            probs_noise = F.softmax(logits_noise, dim=1)[:, 1]

            pred_fft = logits_fft.argmax(dim=1)
            pred_attn = logits_attn.argmax(dim=1)
            pred_noise = logits_noise.argmax(dim=1)

            pred_ensemble = ((pred_fft + pred_attn + pred_noise) >= 2).long()

            all_labels.extend(batch_labels.cpu().numpy())
            all_probs_fft.extend(probs_fft.cpu().numpy())
            all_probs_attn.extend(probs_attn.cpu().numpy())
            all_probs_noise.extend(probs_noise.cpu().numpy())

            all_preds_fft.extend(pred_fft.cpu().numpy())
            all_preds_attn.extend(pred_attn.cpu().numpy())
            all_preds_noise.extend(pred_noise.cpu().numpy())
            all_preds_ensemble.extend(pred_ensemble.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs_fft = np.array(all_probs_fft)
    all_probs_attn = np.array(all_probs_attn)
    all_probs_noise = np.array(all_probs_noise)

    all_preds_fft = np.array(all_preds_fft)
    all_preds_attn = np.array(all_preds_attn)
    all_preds_noise = np.array(all_preds_noise)
    all_preds_ensemble = np.array(all_preds_ensemble)

    acc_fft = 100 * (all_preds_fft == all_labels).mean()
    acc_attn = 100 * (all_preds_attn == all_labels).mean()
    acc_noise = 100 * (all_preds_noise == all_labels).mean()
    acc_ensemble = 100 * (all_preds_ensemble == all_labels).mean()

    real_mask = all_labels == 0
    fake_mask = all_labels == 1

    n_real = int(real_mask.sum())
    n_fake = int(fake_mask.sum())

    acc_real_fft = 100 * (all_preds_fft[real_mask] == 0).mean() if n_real > 0 else 0
    acc_fake_fft = 100 * (all_preds_fft[fake_mask] == 1).mean() if n_fake > 0 else 0

    acc_real_attn = 100 * (all_preds_attn[real_mask] == 0).mean() if n_real > 0 else 0
    acc_fake_attn = 100 * (all_preds_attn[fake_mask] == 1).mean() if n_fake > 0 else 0

    acc_real_noise = 100 * (all_preds_noise[real_mask] == 0).mean() if n_real > 0 else 0
    acc_fake_noise = 100 * (all_preds_noise[fake_mask] == 1).mean() if n_fake > 0 else 0

    acc_real_ensemble = 100 * (all_preds_ensemble[real_mask] == 0).mean() if n_real > 0 else 0
    acc_fake_ensemble = 100 * (all_preds_ensemble[fake_mask] == 1).mean() if n_fake > 0 else 0

    ap_fft = 100 * average_precision_score(all_labels, all_probs_fft)
    ap_attn = 100 * average_precision_score(all_labels, all_probs_attn)
    ap_noise = 100 * average_precision_score(all_labels, all_probs_noise)

    probs_ensemble = (all_probs_fft + all_probs_attn + all_probs_noise) / 3
    ap_ensemble = 100 * average_precision_score(all_labels, probs_ensemble)

    is_chameleon = subset_name.lower() == 'chameleon'

    acc_balanced_fft = (acc_real_fft + acc_fake_fft) / 2
    acc_balanced_attn = (acc_real_attn + acc_fake_attn) / 2
    acc_balanced_noise = (acc_real_noise + acc_fake_noise) / 2
    acc_balanced_ensemble = (acc_real_ensemble + acc_fake_ensemble) / 2

    return {
        "subset": subset_name,
        "total": len(all_labels),
        "n_real": n_real,
        "n_fake": n_fake,
        "is_chameleon": is_chameleon,
        "acc_fft": acc_fft,
        "acc_attn": acc_attn,
        "acc_noise": acc_noise,
        "acc_ensemble": acc_ensemble,
        "acc_balanced_fft": acc_balanced_fft,
        "acc_balanced_attn": acc_balanced_attn,
        "acc_balanced_noise": acc_balanced_noise,
        "acc_balanced_ensemble": acc_balanced_ensemble,
        "acc_real_fft": acc_real_fft,
        "acc_fake_fft": acc_fake_fft,
        "acc_real_attn": acc_real_attn,
        "acc_fake_attn": acc_fake_attn,
        "acc_real_noise": acc_real_noise,
        "acc_fake_noise": acc_fake_noise,
        "acc_real_ensemble": acc_real_ensemble,
        "acc_fake_ensemble": acc_fake_ensemble,
        "ap_fft": ap_fft,
        "ap_attn": ap_attn,
        "ap_noise": ap_noise,
        "ap_ensemble": ap_ensemble,
    }


def evaluate(args, device):
    print("=" * 60)
    print("=> [Stage 2] Evaluation")
    print(f"=> Feature dir: {args.features_dir}")
    print(f"=> Checkpoint: {args.checkpoint}")
    print("=" * 60)

    print(f"=> Loading classifier: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    saved_args = checkpoint.get("args", {})
    feature_dim = saved_args.get("feature_dim", args.feature_dim)
    print(f"=> Feature dim: {feature_dim}")

    model = MultiHeadClassifier(feature_dim=feature_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if "best_acc" in checkpoint:
        print(f"=> Best accuracy during training: {checkpoint['best_acc']:.2f}%")

    subsets = sorted([d for d in os.listdir(args.features_dir)
                      if os.path.isdir(os.path.join(args.features_dir, d))])
    print(f"=> Found {len(subsets)} subsets")

    results = []
    for subset_name in subsets:
        subset_path = os.path.join(args.features_dir, subset_name)

        if not all(os.path.exists(os.path.join(subset_path, f)) for f in FEATURE_FILES):
            print(f"Skipped {subset_name}: missing feature files")
            continue

        print(f"\n=> Evaluating: {subset_name}")

        features_fft, features_attn, features_noise, labels = load_features(
            subset_path, feature_dim=feature_dim
        )

        result = evaluate_single_subset(
            model=model,
            features_fft=features_fft,
            features_attn=features_attn,
            features_noise=features_noise,
            labels=labels,
            batch_size=args.batch_size,
            device=device,
            subset_name=subset_name
        )

        results.append(result)
        print(f"   FFT: Acc={result['acc_fft']:.2f}%, Real={result['acc_real_fft']:.2f}%, Fake={result['acc_fake_fft']:.2f}%")
        print(f"   Attn: Acc={result['acc_attn']:.2f}%, Real={result['acc_real_attn']:.2f}%, Fake={result['acc_fake_attn']:.2f}%")
        print(f"   Noise: Acc={result['acc_noise']:.2f}%, Real={result['acc_real_noise']:.2f}%, Fake={result['acc_fake_noise']:.2f}%")
        print(f"   Ensemble: Acc={result['acc_ensemble']:.2f}%, Real={result['acc_real_ensemble']:.2f}%, Fake={result['acc_fake_ensemble']:.2f}%")

        del features_fft, features_attn, features_noise, labels

    if not results:
        print("=> No subset to evaluate")
        return

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    header = f"{'Subset':<20} {'Acc_FFT':>8} {'AP_FFT':>8} {'Acc_Attn':>8} {'AP_Attn':>8} " \
             f"{'Acc_Noise':>8} {'AP_Noise':>8} {'Acc_Ens':>8} {'AP_Ens':>8}"
    print(header)
    print("-" * 100)

    for r in results:
        print(f"{r['subset']:<20} {r['acc_fft']:>7.2f}% {r['ap_fft']:>7.2f}% "
              f"{r['acc_attn']:>7.2f}% {r['ap_attn']:>7.2f}% "
              f"{r['acc_noise']:>7.2f}% {r['ap_noise']:>7.2f}% "
              f"{r['acc_ensemble']:>7.2f}% {r['ap_ensemble']:>7.2f}%")

    avg_acc_fft = np.mean([r['acc_fft'] for r in results])
    avg_ap_fft = np.mean([r['ap_fft'] for r in results])
    avg_acc_attn = np.mean([r['acc_attn'] for r in results])
    avg_ap_attn = np.mean([r['ap_attn'] for r in results])
    avg_acc_noise = np.mean([r['acc_noise'] for r in results])
    avg_ap_noise = np.mean([r['ap_noise'] for r in results])
    avg_acc_ensemble = np.mean([r['acc_ensemble'] for r in results])
    avg_ap_ensemble = np.mean([r['ap_ensemble'] for r in results])

    avg_acc_real_fft = np.mean([r['acc_real_fft'] for r in results])
    avg_acc_fake_fft = np.mean([r['acc_fake_fft'] for r in results])
    avg_acc_real_attn = np.mean([r['acc_real_attn'] for r in results])
    avg_acc_fake_attn = np.mean([r['acc_fake_attn'] for r in results])
    avg_acc_real_noise = np.mean([r['acc_real_noise'] for r in results])
    avg_acc_fake_noise = np.mean([r['acc_fake_noise'] for r in results])
    avg_acc_real_ensemble = np.mean([r['acc_real_ensemble'] for r in results])
    avg_acc_fake_ensemble = np.mean([r['acc_fake_ensemble'] for r in results])

    print("-" * 100)
    print(f"{'Average':<20} {avg_acc_fft:>7.2f}% {avg_ap_fft:>7.2f}% "
          f"{avg_acc_attn:>7.2f}% {avg_ap_attn:>7.2f}% "
          f"{avg_acc_noise:>7.2f}% {avg_ap_noise:>7.2f}% "
          f"{avg_acc_ensemble:>7.2f}% {avg_ap_ensemble:>7.2f}%")
    print("=" * 100)

    print("\nReal/Fake accuracy:")
    print(f"  FFT: Real={avg_acc_real_fft:.2f}%, Fake={avg_acc_fake_fft:.2f}%")
    print(f"  Attn: Real={avg_acc_real_attn:.2f}%, Fake={avg_acc_fake_attn:.2f}%")
    print(f"  Noise: Real={avg_acc_real_noise:.2f}%, Fake={avg_acc_fake_noise:.2f}%")
    print(f"  Ensemble: Real={avg_acc_real_ensemble:.2f}%, Fake={avg_acc_fake_ensemble:.2f}%")

    results_no_chameleon = [r for r in results if r['subset'].lower() != 'chameleon']

    if results_no_chameleon:
        avg_acc_fft_no_cham = np.mean([r['acc_fft'] for r in results_no_chameleon])
        avg_ap_fft_no_cham = np.mean([r['ap_fft'] for r in results_no_chameleon])
        avg_acc_attn_no_cham = np.mean([r['acc_attn'] for r in results_no_chameleon])
        avg_ap_attn_no_cham = np.mean([r['ap_attn'] for r in results_no_chameleon])
        avg_acc_noise_no_cham = np.mean([r['acc_noise'] for r in results_no_chameleon])
        avg_ap_noise_no_cham = np.mean([r['ap_noise'] for r in results_no_chameleon])
        avg_acc_ensemble_no_cham = np.mean([r['acc_ensemble'] for r in results_no_chameleon])
        avg_ap_ensemble_no_cham = np.mean([r['ap_ensemble'] for r in results_no_chameleon])

        avg_acc_real_fft_no_cham = np.mean([r['acc_real_fft'] for r in results_no_chameleon])
        avg_acc_fake_fft_no_cham = np.mean([r['acc_fake_fft'] for r in results_no_chameleon])
        avg_acc_real_attn_no_cham = np.mean([r['acc_real_attn'] for r in results_no_chameleon])
        avg_acc_fake_attn_no_cham = np.mean([r['acc_fake_attn'] for r in results_no_chameleon])
        avg_acc_real_noise_no_cham = np.mean([r['acc_real_noise'] for r in results_no_chameleon])
        avg_acc_fake_noise_no_cham = np.mean([r['acc_fake_noise'] for r in results_no_chameleon])
        avg_acc_real_ensemble_no_cham = np.mean([r['acc_real_ensemble'] for r in results_no_chameleon])
        avg_acc_fake_ensemble_no_cham = np.mean([r['acc_fake_ensemble'] for r in results_no_chameleon])
    else:
        avg_acc_fft_no_cham = avg_ap_fft_no_cham = 0
        avg_acc_attn_no_cham = avg_ap_attn_no_cham = 0
        avg_acc_noise_no_cham = avg_ap_noise_no_cham = 0
        avg_acc_ensemble_no_cham = avg_ap_ensemble_no_cham = 0
        avg_acc_real_fft_no_cham = avg_acc_fake_fft_no_cham = 0
        avg_acc_real_attn_no_cham = avg_acc_fake_attn_no_cham = 0
        avg_acc_real_noise_no_cham = avg_acc_fake_noise_no_cham = 0
        avg_acc_real_ensemble_no_cham = avg_acc_fake_ensemble_no_cham = 0

    print("\n" + "=" * 100)
    print(f"Average without Chameleon ({len(results_no_chameleon)} subsets)")
    print("=" * 100)
    print(f"{'Average (no Cham)':<20} {avg_acc_fft_no_cham:>7.2f}% {avg_ap_fft_no_cham:>7.2f}% "
          f"{avg_acc_attn_no_cham:>7.2f}% {avg_ap_attn_no_cham:>7.2f}% "
          f"{avg_acc_noise_no_cham:>7.2f}% {avg_ap_noise_no_cham:>7.2f}% "
          f"{avg_acc_ensemble_no_cham:>7.2f}% {avg_ap_ensemble_no_cham:>7.2f}%")
    print("\nReal/Fake Accuracy (no Chameleon):")
    print(f"  FFT: Real={avg_acc_real_fft_no_cham:.2f}%, Fake={avg_acc_fake_fft_no_cham:.2f}%")
    print(f"  Attn: Real={avg_acc_real_attn_no_cham:.2f}%, Fake={avg_acc_fake_attn_no_cham:.2f}%")
    print(f"  Noise: Real={avg_acc_real_noise_no_cham:.2f}%, Fake={avg_acc_fake_noise_no_cham:.2f}%")
    print(f"  Ensemble: Real={avg_acc_real_ensemble_no_cham:.2f}%, Fake={avg_acc_fake_ensemble_no_cham:.2f}%")

    if args.output_log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = args.output_log if args.output_log.endswith(".txt") else f"{args.output_log}_{timestamp}.txt"

        with open(log_path, "w") as f:
            f.write(f"Multi-Head Evaluation (GenImage response vectors) - {timestamp}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n\n")

            f.write("=" * 80 + "\n")
            f.write("Per-subset results\n")
            f.write("=" * 80 + "\n\n")

            for r in results:
                f.write(f"[{r['subset']}] (Total: {r['total']}, Real: {r['n_real']}, Fake: {r['n_fake']})\n")

                if r['is_chameleon']:
                    f.write(f"  FFT:    Acc={r['acc_fft']:.2f}%, BalancedAcc={r['acc_balanced_fft']:.2f}%, Real={r['acc_real_fft']:.2f}%, Fake={r['acc_fake_fft']:.2f}%, AP={r['ap_fft']:.2f}%\n")
                    f.write(f"  Attn:   Acc={r['acc_attn']:.2f}%, BalancedAcc={r['acc_balanced_attn']:.2f}%, Real={r['acc_real_attn']:.2f}%, Fake={r['acc_fake_attn']:.2f}%, AP={r['ap_attn']:.2f}%\n")
                    f.write(f"  Noise:  Acc={r['acc_noise']:.2f}%, BalancedAcc={r['acc_balanced_noise']:.2f}%, Real={r['acc_real_noise']:.2f}%, Fake={r['acc_fake_noise']:.2f}%, AP={r['ap_noise']:.2f}%\n")
                    f.write(f"  Ens:    Acc={r['acc_ensemble']:.2f}%, BalancedAcc={r['acc_balanced_ensemble']:.2f}%, Real={r['acc_real_ensemble']:.2f}%, Fake={r['acc_fake_ensemble']:.2f}%, AP={r['ap_ensemble']:.2f}%\n\n")
                else:
                    f.write(f"  FFT:    Acc={r['acc_fft']:.2f}%, Real={r['acc_real_fft']:.2f}%, Fake={r['acc_fake_fft']:.2f}%, AP={r['ap_fft']:.2f}%\n")
                    f.write(f"  Attn:   Acc={r['acc_attn']:.2f}%, Real={r['acc_real_attn']:.2f}%, Fake={r['acc_fake_attn']:.2f}%, AP={r['ap_attn']:.2f}%\n")
                    f.write(f"  Noise:  Acc={r['acc_noise']:.2f}%, Real={r['acc_real_noise']:.2f}%, Fake={r['acc_fake_noise']:.2f}%, AP={r['ap_noise']:.2f}%\n")
                    f.write(f"  Ens:    Acc={r['acc_ensemble']:.2f}%, Real={r['acc_real_ensemble']:.2f}%, Fake={r['acc_fake_ensemble']:.2f}%, AP={r['ap_ensemble']:.2f}%\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Average over all {len(results)} subsets\n")
            f.write("=" * 80 + "\n\n")

            avg_acc_fft_method2 = np.mean([r['acc_balanced_fft'] if r['is_chameleon'] else r['acc_fft'] for r in results])
            avg_acc_attn_method2 = np.mean([r['acc_balanced_attn'] if r['is_chameleon'] else r['acc_attn'] for r in results])
            avg_acc_noise_method2 = np.mean([r['acc_balanced_noise'] if r['is_chameleon'] else r['acc_noise'] for r in results])
            avg_acc_ensemble_method2 = np.mean([r['acc_balanced_ensemble'] if r['is_chameleon'] else r['acc_ensemble'] for r in results])

            f.write(f"FFT:    Acc={avg_acc_fft:.2f}%, BalancedAccAvg={avg_acc_fft_method2:.2f}%, Real={avg_acc_real_fft:.2f}%, Fake={avg_acc_fake_fft:.2f}%, AP={avg_ap_fft:.2f}%\n")
            f.write(f"Attn:   Acc={avg_acc_attn:.2f}%, BalancedAccAvg={avg_acc_attn_method2:.2f}%, Real={avg_acc_real_attn:.2f}%, Fake={avg_acc_fake_attn:.2f}%, AP={avg_ap_attn:.2f}%\n")
            f.write(f"Noise:  Acc={avg_acc_noise:.2f}%, BalancedAccAvg={avg_acc_noise_method2:.2f}%, Real={avg_acc_real_noise:.2f}%, Fake={avg_acc_fake_noise:.2f}%, AP={avg_ap_noise:.2f}%\n")
            f.write(f"Ens:    Acc={avg_acc_ensemble:.2f}%, BalancedAccAvg={avg_acc_ensemble_method2:.2f}%, Real={avg_acc_real_ensemble:.2f}%, Fake={avg_acc_fake_ensemble:.2f}%, AP={avg_ap_ensemble:.2f}%\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Average without Chameleon ({len(results_no_chameleon)} subsets)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"FFT:    Acc={avg_acc_fft_no_cham:.2f}%, Real={avg_acc_real_fft_no_cham:.2f}%, Fake={avg_acc_fake_fft_no_cham:.2f}%, AP={avg_ap_fft_no_cham:.2f}%\n")
            f.write(f"Attn:   Acc={avg_acc_attn_no_cham:.2f}%, Real={avg_acc_real_attn_no_cham:.2f}%, Fake={avg_acc_fake_attn_no_cham:.2f}%, AP={avg_ap_attn_no_cham:.2f}%\n")
            f.write(f"Noise:  Acc={avg_acc_noise_no_cham:.2f}%, Real={avg_acc_real_noise_no_cham:.2f}%, Fake={avg_acc_fake_noise_no_cham:.2f}%, AP={avg_ap_noise_no_cham:.2f}%\n")
            f.write(f"Ens:    Acc={avg_acc_ensemble_no_cham:.2f}%, Real={avg_acc_real_ensemble_no_cham:.2f}%, Fake={avg_acc_fake_ensemble_no_cham:.2f}%, AP={avg_ap_ensemble_no_cham:.2f}%\n")

        print(f"\n=> Results saved to: {log_path}")


def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Extract test-set perturbation response vectors and evaluate")

    parser.add_argument("--genimage_test_path", type=str, default="/home/don/dev/Datasets/GenImage/test",
                        help="Test data path. Either a root whose subdirectories are subsets, "
                             "or a single directory that directly contains 0_real and 1_fake.")
    parser.add_argument("--output_dir", type=str, default="./extracted_features_genimage_test",
                        help="Directory to save the extracted test features")
    parser.add_argument("--model_path", type=str,
                        default="/home/don/dev/code/Dinov3/dinov3-vit7b16-pretrain-lvd1689m",
                        help="DINOv3 backbone directory")

    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip extraction and evaluate existing features")
    parser.add_argument("--force_extract", action="store_true",
                        help="Re-extract features even if they already exist")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Only extract features, do not evaluate")

    parser.add_argument("--extract_batch_size", type=int, default=32)
    parser.add_argument("--img_resolution", type=int, default=256)
    parser.add_argument("--crop_resolution", type=int, default=224)
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument("--response_layer", type=int, default=-1,
                        help="Index of the hidden state used as feature, must match training")
    parser.add_argument("--pooling", type=str, default="mean_max", choices=["mean", "mean_max"])
    parser.add_argument("--keep_low", type=str2bool, default=False)
    parser.add_argument("--mask_highest", type=str2bool, default=True)

    parser.add_argument("--features_dir", type=str, default="",
                        help="Feature directory to evaluate, defaults to --output_dir")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/classifier_best.pth",
                        help="Classifier checkpoint saved by train_genimage.py")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=8192,
                        help="Feature dimension, 2*hidden_dim when pooling is mean_max")
    parser.add_argument("--output_log", type=str, default="",
                        help="Path to save the result log (empty means no log)")

    args = parser.parse_args()

    if not args.features_dir:
        args.features_dir = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Device: {device}")

    if args.skip_extract:
        print("=> --skip_extract given, skipping feature extraction")
    else:
        extract_features(args, device)

    if args.skip_eval:
        print("=> --skip_eval given, exiting")
        return

    if not os.path.exists(args.checkpoint):
        print(f"\n=> Checkpoint not found: {args.checkpoint}")
        print("=> Run train_genimage.py first, or pass a path via --checkpoint")
        return

    evaluate(args, device)


if __name__ == "__main__":
    main()
