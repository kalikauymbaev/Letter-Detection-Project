import argparse
import os
import random
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# -----------------------------
# IDX (MNIST/EMNIST) gzip readers
# -----------------------------
def _read_idx_images_gz(path: str) -> np.ndarray:
    """Reads idx3-ubyte(.gz) -> (N, H, W) uint8."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic} in {path}")
        data = f.read(n * rows * cols)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)
    return arr

def _read_idx_labels_gz(path: str) -> np.ndarray:
    """Reads idx1-ubyte(.gz) -> (N,) uint8."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic} in {path}")
        data = f.read(n)
        arr = np.frombuffer(data, dtype=np.uint8)
    return arr


def _fix_emnist_orientation(images_nhw: np.ndarray) -> np.ndarray:
    """
    EMNIST images are typically rotated/flipped compared to MNIST.
    Standard fix used in many implementations:
      rotate 90 degrees then flip left-right
    """
    # images: (N, H, W)
    images = np.rot90(images_nhw, k=1, axes=(1, 2))
    images = np.flip(images, axis=2)  # flip horizontally (left-right)
    return images


def load_emnist_letters(emnist_dir: str):
    """
    Loads EMNIST Letters split from 4 gzip IDX files.
    Returns x_train, y_train, x_test, y_test
    with:
      x_*: (N, 28, 28) uint8
      y_*: (N,) int64 in [0..25]
    """
    train_images_path = os.path.join(emnist_dir, "emnist-letters-train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(emnist_dir, "emnist-letters-train-labels-idx1-ubyte.gz")
    test_images_path  = os.path.join(emnist_dir, "emnist-letters-test-images-idx3-ubyte.gz")
    test_labels_path  = os.path.join(emnist_dir, "emnist-letters-test-labels-idx1-ubyte.gz")

    for p in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    x_train = _read_idx_images_gz(train_images_path)
    y_train = _read_idx_labels_gz(train_labels_path)
    x_test  = _read_idx_images_gz(test_images_path)
    y_test  = _read_idx_labels_gz(test_labels_path)

    # Fix EMNIST rotation/flip
    x_train = _fix_emnist_orientation(x_train)
    x_test  = _fix_emnist_orientation(x_test)

    # EMNIST Letters labels are 1..26 -> convert to 0..25
    y_train = y_train.astype(np.int64) - 1
    y_test  = y_test.astype(np.int64) - 1

    if y_train.min() < 0 or y_train.max() > 25:
        raise ValueError(f"Label range looks wrong after remap: [{y_train.min()}, {y_train.max()}]")

    return x_train, y_train, x_test, y_test


# -----------------------------
# Dataset (same idea as MNIST npz)
# -----------------------------
class NumpyImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images.astype(np.float32) / 255.0  # (N,H,W)
        if self.images.ndim == 3:
            self.images = self.images[:, None, :, :]  # (N,1,H,W)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])   # CHW float32 in [0,1]
        label = int(self.labels[idx])
        if self.transform is not None and _HAS_TORCHVISION:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def create_dataloaders(batch_size: int, val_split: float, use_augmentation: bool, emnist_dir: str):
    x_train, y_train, x_test, y_test = load_emnist_letters(emnist_dir)

    # Keep size consistent with model export. You can keep 28x28 or resize to 32x32.
    # If you resize in training, also export ONNX with that size.
    target_hw = (28, 28)

    if _HAS_TORCHVISION and use_augmentation:
        aug = transforms.Compose([
            transforms.Resize(target_hw),
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.Normalize((0.0,), (1.0,)),
        ])
    else:
        aug = transforms.Compose([
            transforms.Resize(target_hw),
            transforms.ConvertImageDtype(torch.float32),
        ]) if _HAS_TORCHVISION else None

    train_dataset_full = NumpyImageDataset(x_train, y_train, transform=aug)
    test_dataset = NumpyImageDataset(x_test, y_test, transform=None)

    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


# -----------------------------
# Model (same, but num_classes=26)
# -----------------------------
class QuantFriendlyEMNIST(nn.Module):
    def __init__(self, num_classes: int = 26, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return 100.0 * correct / max(1, total)


def get_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if not force_cpu and mps_available:
        print("Using Apple MPS (Metal)")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def train_model(args):
    device = get_device(force_cpu=args.cpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        use_augmentation=args.augmentation,
        emnist_dir=args.emnist_dir,
    )

    model = QuantFriendlyEMNIST(num_classes=26).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    os.makedirs("artifacts", exist_ok=True)
    best_pt_path = os.path.join("artifacts", "best_emnist_letters_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if args.log_interval and batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] loss={loss.item():.4f}")

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} validation accuracy: {val_acc:.2f}%")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_pt_path)
            print(f"New best model saved to {best_pt_path} (val_acc={best_val_acc:.2f}%)")

    model.load_state_dict(torch.load(best_pt_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy (best ckpt): {test_acc:.2f}%")
    return best_pt_path, test_acc


# -----------------------------
# Export ONNX (FP32) - EMNIST letters
# -----------------------------
def export_onnx(pt_path: str, onnx_path: str = os.path.join("artifacts", "emnist_letters_cnn.onnx")):
    device = torch.device("cpu")
    model = QuantFriendlyEMNIST(num_classes=26).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"ONNX FP32 model saved to {onnx_path}")
    return onnx_path


# -----------------------------
# ORT calibration reader (uses EMNIST train images)
# -----------------------------
class EMNISTLettersDataReader:
    def __init__(self, input_name: str, emnist_dir: str, calibration_batch: int = 32, num_samples: int = 1024):
        x_train, _, _, _ = load_emnist_letters(emnist_dir)
        if x_train.ndim == 3:
            x_train = x_train[:, None, :, :]
        x_train = x_train.astype(np.float32) / 255.0
        self.images = x_train[:num_samples]
        self.input_name = input_name
        self.batch = calibration_batch
        self._num_batches = int(np.ceil(len(self.images) / self.batch))
        self._iter = 0

    def get_next(self):
        if self._iter >= self._num_batches:
            return None
        start = self._iter * self.batch
        end = min((self._iter + 1) * self.batch, len(self.images))
        self._iter += 1
        batch = self.images[start:end]
        return {self.input_name: batch}

    def rewind(self):
        self._iter = 0


def quantize_onnx(fp32_model_path: str,
                 int8_model_path: str = os.path.join("artifacts", "emnist_letters_cnn.int8.onnx"),
                 emnist_dir: str = "./emnist",
                 calibration_samples: int = 1024,
                 calibration_batch: int = 32):
    from onnxruntime.quantization import QuantType, quantize_static, QuantFormat
    import onnx

    onnx_model = onnx.load(fp32_model_path)
    input_name = onnx_model.graph.input[0].name

    dr = EMNISTLettersDataReader(
        input_name=input_name,
        emnist_dir=emnist_dir,
        calibration_batch=calibration_batch,
        num_samples=calibration_samples,
    )

    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"ONNX INT8 model saved to {int8_model_path}")
    return int8_model_path


def evaluate_onnx(onnx_path: str, emnist_dir: str, batch_size: int = 64):
    import onnxruntime as ort
    _, _, x_test, y_test = load_emnist_letters(emnist_dir)
    if x_test.ndim == 3:
        x_test = x_test[:, None, :, :]
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int64)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    correct = 0
    total = 0
    for i in range(0, len(x_test), batch_size):
        batch_x = x_test[i:i+batch_size]
        batch_y = y_test[i:i+batch_size]
        logits = sess.run([output_name], {input_name: batch_x})[0]
        pred = np.argmax(logits, axis=1)
        correct += int((pred == batch_y).sum())
        total += len(batch_y)

    acc = 100.0 * correct / max(1, total)
    print(f"ONNX eval accuracy: {acc:.2f}% -> {os.path.basename(onnx_path)}")
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train EMNIST Letters, export ONNX, and quantize with ONNX Runtime")
    parser.add_argument("--emnist-dir", type=str, default="./emnist", help="Folder containing emnist-letters-*.gz files")

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-quant", action="store_true")

    parser.add_argument("--calib-samples", type=int, default=1024)
    parser.add_argument("--calib-batch", type=int, default=32)

    parser.add_argument("--onnx-input", type=str, default=None, help="Use existing FP32 ONNX for eval/quantize")
    parser.add_argument("--pt-input", type=str, default=None, help="Use existing .pt for ONNX export when skipping training")
    return parser.parse_args()


def main():
    args = parse_args()

    default_best_pt = os.path.join("artifacts", "best_emnist_letters_cnn.pt")
    default_fp32_onnx = os.path.join("artifacts", "emnist_letters_cnn.onnx")
    onnx_fp32_path = None

    # 1) Training
    if not args.skip_train:
        best_pt_path, _ = train_model(args)
    else:
        best_pt_path = args.pt_input if args.pt_input else default_best_pt

    # 2) FP32 ONNX source
    if args.onnx_input:
        if not os.path.exists(args.onnx_input):
            print(f"Error: --onnx-input not found: {args.onnx_input}")
            return
        onnx_fp32_path = args.onnx_input
        try:
            evaluate_onnx(onnx_fp32_path, emnist_dir=args.emnist_dir)
        except Exception as e:
            print(f"Skip FP32 ONNX eval: {e}")
    else:
        if not args.skip_export:
            if not os.path.exists(best_pt_path):
                print("Error: checkpoint not found. Provide --pt-input or remove --skip-train to train.")
                print(f"Tried: {best_pt_path}")
                return
            onnx_fp32_path = export_onnx(best_pt_path)
            try:
                evaluate_onnx(onnx_fp32_path, emnist_dir=args.emnist_dir)
            except Exception as e:
                print(f"Skip FP32 ONNX eval: {e}")
        else:
            if os.path.exists(default_fp32_onnx):
                onnx_fp32_path = default_fp32_onnx

    # 3) Quantization
    if not args.skip_quant:
        if onnx_fp32_path is None:
            if os.path.exists(default_fp32_onnx):
                onnx_fp32_path = default_fp32_onnx
            else:
                print("Error: No FP32 ONNX available for quantization. Provide --onnx-input or run export.")
                return

        onnx_int8_path = quantize_onnx(
            onnx_fp32_path,
            emnist_dir=args.emnist_dir,
            calibration_samples=args.calib_samples,
            calibration_batch=args.calib_batch,
        )
        try:
            evaluate_onnx(onnx_int8_path, emnist_dir=args.emnist_dir)
        except Exception as e:
            print(f"Skip INT8 ONNX eval: {e}")


if __name__ == "__main__":
    main()
