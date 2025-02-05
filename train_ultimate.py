import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from datetime import datetime, timedelta
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ultimate_ensemble import UltimateEnsembleModel

class MURADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with subfolders for each body part (e.g., XR_ELBOW, XR_FINGER, etc.)
            transform: torchvision transforms to apply.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.body_parts = []

        parts = [d for d in os.listdir(root_dir) if d.startswith('XR_')]
        print("\nDataset composition:")
        for body_part in parts:
            part_dir = os.path.join(root_dir, body_part)
            part_count = 0
            for root, _, files in os.walk(part_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        self.image_paths.append(full_path)
                        label = 1 if 'positive' in root.lower() else 0
                        self.labels.append(label)
                        self.body_parts.append(body_part)
                        part_count += 1
            print(f"{body_part}: {part_count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        body_part = self.body_parts[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, body_part, img_path  


def load_misclassified(file_path):
    """
    Reads a text file (one image path per line) of misclassified samples.
    Returns a set of file paths.
    """
    misclassified_set = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    misclassified_set.add(path)
        print(f"Loaded {len(misclassified_set)} misclassified sample paths from {file_path}")
    else:
        print(f"Misclassified file not found at {file_path}")
    return misclassified_set

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch_num, rl_loss_weight=0.1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0


    part_correct = {}
    part_total = {}

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch_num}")
    for inputs, labels, body_parts, img_paths in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        final_pred, _ = model(inputs, body_parts[0])
        supervised_loss = criterion(final_pred, labels)

        # Compute predictions and assign a simple reward: +1 for correct, -1 for incorrect.
        _, predicted = torch.max(final_pred, 1)
        reward = (predicted == labels).float() * 2 - 1.0  # +1 for correct, -1 for incorrect

        try:
            rl_loss = model.compute_total_rl_loss(reward)
        except ValueError:
            rl_loss = 0.0 * supervised_loss  # fallback if RL log probabilities are not set

        total_loss = supervised_loss + rl_loss_weight * rl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += total_loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i, part in enumerate(body_parts):
            if part not in part_correct:
                part_correct[part] = 0
                part_total[part] = 0
            part_total[part] += 1
            if predicted[i] == labels[i]:
                part_correct[part] += 1

        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "Total Loss": f"{total_loss.item():.4f}",
            "Acc": f"{100 * correct / total:.2f}%",
            "LR": f"{current_lr:.6f}"
        })

        del inputs, labels, final_pred, total_loss, supervised_loss, rl_loss
        torch.cuda.empty_cache()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    part_accuracy = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return epoch_loss, epoch_acc, part_accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    part_correct = {}
    part_total = {}

    with torch.no_grad():
        for inputs, labels, body_parts, _ in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            final_pred, _ = model(inputs, body_parts[0])
            loss = criterion(final_pred, labels)
            running_loss += loss.item()
            _, predicted = torch.max(final_pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i, part in enumerate(body_parts):
                if part not in part_correct:
                    part_correct[part] = 0
                    part_total[part] = 0
                part_total[part] += 1
                if predicted[i] == labels[i]:
                    part_correct[part] += 1

            del inputs, labels, final_pred, loss
            torch.cuda.empty_cache()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    part_accuracy = {part: 100 * part_correct[part] / part_total[part] for part in part_total}
    return epoch_loss, epoch_acc, part_accuracy

def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    num_classes = 2
    rl_loss_weight = 0.1  # Weight for the RL loss component

    # Data directories
    train_dir = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\train"
    val_dir   = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\data\valid"
    misclassified_path = r"C:\Users\blahm\PycharmProjects\Work\CNNXRAY\misclassified.txt"

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomResizedCrop(160, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomEqualize(p=1.0)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    misclassified_set = load_misclassified(misclassified_path)
    train_dataset = MURADataset(train_dir, transform=train_transform)
    val_dataset   = MURADataset(val_dir, transform=val_transform)

    sample_weights = []
    for path in train_dataset.image_paths:
        if path in misclassified_set:
            sample_weights.append(2.0)  # Higher weight for misclassified images
        else:
            sample_weights.append(1.0)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = UltimateEnsembleModel(num_classes=num_classes, beta=0.5).to(device)

    # Load pretrained weights from the previous enhanced ensemble (82.4% val acc)
    pretrained_path = os.path.join(os.path.dirname(train_dir), "ultimate_ensemble_RAAAH.pth")
    if os.path.exists(pretrained_path):
        print("Loading pretrained weights from previous enhanced ensemble...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.rl_ensemble.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.rl_ensemble.load_state_dict(checkpoint, strict=False)
        print("Pretrained weights loaded successfully.")
    else:
        print("No pretrained weights found, training from scratch.")

    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    best_val_acc = 0.0

    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc, train_part_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch + 1, rl_loss_weight=rl_loss_weight)
        val_loss, val_acc, val_part_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch [{epoch+1}/{epochs}] - Time: {epoch_time/60:.2f} minutes")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print("Per-body-part Train Accuracy:")
        for part, acc in train_part_acc.items():
            print(f"  {part}: {acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("Per-body-part Validation Accuracy:")
        for part, acc in val_part_acc.items():
            print(f"  {part}: {acc:.2f}%")

        # Save best model checkpoint based on validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc
            }
            save_path = os.path.join(os.path.dirname(train_dir), "ultimate_ensemble_best.pth")
            torch.save(checkpoint, save_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time * (epochs / (epoch + 1))
        estimated_remaining = estimated_total - elapsed_time
        print(f"\nTime elapsed: {elapsed_time/3600:.2f} hours")
        print(f"Estimated time remaining: {estimated_remaining/3600:.2f} hours")
        print(f"Estimated completion: {(datetime.now() + timedelta(seconds=estimated_remaining)).strftime('%Y-%m-%d %H:%M:%S')}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
