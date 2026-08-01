import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block (MBConv)
    This is the basic building block of EfficientNet
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()  # Swish activation
        )
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride=stride, padding=padding, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Squeeze and excitation
        se_channels = max(1, int(expanded_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Output phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_residual = in_channels == out_channels and stride == 1
        
    def forward(self, x):
        residual = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + residual
            
        return x

class Autostorm_model(nn.Module):
    """
    EfficientNet-B3 implementation
    """
    def __init__(self, num_classes=10):
        super(Autostorm_model, self).__init__()
        
        # B3 specific parameters with increased capacity
        width_mult = 1.3  # Increased from 1.2
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32 * width_mult), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * width_mult)),
            nn.SiLU()
        )
        
        # MBConv blocks for B3 with increased depth
        self.blocks = nn.Sequential(
            MBConvBlock(int(32 * width_mult), int(16 * width_mult), 3, 1, 1),
            MBConvBlock(int(16 * width_mult), int(24 * width_mult), 3, 2, 6),
            MBConvBlock(int(24 * width_mult), int(24 * width_mult), 3, 1, 6),
            MBConvBlock(int(24 * width_mult), int(48 * width_mult), 5, 2, 6),
            MBConvBlock(int(48 * width_mult), int(48 * width_mult), 5, 1, 6),
            MBConvBlock(int(48 * width_mult), int(48 * width_mult), 5, 1, 6),  # Added block
            MBConvBlock(int(48 * width_mult), int(88 * width_mult), 3, 2, 6),
            MBConvBlock(int(88 * width_mult), int(88 * width_mult), 3, 1, 6),
            MBConvBlock(int(88 * width_mult), int(88 * width_mult), 3, 1, 6),
            MBConvBlock(int(88 * width_mult), int(120 * width_mult), 5, 1, 6),
            MBConvBlock(int(120 * width_mult), int(120 * width_mult), 5, 1, 6),
            MBConvBlock(int(120 * width_mult), int(208 * width_mult), 5, 2, 6),
            MBConvBlock(int(208 * width_mult), int(208 * width_mult), 5, 1, 6),
            MBConvBlock(int(208 * width_mult), int(208 * width_mult), 5, 1, 6),  # Added block
            MBConvBlock(int(208 * width_mult), int(352 * width_mult), 3, 1, 6)
        )
        
        # Final layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(352 * width_mult), int(1408 * width_mult), 1, bias=False),
            nn.BatchNorm2d(int(1408 * width_mult)),
            nn.SiLU()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)  # Changed back to 0.4
        self.fc = nn.Linear(int(1408 * width_mult), num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5):
    """
    Training function with progress tracking and early stopping
    """
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    patience_counter = 0
    scaler = GradScaler('cuda')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Use scaler for backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{train_loss/train_total:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                # Use AMP for validation
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_pbar.set_postfix({
                    'loss': f'{val_loss/val_total:.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'autostorm_weights.pth')
            torch.save(model, 'autostorm.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation metrics
    """
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('efficientnet_b3_metrics.png')
    plt.close()

def test_model(model, test_loader, criterion, device):
    """
    Test the model on the test dataset
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    class_correct = [0] * len(test_loader.dataset.classes)
    class_total = [0] * len(test_loader.dataset.classes)
    
    print("\n=== Testing Model ===")
    test_pbar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # Calculate per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
            
            test_pbar.set_postfix({
                'loss': f'{test_loss/test_total:.4f}',
                'acc': f'{100.*test_correct/test_total:.2f}%'
            })
    
    # Print overall accuracy
    print(f'\nOverall Test Accuracy: {100.*test_correct/test_total:.2f}%')
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    
    # Print per-class accuracy
    print('\nPer-class Accuracy:')
    for i in range(len(test_loader.dataset.classes)):
        if class_total[i] > 0:
            accuracy = 100. * class_correct[i] / class_total[i]
            print(f'{test_loader.dataset.classes[i]}: {accuracy:.2f}%')
    
    return 100. * test_correct / test_total

def main():
    # Set device and display detailed information
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Set memory optimization
        torch.backends.cudnn.benchmark = True
        print("\n=== NVIDIA GPU Information ===")
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
        print(f'CUDA Version: {torch.version.cuda}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')  # For Apple Silicon
        print("\n=== Apple Silicon Information ===")
        print('Using Apple MPS (Metal Performance Shaders)')
    elif hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
        device = torch.device('rocm')  # For AMD GPUs
        print("\n=== AMD GPU Information ===")
        print('Using AMD ROCm')
    else:
        device = torch.device('cpu')
        print("\n=== CPU Information ===")
        print('Running on CPU')
        print(f'CPU Cores: {torch.get_num_threads()}')
    print("========================\n")
    
    # Data transforms with torchvision - simplified and smaller size
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder('dataset/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('dataset/val', transform=val_transform)
    test_dataset = datasets.ImageFolder('dataset/test', transform=val_transform)
    
    # Initialize model
    model = Autostorm_model(num_classes=len(train_dataset.classes))
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    
    # Enhanced optimizer and scheduler with stronger regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.075)  # Changed from 0.05 to 0.075
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Cosine learning rate schedule with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=40,  # Changed from 30 to 40
        steps_per_epoch=len(train_loader),
        pct_start=0.4,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    # Train the model
    num_epochs = 40  # Changed from 30 to 40
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience=5
    )
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model and test
    model.load_state_dict(torch.load('best_efficientnet_b3_weather.pth'))
    test_accuracy = test_model(model, test_loader, criterion, device)
    print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main() 