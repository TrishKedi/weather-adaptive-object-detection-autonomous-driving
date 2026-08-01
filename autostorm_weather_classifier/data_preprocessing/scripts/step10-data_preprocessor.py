import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np



class DataPreProcessor(Dataset):
    def __init__(self, image_dir, annotation_file, transforms=None, category_ids=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.category_ids = category_ids

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Filter out invalid boxes and categories not in remap (if specified)
        if self.category_ids is not None:
            anns = [ann for ann in anns if ann['category_id'] in self.category_ids and ann['iscrowd'] == 0]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])  # Convert to Pascal VOC format
                labels.append(ann['category_id'])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # Prepare target dictionary
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([img_id])

        return image, target

    def __len__(self):
        return len(self.ids)


# === Albumentations Transform Pipeline ===
def get_train_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=800),
        A.PadIfNeeded(min_height=800, min_width=800, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# === Custom collate_fn for DataLoader ===
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


image_dir = "../dataset/train/images"
annotation_file = "../dataset/train/coco_annotations.json"

dataset = DataPreProcessor(
    image_dir=image_dir,
    annotation_file=annotation_file,
    transforms=get_train_transforms(),
    category_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9]
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# Test a batch
for imgs, targets in dataloader:
    print("Batch of images:", len(imgs))
    print("First target sample:", targets[0])
    break
