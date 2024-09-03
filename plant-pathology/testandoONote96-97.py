import pandas as pd
import torch
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

class ImageDataset(Dataset):
    def __init__(self, df, img_dir='./', transform=None, is_test=False):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = self.img_dir + img_id + '.jpg'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.is_test:
            return image
        else:
            label = np.argmax(self.df.iloc[idx, 1:5])
            return image, label

def main():
    seed = 50
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = ''

    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    submission = pd.read_csv(data_path + 'sample_submission.csv')

    train, valid = train_test_split(train, 
                                    test_size=0.1,
                                    stratify=train[['healthy', 'multiple_diseases', 'rust', 'scab']],
                                    random_state=50)

    transform_train = A.Compose([
        A.Resize(450, 650),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.VerticalFlip(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.3),
        A.OneOf([A.Emboss(p=1), A.Sharpen(p=1), A.Blur(p=1)], p=0.3),
        A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.3),  # Alternativa rápida
        A.Normalize(),
        ToTensorV2()
    ])

    transform_test = A.Compose([
        A.Resize(450, 650),
        A.Normalize(),
        ToTensorV2()
    ])

    img_dir = 'images/train/'

    dataset_train = ImageDataset(train, img_dir=img_dir, transform=transform_train)
    dataset_valid = ImageDataset(valid, img_dir=img_dir, transform=transform_test)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(0)

    batch_size = 4

    loader_train = DataLoader(dataset_train, batch_size=batch_size, 
                              shuffle=True, worker_init_fn=seed_worker,
                              generator=g, num_workers=2)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, 
                              shuffle=False, worker_init_fn=seed_worker,
                              generator=g, num_workers=2)

    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=4)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006, weight_decay=0.0001)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(loader_train)*3,
                                                num_training_steps=len(loader_train)*39)

    for epoch in range(39):
        model.train()
        epoch_train_loss = 0
        count = 0

        for images, labels in tqdm(loader_train):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f'Epoch [{epoch+1}/39] - Training Data Loss: {epoch_train_loss/len(loader_train):.4f}')
        
        model.eval()
        epoch_valid_loss = 0
        preds_list = []
        true_onehot_list = []

        with torch.no_grad():
            for images, labels in loader_valid:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_valid_loss += loss.item()
                preds = torch.softmax(outputs.cpu(), dim=1).numpy()
                true_onehot = torch.eye(4).to(labels.device)[labels].cpu().numpy()
                preds_list.extend(preds)
                true_onehot_list.extend(true_onehot)
        
        print(f'Epoch [{epoch+1}/39] - Validation Data Loss: {epoch_valid_loss/len(loader_valid):.4f} / Validation Data ROC AUC: {roc_auc_score(true_onehot_list, preds_list):.4f}') 

    dataset_test = ImageDataset(test, img_dir=img_dir, transform=transform_test, is_test=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=2)

    dataset_TTA = ImageDataset(test, img_dir=img_dir, transform=transform_train, is_test=True)
    loader_TTA = DataLoader(dataset_TTA, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=2)

    model.eval()

    preds_test = np.zeros((len(test), 4))

    with torch.no_grad():
        for i, images in enumerate(loader_test):
            images = images.to(device)
            outputs = model(images)
            preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
            preds_test[i*batch_size:(i+1)*batch_size] += preds_part

    submission_test = submission.copy()
    submission_test[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_test

    num_TTA = 7

    preds_tta = np.zeros((len(test), 4))

    for i in range(num_TTA):
        with torch.no_grad():
            for i, images in enumerate(loader_TTA):
                images = images.to(device)
                outputs = model(images)
                preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
                preds_tta[i*batch_size:(i+1)*batch_size] += preds_part

    preds_tta /= num_TTA

    submission_tta = submission.copy()
    submission_tta[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_tta

    submission_test.to_csv('submission_test.csv', index=False)
    submission_tta.to_csv('submission_tta.csv', index=False)

if __name__ == '__main__':
    main()
