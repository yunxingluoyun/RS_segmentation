import os
import cv2
import numpy as np
import torch
import imageio
import segmentation_models_pytorch as smp
from train_model import Dataset,get_validation_augmentation,get_preprocessing
from sklearn.metrics import classification_report



# 设置参数及路径
DEVICE = 'cuda'
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

DATA_DIR = "patch"
MODEL_DIR = './best_model.pth'
SAVE_DIR = "pred"
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

# 加载最优模型 
best_model = torch.load(MODEL_DIR)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

#x_test_dir = os.path.join(os.path.join(DATA_DIR, 'val'),'img')
#y_test_dir = os.path.join(os.path.join(DATA_DIR, 'val'),'gt')
x_test_dir = os.path.join(DATA_DIR, 'img')
y_test_dir = os.path.join(DATA_DIR, 'gt')

# test dataset without transformations for image visualization
# test_dataset = Dataset(
#     x_test_dir, y_test_dir, 
#     classes=CLASSES,
# )
test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


y_true = []
y_pred = []
for i,gt_name in enumerate(sorted(os.listdir(y_test_dir))):
    
    image, gt_mask = test_dataset[i]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    
    pr_mask = best_model.predict(x_tensor).argmax(dim=1)
    pr_mask = (pr_mask.squeeze().cpu().numpy())
    # gt_mask = gt_mask.squeeze().cpu().numpy()
    #print("pr_mask",pr_mask.shape)
    #print(gt_mask.argmax(axis=0).shape)
    cv2.imwrite(os.path.join(SAVE_DIR,gt_name),pr_mask)
    #print(gt_mask.argmax(axis=0))
    # print(gt_mask[:1,:1,:])
    # print(pr_mask[:5,:5])
    y_true.extend(gt_mask.argmax(axis=0).flatten().tolist())
    y_pred.extend(pr_mask.flatten().tolist())
# print(len(y_true),len(y_pred))
print(classification_report(y_true, y_pred, target_names=CLASSES))

        
