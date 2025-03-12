# Deepfake Detection using EfficientNet-B0

## Project Overview
This project focuses on detecting deepfake images using a convolutional neural network (CNN) model. We implemented **EfficientNet-B0** using PyTorch for classification between real and fake images.

## Dataset
- **Fake Frames:** `/content/drive/MyDrive/deepfake_dataset/fake_frames/fake_frames`
- **Real Frames:** `/content/drive/MyDrive/deepfake_dataset/real_frames/real_frames`

The dataset was structured using symbolic links to create a structured dataset for training and validation.

## Model Architecture
- **Base Model:** EfficientNet-B0 (Pretrained on ImageNet)
- **Modifications:** Last layer modified for binary classification (Real/Fake)
- **Loss Function:** Binary Cross-Entropy Loss
- **Optimizer:** Adam (learning rate = 1e-4)
- **Batch Size:** 32
- **Epochs:** 10
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Evaluation Results

✅ **Test Accuracy:** 90.36%

📌 **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fake  | 0.85      | 0.96   | 0.90     | 6019    |
| Real  | 0.97      | 0.85   | 0.91     | 7150    |

📌 **Confusion Matrix:**
```
[[5802  217]
 [1053 6097]]
```

📌 **ROC-AUC Score:** 0.9748

## Output Files
- **results.csv** → Contains model performance metrics per epoch.
- **confusion_matrix.png** → Raw confusion matrix visualization.
- **normalized_confusion_matrix.png** → Normalized confusion matrix visualization.
- **f1_curve.png** → F1-score vs. threshold.
- **p_curve.png** → Precision vs. threshold.
- **pr_curve.png** → Precision-Recall curve.
- **r_curve.png** → Recall vs. threshold.
- **results.png** → Graphical summary of model performance.

## Model Checkpoint
The best model was saved at:
```
/content/drive/MyDrive/deepfake_efficientnet_best.pth
```

## How to Run
1. Ensure all dependencies (`torch`, `torchvision`, `tqdm`, `matplotlib`, `sklearn`) are installed.
2. Load the trained model using:
    ```python
    import torch
    from torchvision import models

    model = models.efficientnet_b0(pretrained=False)
    model.load_state_dict(torch.load("/content/drive/MyDrive/deepfake_efficientnet_best.pth"))
    ```
3. Run inference on test images.

## Future Work
- Improve model robustness using augmentation techniques.
- Experiment with other EfficientNet variants (B1-B7) for better performance.
- Implement Grad-CAM for model explainability.

🚀 **Project Completed!**
