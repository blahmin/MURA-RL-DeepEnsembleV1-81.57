# MURA-RLEnsemble-V1

## Overview  
MURA-RLEnsemble-V1 is an ensemble model inspired by the research presented in DeepSeek-R1, exploring reinforcement learning (RL) techniques to improve the classification of musculoskeletal fractures in X-ray images. The model is a reinforcement learning-enhanced ensemble of multiple base models, where a learned RL-based weighting agent combines predictions based on body part classification, providing a more tailored decision-making process.

While the results were slightly disappointing with a gap between training and validation accuracy and a Cohen's Kappa score of ~0.63, the architecture shows promising potential, particularly with the reinforcement learning approach. It was trained with extensive data augmentation and varied loss functions, along with reward-based feedback that helped the model adjust its weights during training.

## Download Model
The model weights can be downloaded from the [GitHub Releases] https://github.com/blahmin/MURA-RL-DeepEnsembleV2-81.57/releases/tag/DeepEnsembleV1  

## Model Performance  
- **Train Accuracy:** 91.27%  
- **Validation Accuracy:** 81.54%  
- **Cohen's Kappa Score:** 0.63  
- **Per-Body-Part Performance:**  
  - **XR_ELBOW**: 91.99% (Train), 87.53% (Val)  
  - **XR_FINGER**: 92.12% (Train), 75.70% (Val)  
  - **XR_HAND**: 88.48% (Train), 79.35% (Val)  
  - **XR_SHOULDER**: 89.93% (Train), 77.80% (Val)  
  - **XR_WRIST**: 92.33% (Train), 84.37% (Val)  
  - **XR_HUMERUS**: 95.06% (Train), 85.42% (Val)  
  - **XR_FOREARM**: 93.01% (Train), 81.67% (Val)  

## Model Architecture  
The **MURA-RLEnsemble-V1** architecture is based on three core components:  

### **1. BaseModel (Residual CNN with Attention)**
- Residual-based feature extraction with **EnhancedBlock** architecture.
- Incorporates a **spatial attention mechanism** to enhance key features during prediction.
- Outputs are combined in an ensemble strategy, with each base model predicting independently.

### **2. RL-Inspired Weighting Agent**
- A reinforcement learning agent learns to combine predictions from each model in the ensemble.
- The agent uses a **Softmax** function to determine the importance of each model’s prediction, improving ensemble accuracy.

### **3. Specialized Attention for Body Parts**
- Body-part specific attention modules adapt predictions for challenging body parts (e.g., **SHOULDER**, **ELBOW**).

## Key Features  
- **Ensemble Learning**: Combines predictions from three individual models using RL-based weighting.
- **Reinforcement Learning**: Uses a reinforcement learning agent to dynamically adjust the importance of each model’s predictions.
- **Data Augmentation**: Extensive transformations applied to training data to improve generalization.
- **Dropout Regularization**: Prevents overfitting and enhances model robustness.
  
## Training Pipeline  
- **Dataset:** MURA (Musculoskeletal Radiographs)  
- **Augmentations:** Random rotations, flips, color jitter, affine transforms, etc.  
- **Optimizer:** AdamW  
- **Learning Rate:** 1e-4  
- **Batch Size:** Configurable via DataLoader  

## Installation  
To install dependencies, run:
pip install torch torchvision albumentations pillow

python
Copy
Edit


## Future Improvements
Explore more complex RL-based weighting methods to enhance model prediction accuracy.
Implement transfer learning from larger pre-trained models to fine-tune the ensemble.
Investigate fine-tuning RL agents using reward functions specific to musculoskeletal X-rays.
Incorporate expert knowledge into the reward model to improve model performance for hard examples.
This model represents an important step in exploring RL-based techniques for medical image classification, and future iterations will aim to improve accuracy and efficiency in these tasks.
