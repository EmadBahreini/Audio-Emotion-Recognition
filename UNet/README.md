## UNet Model Overview

This document provides an overview of a UNet model that has been trained and evaluated with the following parameters and results.

### Training Parameters

- **Loss Function**: CrossEntropyLoss

  - The criterion used for training the model is `nn.CrossEntropyLoss()`. This loss function is commonly used for classification tasks and measures the performance of a classification model whose output is a probability value between 0 and 1.

- **Learning Rate**: 0.001

  - The learning rate determines the step size at each iteration while moving toward a minimum of the loss function. A learning rate of 0.001 is relatively low, which means the model updates its weights more gradually with each iteration.

- **Optimizer**: Adamax

  - The optimizer used is `optim.Adamax`, which is a variant of the Adam optimizer based on the infinity norm. It is designed to be more stable and perform better with sparse gradients. Adamax combines the advantages of both AdaGrad and RMSprop, making it suitable for a wide range of deep learning tasks. Additionally, L2 regularization is applied with a weight decay of 1e-4 to prevent overfitting.

- **Learning Rate Scheduler**: ReduceLROnPlateau

  - The learning rate scheduler used is `torch.optim.lr_scheduler.ReduceLROnPlateau`. It reduces the learning rate by a factor of 0.5 if the monitored metric does not improve for 10 consecutive epochs. This helps in fine-tuning the learning rate dynamically based on the model's performance, preventing it from getting stuck in local minima.

- **Epochs**: 150
  - An epoch refers to one complete pass through the entire training dataset.

### Final Training Results

- **Final Epoch**: [150/150]

- **Training Loss (TRLoss)**: 1.2782

- **Training Accuracy (TRAccuracy)**: 99.68%

- **Test Loss (TSLoss)**: 0.0492

- **Test Accuracy (TSAccuracy)**: 77.02%

### Learning Rate Scheduler Adjustments

- **Epoch 109**: Reduced learning rate to 0.0005
- **Epoch 122**: Reduced learning rate to 0.00025
- **Epoch 133**: Reduced learning rate to 0.000125

### Loss and Accuracy Curves

![alt text](Loss_Curves.png)

![alt text](Acc_Curves.png)

### Validation Report:

              precision    recall  f1-score   support

     neutral       0.44      0.51      0.47        37
        calm       0.79      0.87      0.83        61
       happy       0.85      0.71      0.78        87
         sad       0.68      0.74      0.71        77
       angry       0.88      0.81      0.85        91
     fearful       0.71      0.80      0.75        66
     disgust       0.76      0.75      0.76        77
    surprised      0.78      0.72      0.75        80

    accuracy                           0.75       576
    macro avg      0.74      0.74      0.74       576
    weighted avg   0.76      0.75      0.76       576

![alt text](Conf_Mtrx.png)
