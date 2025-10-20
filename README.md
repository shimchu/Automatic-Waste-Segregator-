# **Sahaay Club:** 

# **AUTOMATIC WASTE SEGREGATOR**

- Nishitha.D, EE23B188  
  


I contributed to the software module of the Automatic Waste Segregator, a Real-time Multi Waste Segregation Bin, that could detect upto 5 categories of waste and segregate them into their respective class. This was an effort to improve recycling efficiency and waste management.

The goal of the project was to develop a bin that could identify the waste kept on top of it with a camera, identify the class of the object and rotate the lid to the corresponding section and drop the item.

**Here are some visuals:**

  

[AWS results](https://drive.google.com/drive/folders/1-21b9bq_ADS3sBiYTX9dZydG3qhoDjXB?usp=drive_link)

For the Computer Vision aspect, we explored 3 different architectures, used transfer learning along with making our own custom dataset of 5000 images.   
The classes we chose were \- Plastic waste, Paper, Mixed (contains E waste and food waste), Metal and Glass. 

The initial challenge was the dataset. At the time the Kaggle dataset we chose that best represented our problem had only 200 images belonging each class, which we knew our model would easily overfit over considering the complex architectures we were using, plus there was not enough variation in the dataset such as different camera angles, lightning conditions that our camera was most likely capturing the data in. Hence we collected 1000 samples per class in various conditions, capturing through our phone and sourcing from images online. This dataset was then later used to finetune ResNet, MobileNet, InceptionNet using data augmentation available in Tensorflow Keras library using tf.keras.Sequential. This improved the model’s ability to generalize well to the real world, improving the model accuracy from 88 to 90% and improving the F1 score.

**Code snippet example:**  
    data\_augmentation \= tf.keras.Sequential(\[  
      layers.RandomFlip("horizontal\_and\_vertical"),  
      layers.RandomRotation(0.2),  
      layers.RandomZoom(0.2),  
    \])
**The final dataset:**

DATASET:[Dataset](https://github.com/RishiNandha/AWS_Dataset)

The models read upon and implemented \- ResNet \- 18, 34, 50, 152, InceptionNet, MobileNetV3.  
The second challenge was that there were no keras pretrained models for ResNet 34, hence we coded it from scratch based on the Research paper. 

The reason these models were chosen \- these models were the SOTA in image classification tasks on ImageNet data and CIFAR- 10, a subset of which is different kinds of utility objects and they also serve as the fundamentals required to understand more advanced architectures such as ViTs for image classification. 

We started off implementing a simple model on our dataset \-  
Layers \= \[Conv2D(72, kernel\_size \= (3,3), input\_shape \= (256,256,3)),  
          Conv2D(72, kernel\_size \= (3,3), activation \= 'relu'),  
          MaxPooling2D(pool\_size \= (2,2)),  
          Dropout(0.5),  
          Flatten(),  
          Dense(250, activation \= 'relu'),  
          Dense(5, activation \='softmax')\]

model \= keras. Sequential(Layers)

This model only reached 66% accuracy, this was because it was a small model with just one convolution block.   
[Simple AWS model](https://colab.research.google.com/drive/1S7VQNnuz_lclt4moL3tNKSMjP9KdE2w_?usp=sharing)

‘

**ResNet 34  \-**   
The ResNet 34 was our choice to finetune, 

Epoch 30: val\_accuracy did not improve from 0.62315  
57/57 \[==============================\] \- 34s 592ms/step \- loss: 1.0155 **\- accuracy: 0.6286** \- val\_loss: 1.0400 \- **val\_accuracy: 0.6232** \- lr: 1.0000e-05

Concluding it kept getting confused between glass and plastic, predicted almost every item as glass.There was also no data augmentation.  Thus the model clearly underfit.  
[62 Sahaay Resnet 34](https://colab.research.google.com/drive/1LdfSahoLb9srUUB8s1kBHIyR9RNrauU2?usp=sharing)

![][image1]

Hence we tried unfreezing more layers, augmenting, removing the fc layer and replacing it with a global avg pooling 2d and then some fully connected layers, reducing the parameters.

**InceptionNetV3**  
Initially training the model only gave 63% validation accuracy which was brought upto 86.7 % validation accuracy by increasing the trainable  layers and the dataset size through augmentation.But only after training a 100 epochs or so loading in checkpoints in multiple places.  
Epoch 30: val\_accuracy did not improve from 0.93528  
128/128 \[==============================\] \- 100s 785ms/step \- loss: 0.0443 \- accuracy: 0.9866 \- val\_loss: 0.3678 \- val\_accuracy: 0.9202 \- lr: 1.0000e-06  
[Copy of 93.528](https://colab.research.google.com/drive/1ScgWa2jBtiTIiRMV1MRmBwrBx1iduPsd?usp=sharing)

**MobileNetv2:** 

Train Accuracy: 0.9447572827339172  
Validation Accuracy: 0.9884687662124634

[Sahaay AWS MobileNet model](https://colab.research.google.com/drive/1aVJqvFQXqdbA4MNruOPGIPBTMQGsk_wQ?usp=sharing)

Validation F1 Score (macro): 0.9796797087654346  
\---------------  
Confusion Matrix:  
 \[\[674   1   2   0   3   0\]  
 \[ 13 528   3   1   0   0\]  
 \[ 13   1 519   3   2   0\]  
 \[ 12   0   3 878   2   0\]  
 \[ 10   1   4   1 479   0\]  
 \[  0   0   0   0   0 576\]\]  
\---------------  
Classification Report:  
               precision    recall  f1-score   support

           0       0.93      0.99      0.96       680  
           1       0.99      0.97      0.98       545  
           2       0.98      0.96      0.97       538  
           3       0.99      0.98      0.99       895  
           4       0.99      0.97      0.98       495  
           5       1.00      1.00      1.00       576

    accuracy                           0.98      3729  
   macro avg       0.98      0.98      0.98      3729  
weighted avg       0.98      0.98      0.98      3729

Thus the mobileNetV2 was the clear winner. Possible Reasons:

1. Dataset size vs model capacity (under-/over-fitting)  
- MobileNet is much smaller (fewer parameters). On small datasets it generalizes better because it’s less likely to overfit.  
- ResNet34 and Inception have higher capacity

Implementing these models served as a true learning experience into transfer learning, implementing models from scratch, understanding the intuition, algorithm and applications of different CNNs, advantages and disadvantages of each model.  
   
We chose to move forward with the MobileNetV2 model because:

1. Its lightweight architecture allows for efficient deployment on mobile and embedded devices with limited computational resources  
2. High accuracy achieved on the validation and test dataset  
3. The model’s small size enables faster inference times, making it suitable for real-time applications.  
   

 

[image1]: <data:image/png;base64,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>

   The dataset can be seen at https://github.com/RishiNandha/AWS_Dataset
   Collection of 5300+ images with split of 70:15:15 (train,val,test)


   
