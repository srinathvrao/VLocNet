# VLocNet
A pytorch implementation of VLocNet, as described in [Deep Auxiliary Learning for Visual Localization and Odometry.](https://arxiv.org/pdf/1803.03642.pdf)

I worked on the [Microsoft 7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) for this project. Just like the paper, I have omitted the depth map while building the pipeline, and trained a siamese-type Convolutional Neural Network to predict the Global position of the camera, and the relative transformation (change in orientation) between two consecutive frames.

## Architecture
Here's the architecture of the model I built and trained. I used the weights from the ResNet-50 model trained on the ImageNet dataset, and trained this entire pipeline on Google Colab for about 48 hours. I got a Median Localization Error of 0.40 metres (400 cm), while the paper did a lot better (0.048m / 48cm).
![vlocnnarch](architecture_vlocnet.png)
