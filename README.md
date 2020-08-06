This little research is made under my internship at IITP (http://iitp.ru/en/science) and is based on two articles:

1) https://arxiv.org/abs/1805.01934

The first problem is about adjusting the existing NN pipeline for an arbitrary photo camera with the aim to transform short exposed raw photos into high-quality RGB images with the help of U-net with output quality better than hardcoded ISP algorithm in camera can provide. PSNR AND SSIM metrics are used to compare output of the NN with the ground truth (long exposed photo). The issue of applicability of a pretrained model for a new camera (another image sensor) as well as the reducing of the train set are investigated too. 

2) https://arxiv.org/abs/1801.06724

Under the second theme the goal is to mimic a given camera's ISP algoritm using NN with residual blocks.
 
All current experiments, datasets used and Tensrflow code are stored in this Google Drive: https://drive.google.com/drive/folders/1LC2DO_McdGIn8e3MNfx_SixUIco0M240?usp=sharing 

