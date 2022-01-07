This research project was made under my internship at Institute for Information Transmission Problems of the Russian Academy of Sciences (Kharkevich Institute) (http://iitp.ru/en/science) and is based on the article https://arxiv.org/abs/1805.01934 

The problem solved is adjusting the existing NN pipeline for an arbitrary photo camera with the aim to transform short exposed raw photos into high-quality RGB images with the help of U-net. PSNR and SSIM metrics are used to compare output of the NN with the ground truth (long exposed photo). The issue of applicability of a pretrained model for a new camera (that may have another image sensor) as well as the reducing of the train set size are investigated too. 

Code snippets for training and testing are presented above in Tensorflow 2.0.

