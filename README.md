This research is made under my internship at IITP (http://iitp.ru/en/science) and is based on the article https://arxiv.org/abs/1805.01934 (SID.pdf)

The problem is about adjusting the existing NN pipeline for an arbitrary photo camera with the aim to transform short exposed raw photos into high-quality RGB images with the help of U-net with output quality better than hardcoded ISP algorithm in camera can provide. PSNR AND SSIM metrics are used to compare output of the NN with the ground truth (long exposed photo). The issue of applicability of a pretrained model for a new camera (another image sensor) as well as the reducing of the train set are investigated too. 
 
All other experiments, datasets used and Tensrflow code are stored in this Google Drive: https://drive.google.com/drive/folders/1LC2DO_McdGIn8e3MNfx_SixUIco0M240?usp=sharing 

All held experiments and achieved results are described in Report.pdf
