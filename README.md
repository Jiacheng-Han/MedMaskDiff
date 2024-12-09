<div align="center">
<h1>MedMaskDiff</h1>
<h3>Mamba-based Medical Semantic Image Synthesis for Segmentation</h3>
</div>

**1. Datasets.** </br>
For CT images, we selected the classic liver dataset LiTS and randomly chose 1030 liver slices. For ultrasound images, we selected the thyroid nodule dataset TG3K and randomly chose 1500 thyroid ultrasound images. For cellular microscopy, we used all the low-grade intraepithelial neoplasia images from the EBHI-SEG dataset.
- [LiTS.zip](https://pan.baidu.com/s/13DVRwZTf00kilcrhppU-xA)
- [TG3K.zip](https://pan.baidu.com/s/1DqlqZtC1X0LiB9VHfx1OEQ)
- [EBHI-SEG.zip](https://pan.baidu.com/s/1goFgtYOMIrWf46K1X8Zsvw)

**2. Train the KMC-UNet** </br>
You can try using the model in `MedMaskDiff.py`.

## Acknowledgement
Thanks to [VMamba](https://github.com/MzeroMiko/VMamba) for their outstanding work.