# Brightfield-Image-Segmentation

This is an imaging processing pipeline for segmenting warped brightfield images by domain structure. 
To install, use the command 

'''bash
    wget https://github.com/zhangxinqiao314/brightfield-image-segmentation.git
'''

Drag your input folder into the repository. The folder should have the following folder format, with brighfield images named by their temperature:
|_Environment_name
| |_Ramp_up
| | |_temperature.png
| |_Ramp_down
| | |_temperature.png

If you would like to store your data in zenodo, create an account and find your access key.

The analysis is broken into 3 notebooks:

1. Generate a windowed fft dataset given a series of brightfield images. Optional: Write to zenodo
2. Train and save model. Optional: Download windowed dataset.
3. 
