# Brightfield-Image-Segmentation

This is an imaging processing pipeline for segmenting warped brightfield images by domain structure. 

Download the repository from this main page. Drag your input folder into the repository. The folder should have the following folder format, with brighfield images named by their temperature:
```
|-- Environment_name
| |-- Ramp_up
| | |-- temperature.png
| |-- Ramp_down
| | |-- temperature.png
```
If you would like to store your data in zenodo, create an account and input your access key in the notebook

The analysis pipeline is broken into 3 notebooks:

1. Generate a windowed fft dataset given a series of brightfield images. Optional: Write to zenodo
2. Train and save model. Optional: Download windowed dataset.
3. Construct and Analyze Embeddings.

