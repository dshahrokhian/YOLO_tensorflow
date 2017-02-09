# Tensorflow implementation of You Only Look Once

*This project is still under development*

An improvement of the implementation from @gliese581gg, with training, testing and video parsing.

We also used the VOC tools to parse the VOC dataset from @mprat

I know, I know: The code needs cleaning, bruh.

# Installation

## If you want to train the network yourself
`cd data_parsing`

`python data_parsing/setup.py install` or `python install develop` depending on what you will be doing

## If you want to use pre-trained weights
Download YOLO weight file from: [https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing]

Put the 'YOLO_small.ckpt' in the 'weight' folder of downloaded code

## Uninstall (training)
`cd data_parsing`

`python setup.py develop --uninstall`

# Usage
run `python network/Yolo_small_video -fromfile "name of file" -tofile_img "name file output"`
