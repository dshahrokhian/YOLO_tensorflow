# Tensorflow implementation of You Only Look Once

[![Video](https://i.imgur.com/szGut1Z.png)](https://www.youtube.com/watch?v=EJy0EI3hfSg)

An improvement of the implementation from @gliese581gg, with added training, testing and video parsing. We also used the VOC tools to parse the VOC dataset from @mprat.

I know, I know: the code needs cleaning, bruh.

# Installation

## Requirenments
- Tensorflow
- OpenCV2

## If you want to train the network yourself
`cd data_parsing`

`python data_parsing/setup.py install`

## If you want to use pre-trained weights
Download YOLO weight file from: [https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing].

Put the 'YOLO_small.ckpt' in the 'weight' folder of downloaded code.

## Uninstall (training)
`cd data_parsing`

`python setup.py develop --uninstall`

# Usage
## Images
`python network/YOLO_small_tf.py -fromfile "name of input file" -tofile_img "name of output file"`

## Videos
`python network/YOLO_small_tf.py -video "name of input file" -tofile_vid "name of output file"`

# License
Refeer to the LICENSE files of both *data_parsing* and *network*.
