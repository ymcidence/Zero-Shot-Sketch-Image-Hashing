# Zero-Shot Sketch-Image Hashing
-------------------------------------------
-------------------------------------------
![fig](/pic/fig.jpg)

Sorry for the late release... It's been roughly a year since acceptance, but I was so lazy.

This is the released version of our work at CVPR2018:

[Yuming Shen, Li Liu, Fumin Shen, and Ling Shao. "Zero-shot sketch-image hashing." In CVPR. 2018.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Zero-Shot_Sketch-Image_Hashing_CVPR_2018_paper.pdf)


*Note that this is a simplified version of the original paper with feature inputs only for GENERALITY.*

*One can attach any CNN (+ attention) to the bottom of the network.*


## Requirements
* Python 3 (or 2 if you manually add the \_\_future\_\_ import)
* Tensorflow (any up-to-date version)
* Numpy, SkLearn, SciPy, six(not really)

## How to play?
Modify `try.py` to fit the document paths, and then
```
python try.py
```

## Data

We are using a lazy data format, i.e., `.mat` files...

Make sure that you have the following objects in the data file:
* **feat:** feature of the images/sketches
* **label:** label (this is for evaluation)
* **wv:** word vector of the label of each image/sketch

One needs to save the image data and the sketch data separately into two files.

If you are interested in getting the original image and sketch data, please refer to [this work](https://github.com/ymcidence/DeepSketchHashing).