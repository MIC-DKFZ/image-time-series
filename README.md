# Image Time Series

This repository contains code and experiments for interpolation of image/segmentation time series that are sparse and irregularly sampled (and possibly stochastic). The first publication in this line of work is our MICCAI 2021 paper called ["Continuous-Time Deep Glioma Growth Models"](https://arxiv.org/abs/2106.12917). We're still actively working on the code and have added a number of options that go beyond what was presented in the paper (e.g. working without context segmentations).

## Installation

The package is not yet on PyPI, so you will have to clone the repository and then do

    pip install -e path/to/this/folder

The `-e` is recommended because we still push changes regularly that you can then just pull if you installed with this flag. The installed package is available via

    import gliomagrowth as gg

## Package Structure

At some point we will hopefully add proper documentation, the docstrings are already prepared and quite comprehensive. Until then, here's a quick overview of the package structure:

* data (dataloaders etc.)
    * glioma (what we used for our experiments)
* eval (metrics and stuff)
    * metrics (so far only dice and a generic evaluator to be expanded)
* experiment (experiment scripts)
    * continuous_image (continuous-time and with images, more will follow e.g. scalar)
* nn (model components)
    * attention (contains a re-implementation of PyTorch's MultiheadAttention)
    * block (reusable components, encoder, decoder, etc.)
    * loss (Losses obv., also put your own custom losses here)
    * neuralprocess (combines components to build NPs and related models)
* util (utility stuff)
    * lightning (anything to do with running experiments)
    * util (generic)

## Running Experiments

We are using PyTorch Lightning for our experiments and do logging with MLFlow and Visdom (mostly while prototyping). It should be trivial to include other loggers or use Lightning's default logging, but we didn't check :)

### Example command

Here's an exemplary command to run an experiment, we explain below what everything means

    python continuous_image.py /YOUR/LOGGING/LOCATION/mlflow --data glioma --data_dir /YOUR/DATA/LOCATION --gpus 1 --visdom 8080 --split_val 3 --split_test 4 --model_spatial_attention 1 --model_temporal_attention 0 --model_upsample convtranspose --model_upsample_kwargs kernel_size=4 padding=1 stride=2

This assumes you're in the experiment module. Your always need to provide a logging location, the other arguments are optional

* `--data` selects the data module. At the moment `glioma` is the only one there is, but if you implement your own, e.g. for other types of tumor, you need to use this selector (it does a getattr on the data module).
* `--data_dir` tells the experiment where the data is. You can also hardcode a default `data_dir` in the data modules and then leave this option out.
* `--gpus` is a Lightning option that selects the number of GPUs (without it you're running on CPU!!).
* `--visdom` activates Visdom logging the specified port. Make sure to start the Visdom server beforehand, e.g. with `python -m visdom.server -port 8080`.
* `--split_val` selects the validation set. By default we split the data into 5 sets and select val and test via indices.
* `--split_test` See previous option. Make sure this is always the same, otherwise you will be optimizing on the test set!
* `--model_spatial_attention` selects the number of spatio-temporal attention mechanism. This counts from the bottom up, so 1 will add one mechanism at the lowest possible resolution. Reducing this is useful for workstation prototyping, as it's the main source of GPU memory consumption.
* `--model_temporal_attention` selects the number of temporal attention blocks. This also counts from the bottom up, but these will be inserted above (i.e. higher resolution) the spatio-temporal attention.
* `--model_upsample` selects the upsampling operator in the decoder. The default is Upsample, and this example shows how to use ConvTranspose upsampling instead.
* `--model_upsample_kwargs` provides the initialization arguments for the previous arguments. It shows you the general pattern of how to work with dictionary arguments.

### Working with nn.Module and dictionary arguments

Many of the experiment arguments can select an operator/nn.Module to use for a certain task. In the example above, we want to use nn.ConvTranspose2d/3d upsampling instead of nn.Upsample. You can do this by simply supplying the name of the desired operator as a string, either fully lowercase or precisely capitalized (Convtranspose wouldn't work!). You don't have to include the dimension argument, but if you do, it has to be correct (if the `dim` argument is 2 and you select ConvTranspose3d, it won't work!). If your argument is `convtranspose`, the matching order would be `convtranspose`, `ConvTranspose`, `ConvTransposeNd`, in case multiple matches are possible. The function that does this lookup is `util.util.nn_module_lookup`. It also works with custom losses in our `nn.loss` module. The operator arguments will also have a corresponding kwargs argument where you can provide a dictionary for initialization. This is simply filled by adding `key=value` arguments on the command line.

## Preparing Your Data

Unfortunately we were not yet able to get clearance to publish our data. If you want to work with your own data, you can use the dataloaders provided in `data.glioma`. We expect a folder with the following (if you want to use the dataloaders without changes):

* Data as `patientID.npy` files, where each file has shape (T,C,X,Y,Z), and the last entry along the channel axis is the segmentation. If you want to draw 2D slices, we strongly suggest you use the X axis, it's much faster.
* `default_splits.json`, which contains a list of lists of patient IDs.
* `multi_days.csv`, where each row is a patient ID followed by the time values of the scans (i.e. the first axis of the corresponding array).
* `multi_overlaps.json`, a dictionary of patient IDs to lists, where each list contains the (foreground) Dice overlaps of consecutive time points. We only use this for evaluation to filter out certain cases.
* `multi_tumor_crop.json`, which is a dictionary of patient IDs to tumor bounding boxes ([[min_x, min_y, min_z], [max_x, max_y, max_z]]). Each time point obviously has its own bounding box, the joint box for a patient is then the box that contains all of the individual ones.

We also recommend to work with cropped data. In the paper we first removed all zero-valued regions around the brain and then resampled the resulting arrays to 128^3.

## Pre-trained Models

Will be added soon :)

## Cite

If you found this repository useful or our paper is relevant to your work, please cite using this entry


    @inproceedings{petersen_deepgliomagrowth_2021,
	    title = {Continuous-Time Deep Glioma Growth Models},
        pages = {83--92},
	    series = {Lecture Notes in Computer Science},
        booktitle = {Medical Image Computing and Computer Assisted Intervention – {MICCAI} 2021},
        publisher = {Springer International Publishing},
        author = {Petersen, Jens and Isensee, Fabian and Köhler, Gregor and Jäger, Paul F. and Zimmerer, David and Neuberger, Ulf and Wick, Wolfgang and Debus, Jürgen and Heiland, Sabine and Bendszus, Martin and Vollmuth, Philipp and Maier-Hein, Klaus H.},
        editor = {de Bruijne, Marleen and Cattin, Philippe C. and Cotin, Stéphane and Padoy, Nicolas and Speidel, Stefanie and Zheng, Yefeng and Essert, Caroline},
        year = {2021},
    }
