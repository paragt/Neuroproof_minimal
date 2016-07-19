# Neuroproof_minimal

The bare-bones code set for superpixel agglomeration and classifier training. 

The methods implemeted in this code repo are described in the PloS ONE paper: Parag, T. et. al (2015). A Context-Aware Delayed Agglomeration Framework for Electron Microscopy Segmentation (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0125825)

This repository borrows implementations of the graph and feature datastructures from Janelia FlyEM NeuroProof (https://github.com/janelia-flyem/NeuroProof), which I am also a contributor to.


# Build
Linux: Install miniconda on your workstation. Create and activate the conda environment using the following commands:

  conda create -n my_conda_env -c flyem vigra opencv 

  source activate tst_my

Then follow the usual procedure of building:

  mkdir build
  cd build

  cmake -DCMAKE_PREFIX_PATH=<CONDA_ENV PATH>/my_conda_env ..

# Example

The inputs and outputs for the following example are self explanatory. The option -threshold indicates the boundary predictor confidence at which the agglomeration should stop. 

build/NeuroProof_stack -watershed watershed2_4class_600000_10_800_1000_1.0_2.h5  stack -prediction  pixel_prediction2_4class_600000_10_800_1000_1.0_2.h5  stack -classifier int_classifier2_600000_1000_800_e5000_1.xml -output /result_600000_800_1_mthd0.35_e500_thd0.3_m4.h5 stack -algorithm 1 -threshold 0.3



The necesary files are uploaded to Dropbox instead due to filesize limitation.

https://www.dropbox.com/sh/y90ygc8nunpolyw/AADnypNkfF7om067Z3PaXJ9ca?dl=0

More to come soon.
