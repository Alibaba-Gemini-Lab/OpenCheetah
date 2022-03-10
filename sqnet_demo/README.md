## More Images for SqueezeNet

This folder contains some scripts to prepare more inputs for the SqueezeNet. 
These scripts all taken from the [EzPc](https://github.com/mpc-msri/EzPC/tree/master/Athos) project with some modifications.

Use the bash script `bash run.sh` to finish all the work. Basically this script do three things.

1. First fetech a small subset of ImageNet from https://github.com/EliSchwartz/imagenet-sample-images
2. Create a folder for the preprocessed images `preprocessed/`
3. Run the python script `squeezenet_main.py` which will run prediction on these images and save the preprocessed images to folder `preprocessed/`

