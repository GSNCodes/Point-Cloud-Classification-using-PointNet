# Point-Cloud-Classification-using-PointNet
Classifying point clouds of different objects using PointNet

We will be focussing on classifying point clouds as belonging to one of several different object classes.  

![](images/tasks.png)

## Architecture

Shown below is the architecture proposed by the authors of the PointNet paper. This implementation will take up the classification network alone.

![](images/architecture.jpg)

## Usage
Download the dataset from the website given here:- http://3dvision.princeton.edu/projects/2014/3DShapeNets/   
You can use either the Model10 or Model40 dataset, make sure you change the folder path to the dataset in the `config.py` file accordingly.  

To run the training process on the downloaded dataset:- `python pointnet.py`  

All hyperparameters such as learning rate, number of epochs, batch size, 
number of classes in the dataset, number of points in the pointcloud can be set in the `config.py` file

## Reference
[PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
