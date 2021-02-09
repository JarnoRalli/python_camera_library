# Python Camera Library

This project demonstrates a simple pinhole camera library, implemented in Python using numpy. The library implements
a few simple functions, such as forward- and back-projections. Here forward projection refers to how 3D points, from the
real world, are projected onto a pinhole camera, and the image plane. In this context backward projection refers to the opposite, i.e. back projecting image points, as seen in the image plane, back to the 3D world.

Files
* Camera library `./cameralib.py`
* KITTI examples `./kitti_example.py`
* Stereo camera examples `./stereo_camera_example.py`

Directories
* `./conda_environments/` contains a YAML file for generating a Conda environment that can be used to execute the examples
* `./test_data/` contains test data used in the examples

## Examples

### KITTI Examples

The KITTI examples demonstrate how the lidar generated point cloud can be transformed to the point of view of a camera.

### Stereo Camera Examples

The stereo camera examples demonstrate how a X- and Y-coordinates can be generated based on a depth map (Z-coordinate).

## Test Data

The directory `./test_data` contains test 3D point clouds and images. The KITTI data is from the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) [raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). For convenience, I have provided some data from one of the raw KITTI datasets. The stereo
camera data contains a pair of rectified stereo camera images, along with the rectified camera parameters, and a depth map.

KITTI lidar                     |  Stereo camera 3D-reconstruction
:--------------------------------:|:-------------------------:
![](./test_data/kitti_lidar.png)  |  ![](./test_data/stereo_camera_3d.png)


If you use any data from the KITTI raw dataset, please cite the following.

```
{
  @ARTICLE{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
} 
```

