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

Before executing the examples, create and activate a Conda virtual environment defined in `./conda_environments/3d-computer-vision.yml`. After cloning the repo, the test data used in the examples needs to be pulled 
using `git pull` or `git lfs pull`.

### KITTI Example

The KITTI example demonstrates how lidar generated point clouds can be transformed to different coordinate frames. Outputs PLY files that can be viewed, for example, using CloudCompare or MeshLab.

```python
python ./kitti_example.py
```

### Stereo Camera Example

The stereo camera example demonstrates how a X- and Y-coordinates can be generated based on a depth map (Z-coordinate). Outputs a PLY file that can be viewed, for example, using CloudCompare or MeshLab.

```python
python ./stereo_camera_example.py
```

## Test Data

The directory `./test_data` contains test 3D point clouds and images. The KITTI data is from the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) [raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). For convenience, I have provided some data from one of the raw KITTI datasets. The stereo camera data contains a pair of rectified stereo camera images, along with the rectified camera parameters, 
and a depth map. Actual test data need to be pulled using `git pull` or `git lfs pull`.

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

