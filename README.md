# 1 Python Camera Library

[![pre-commit](https://github.com/JarnoRalli/python_camera_library/actions/workflows/pre-commit.yml/badge.svg?branch=main&event=push)](https://github.com/JarnoRalli/python_camera_library/actions/workflows/pre-commit.yml)

This repository demonstrates how several different camera models, such as the rectilinear (pinhole), omnidirectional and the fish-eye camera models operate.
The rectilinear camera module contains, for example, forward- and back-projection functions, the omnidirectional camera module contains functionality for
creating look-up-table (LUT) that can be used in interpolation for converting images taken with omnidirectional camera into rectilinear images.
Here forward projection refers to how 3D points, from the real world, are projected onto a pinhole camera, and the image plane. Similarly backward projection refers to the opposite, i.e. back projecting image points, as seen in the image plane, back to the 3D world.

## 1.1 Installing the Library

You can either build and install the library or install it in editable mode. In editable mode you can make modifications to the library, and those modifications are reflected immediately in the functionality. Following
shows how to build and install the library:

```bash
python -m build .
pip install ./dist/python_camera_library-1.0.0-py3-none-any.whl
```

Following shows how to install the library in editable mode:

```bash
pip install --editable .
```

## 1.2 Directories
* [conda_environments](./conda_environments/README.md) contains a YAML file for generating a Conda environment that can be used to execute the examples
* [documentation](./documentation) contains Jupyter notebooks and other documentation
* [python_camera_library](./python_camera_library) contains the actual camera library
* [test_data](./test_data/README.md) contains test data used in the examples. You need to pull the data with `git lfs pull`
* [tests](./tests) contains tests for the modules

## 1.3 Modules
* [Rectilinear camera](./python_camera_library/rectilinear_camera.py)
  * Rectilinear (pinhole) camera module
* [Omnidirectional camera](./python_camera_library/omnidirectional_camera.py)
  * Omnidirectional camera module
* [Equidistant fish-eye camera](./python_camera_library/fisheye_camera/equidistant.py)
  * Equidistant fish-eye camera
* [Equisolid fish-eye camera](./python_camera_library/fisheye_camera/equisolid.py)
  * Equisolid fish-eye camera
* [Homography](./python_camera_library/homography.py)
  * Homography module that contains, for example, the DLT-algorithm (Direct Linear Transformation) for estimating homographies

## 1.4 Examples
* [KITTI example](./kitti_example.py)
  * Shows how to change vantage point between different sensors
* [Stereo camera example](./stereo_camera_example.py)
  * Shows how a depth map can be converted into a point cloud
* [Omnidirectional camera example](./omnidirectional_example.py)
  * Omnidirectional camera remapping into a pin-hole camera
* [Homography example](./homography_example.py)
  * The homography example shows how to calculate a homography between an object defined in the object coordinates and image of that object seen in the image plane
* [Fisheye camera models](./documentation/fisheye_camera_models.ipynb)
  * A Jupyter notebook that demonstrates how both the equidistant- and equisolid fish-eye cameras work

Steps before running the examples:
* After cloning the repo, pull the test data (LFS) using `git pull` or `git lfs pull`.
  * Take a look [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) regarding how to install Git LFS.
* Create a Conda virtual environment as per these [instructions](./conda_environments/README.md)

### 1.4.1 KITTI Example

The KITTI example demonstrates how lidar generated point clouds can be transformed to different coordinate frames. Outputs PLY files that can be viewed, for example, using [CloudCompare](https://www.danielgm.net/cc/) or [MeshLab](https://www.meshlab.net/).

```python
python ./kitti_example.py
```

Above code generates the following PLY files:

* 3d_cam0.ply
* 3d_cam2.ply
* 3d_lidar.ply
* 3d_proj_cam2.ply

### 1.4.2 Stereo Camera Example

The stereo camera example demonstrates how a X- and Y-coordinates can be generated based on a depth map (Z-coordinate). Outputs a PLY file that can be viewed, for example, using [CloudCompare](https://www.danielgm.net/cc/) or [MeshLab](https://www.meshlab.net/).

```python
python ./stereo_camera_example.py
```

Above code generates the following PLY file:

* stereo_camera.ply

### 1.4.3 Omnidirectional Camera Example

The fish-eye camera example shows images captured using a camera with "fish-eye" lenses, described using the [omnidirectional camera model](http://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf), can be re-mapped into a pin-hole camera image.

```python
python ./fisheye_example.py
```

### 1.4.4 Homography Example

The homography example shows how to calculate a homography between an object defined in the object coordinates and image of that object seen in the image plane. Extracts R (rotation matrix) and t (translation vector)
from the calculated homography.

```python
python ./homography_example.py
```

## 1.5 Test Data

The directory `./test_data` contains test 3D point clouds and images. The KITTI data is from the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) [raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). The fish-eye camera data is from the [OCamCalib Toolbox](https://sites.google.com/site/scarabotix/ocamcalib-toolbox/ocamcalib-toolbox-download-page) (University of Zurich).

For convenience, I have provided some data from one of the raw KITTI datasets, and a test image from the OCamCalib Toolbox. The stereo camera data contains a pair of rectified stereo camera images, along with the rectified camera parameters,
and a depth map. **The test data needs to be pulled using `git pull` or `git lfs pull`**.

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

If you use the omnidirectional camera data, please cite the following.

```
{
  @ARTICLE{Scaramuzza2006ICVS,
  author = {Scaramuzza, D., Martinelli, A., Siegwart, R.},
  title = {A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion},
  journal = {Proceedings of IEEE International Conference of Vision Systems (ICVS'06), New York, January 5-7, 2006},
  year = {2006}
}
```

