# Python Camera Library

This project demonstrates a simple pin-hole camera library, together with forward- and back-projection functions, and an
implementation of an omnidirectional camera model that can be used, for example, for converting fish-eye images into pin-hole camera images.
Here forward projection refers to how 3D points, from the real world, are projected onto a pinhole camera, and the image plane. Similarly backward projection refers to the opposite, i.e. back projecting image points, as seen in the image plane, back to the 3D world.

**Files**
* Camera library `./cameralib.py`
* Omnidirectional camera library `./omnidirectional_camera.py`
* KITTI examples `./kitti_example.py`
* Stereo camera examples `./stereo_camera_example.py`
* Fish-eye camera remapping into a pin-hole camera `./fisheye_example.py`

**Directories**
* [conda_environments](./conda_environments/README.md) contains a YAML file for generating a Conda environment that can be used to execute the examples
* [test_data](./test_data) contains test data used in the examples

## Examples

Before executing the examples, create and activate a Conda virtual environment defined in `./conda_environments/3d-computer-vision.yml`. After cloning the repo, the test data used in the examples needs to be pulled 
using `git pull` or `git lfs pull`.

### KITTI Example

The KITTI example demonstrates how lidar generated point clouds can be transformed to different coordinate frames. Outputs PLY files that can be viewed, for example, using [CloudCompare](https://www.danielgm.net/cc/) or [MeshLab](https://www.meshlab.net/).

```python
python ./kitti_example.py
```

Above code generates the following PLY files:

* 3d_cam0.ply
* 3d_cam2.ply
* 3d_lidar.ply
* 3d_proj_cam2.ply

### Stereo Camera Example

The stereo camera example demonstrates how a X- and Y-coordinates can be generated based on a depth map (Z-coordinate). Outputs a PLY file that can be viewed, for example, using CloudCompare or MeshLab.

```python
python ./stereo_camera_example.py
```

Above code generate the following PLY file:

* stereo_camera.ply

### Fish-eye Camera Example

The fish-eye camera example shows images captured using a camera with "fish-eye" lenses, described using the [omnidirectional camera model](http://rpg.ifi.uzh.ch/docs/IROS06_scaramuzza.pdf), can be re-mapped into a pin-hole camera image.

```python
python ./fisheye_example.py
```

### Homography Example

The homography example shows how to calculate a homography between an object defined in the object coordinates and image of that object seen in the image plane. Extracts R (rotation matrix) and t (translation vector)
from the calculated homography.

```python
python ./homography_example.py
```

## Test Data

The directory `./test_data` contains test 3D point clouds and images. The KITTI data is from the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) [raw dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php). The fish-eye camera data is from the [OCamCalib Toolbox](https://sites.google.com/site/scarabotix/ocamcalib-toolbox/ocamcalib-toolbox-download-page) (University of Zurich).

For convenience, I have provided some data from one of the raw KITTI datasets, and a test image from the OCamCalib Toolbox. The stereo camera data contains a pair of rectified stereo camera images, along with the rectified camera parameters, 
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

If you use the fish-eye camera data, please cite the following.

```
{
  @ARTICLE{Scaramuzza2006ICVS,
  author = {Scaramuzza, D., Martinelli, A., Siegwart, R.},
  title = {A Flexible Technique for Accurate Omnidirectional Camera Calibration and Structure from Motion},
  journal = {Proceedings of IEEE International Conference of Vision Systems (ICVS'06), New York, January 5-7, 2006},
  year = {2006}
} 
```

