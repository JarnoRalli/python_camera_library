__author__ = "Jarno Ralli"
__copyright__ = "Copyright, 2021, Jarno Ralli"
__license__ = "3-Clause BSD License"
__maintainer__ = "Jarno Ralli"
__email__ = "jarno@ralli.fi"
__status__ = "Development"

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
#INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
#OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import cameralib
import numpy as np
import open3d as o3d


# Source: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def readConfFile(fileName):
    """Reads a Kitti camera/velodyne configuration file.

    Name of the parameter and the data is separated by ':', i.e 'T: 0.0 0.0 0.0'.

    Parameters
    ----------
    fileName : str
            Name of the file to be read.

    Returns
    -------
    dictionary
            a dictionary containing the configuration data.
    """

    conf_dict = dict()
    try:
        with open(fileName) as f:
            for line in f:
                data = line.split(":")
                conf_dict[data[0]] = data[1]
        return conf_dict
    except Exception as e:
        raise


def extractMatrix(input_str, matrix_shape=None):
    """Convert a str into a matrix/vector.

    Parameters
    ----------
    input_str : str
            String to be converted into numpy matrix.

    matrix_shape : tuple
            Tuple defining the output shape.

    Returns
    -------
    numpy.array
            Numpy array that has the shape matrix_shape
    """

    try:
        if matrix_shape is None:
            output = np.fromstring(input_str, dtype=float, sep=' ').tolist()
        else:
            output = np.fromstring(input_str, dtype=float, sep=' ').reshape(matrix_shape)
        return output
    except Exception as e:
        raise

#--------------
# Test program
#--------------

# Read configuration files
cam_conf = readConfFile('./test_data/kitti/2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt')
lidar_conf = readConfFile('./test_data/kitti/2011_09_26_calib/2011_09_26/calib_velo_to_cam.txt')

lidar_data = np.transpose(load_velo_scan(
    './test_data/kitti/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin')[
                          :, :3])
image_data = np.array(cv2.imread(
    './test_data/kitti/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png'))

# Rotation and traslation from velodyne to camera 0
RvelTocam0 = extractMatrix(lidar_conf['R'], (3, 3))
TvelTocam0 = extractMatrix(lidar_conf['T'], (3, 1))
# Trans_velTocam0 = transform_from_rot_trans(RvelTocam0, TvelTocam0)
Trans_velTocam0 = cameralib.concatenateRt(RvelTocam0, TvelTocam0)

# Rotation and traslation from camera 0 to camera 2
Rcam0Tocam2 = extractMatrix(cam_conf['R_02'], (3, 3))
Tcam0Tocam2 = extractMatrix(cam_conf['T_02'], (3, 1))
Trans_cam0Tocam2 = cameralib.concatenateRt(Rcam0Tocam2, Tcam0Tocam2)

# Projection matrix from camera 2 to rectified camera 2
Pcam2 = extractMatrix(cam_conf['P_rect_02'], (3, 4))
Kcam2 = extractMatrix(cam_conf['K_02'], (3, 3))
Rcam2rect = extractMatrix(cam_conf['R_rect_02'], (3, 3))
im_size_rcam2 = extractMatrix(cam_conf['S_rect_02'])
im_size_rcam2.reverse()

# Extract K-matrix from the projection matrix P = K[R | t]
Kcam2rect = np.matmul(Pcam2[:3, :3], Rcam2rect.transpose())
#print("Kcam 2: " + str(Kcam2))
#print("Kcam rectified 2: " + str(Kcam2rect))

# Transform lidar points to camera 0 coordinate frame
lidar_data_cam0 = cameralib.transform(Trans_velTocam0, lidar_data)

# Transform lidar points from camera0 to camera 2 coordinate frame
lidar_data_cam2 = cameralib.transform(Trans_cam0Tocam2, lidar_data_cam0)

# Project lidar points into rectified camera 2
cam2_lidar, uv, RGB_lidar, depth_map = cameralib.forwardprojectP(lidar_data_cam2, Pcam2, im_size_rcam2, image_data)

# Write original lidar points into ply-file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_data.transpose())
print("Writing file: 3d_lidar.ply")
o3d.io.write_point_cloud("3d_lidar.ply", pcd)

# Write lidar points in cam0 coordinate frame points into ply-file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_data_cam0.transpose())
print("Writing file: 3d_cam0.ply")
o3d.io.write_point_cloud("3d_cam0.ply", pcd)

# Write lidar points in cam2 coordinate frame points into ply-file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_data_cam2.transpose())
print("Writing file: 3d_cam2.ply")
o3d.io.write_point_cloud("3d_cam2.ply", pcd)

# Write "filtered" points into ply-file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cam2_lidar.transpose())
pcd.colors = o3d.utility.Vector3dVector(RGB_lidar / 255)
print("Writing file: 3d_proj_cam2.ply")
o3d.io.write_point_cloud("3d_proj_cam2.ply", pcd)

