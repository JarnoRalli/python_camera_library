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

import numpy as np
import cameralib as cam
import homographylib as hog

def rotation(rot_z: float, rot_y: float, rot_x: float) -> np.array:
    """Calculate a 3D rotation, defined by a rotation along the z-, y- and z-axes

    Parameters
    ----------
    rot_z : float
        Rotation with respect the z-axis, in radians
    rot_y : float
        Rotation with respect the y-axis, in radians
    rot_x : float
        Rotation with respect the x-axis, in radians

    Returns
    -------
    numpy.array, shape (3, 3)
        A 3x3 rotation matrix
    """
    rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0.0],
                   [np.sin(rot_z),  np.cos(rot_z), 0.0],
                   [0.0, 0.0, 1.0]])
    ry = np.array([[np.cos(rot_y), 0.0, np.sin(rot_y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(rot_y), 0.0, np.cos(rot_y)]])
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(rot_x), -np.sin(rot_x)],
                   [0.0, np.sin(rot_x), np.cos(rot_x)]])

    return rz @ ry @ rx

# This example tests the homography library. We calculate the homography between 4 3D points (points_object) given in the object coordinate system.
# points_object are converted into camera coordinate system using points_camera = [R | t] * points_object, and then projected to the image plane uv.
# After this we extract the rotation matrix R, and the translation vector t/||t||, from the homography. Finally we calculate the scale factor s that,
# when applied, t == s*t/||t||.

# Camera intrinsic matrix
K = np.array([
    [100, 0, 150],
    [0, 100, 150],
    [0, 0, 1]
])

# 3D points defined in the object coordinate system, i.e. z=0
# Object and camera coordinates overlap
points3D_obj = np.array([
    [0, 10, 0, 10],
    [0, 0, 10, 10],
    [0, 0, 0, 0]
])

# Transformation that moves the object coordinate system
R = rotation(np.pi/4, 0, np.pi/4)
t = np.array([1, 2, 10]).reshape(3,1)

# Transformed object coordinates
points3D_cam = R @ points3D_obj + t

# Project object coordinates (transformed) to the camera plane
(pts3D, uv, depth) = cam.forwardprojectK(points3D_cam, K, (6000, 6000))

# Calculate homography such that uv = H * points3D.
# First make the coordinates homogeneous
points3D_h = np.copy(points3D_obj)
points3D_h[-1, :] = np.ones([1,4])
H = hog.dlt_homography(points3D_h, uv)

# Extract R_ and t_ from the homography, using K
# R is 'fully defined', whereas t_ is defined only up to scale
(R_, t_) = hog.extract_transformation(K, H)

scale = hog.transformation_scale(K, R_, t_, uv[:,1], points3D_obj[:,1])
t_ *= scale

print('These should be the same')
print('t: ')
print(t)
print('t_: ')
print(t_)
print('R: ')
print(R)
print('R_: ')
print(R_)

