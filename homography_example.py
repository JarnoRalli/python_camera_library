__author__ = "Jarno Ralli"
__copyright__ = "Copyright, 2021, Jarno Ralli"
__license__ = "3-Clause BSD License"
__maintainer__ = "Jarno Ralli"
__email__ = "jarno@ralli.fi"
__status__ = "Development"

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from python_camera_library import rectilinear_camera as cam
from python_camera_library import homography as homog
from python_camera_library import utils


# This example tests the homography library. We calculate the homography between 4 3D points (points_object) given in the object coordinate system.
# points_object are converted into camera coordinate system using points_camera = [R | t] * points_object, and then projected to the image plane uv.
# After this we extract the rotation matrix R, and the translation vector t/||t||, from the homography. Finally we calculate the scale factor s that,
# when applied, t == s*t/||t||.

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# Camera intrinsic matrix
K = np.array([[100, 0, 150], [0, 100, 150], [0, 0, 1]])

# 3D points defined in the object coordinate system, i.e. z=0
# Object and camera coordinates overlap
points3D_obj = np.array([[0, 10, 0, 10], [0, 0, 10, 10], [0, 0, 0, 0]])

# Transformation that moves the object coordinate system
R = utils.rotation(np.pi / 4, 0, np.pi / 4)
t = np.array([1, 2, 10]).reshape(3, 1)

# Transformed object coordinates
points3D_cam = R @ points3D_obj + t

# Project object coordinates (transformed) to the camera plane
(pts3D, uv, depth) = cam.forwardprojectK(points3D_cam, K, (6000, 6000))

# Calculate homography such that uv = H * points3D.
# First make the coordinates homogeneous
points3D_h = np.copy(points3D_obj)
points3D_h[-1, :] = np.ones([1, 4])
H = homog.dlt_homography(points3D_h, uv)

print("----------------------")
print("Calculated homography:")
print(H)

# Extract R_ and t_ from the homography, using K
# R is 'fully defined', whereas t_ is defined only up to scale
(R_, t_) = homog.extract_transformation(K, H)

scale = homog.transformation_scale(K, R_, t_, uv[:, 1], points3D_obj[:, 1])
t_ *= scale

print("----------------------")
print("t and t_ should be the same, t_ is the scaled and extracted translation")
print("t: ")
print(t)
print("t_: ")
print(t_)
print("----------------------")
print("R and R_ should be the same, R_ is the extracted rotation")
print("R: ")
print(R)
print("R_: ")
print(R_)
print("----------------------")
