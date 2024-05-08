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

import cv2
from python_camera_library import rectilinear_camera
import open3d as o3d
from pathlib import Path
import sys
import os


def read_camera_configuration(file_name: str):
    """Reads a camera configuration file.

    Parameters
    ----------
    file_name : str
            Name of the file to be read.

    Returns
    -------
    (K, lens_param, R, t, resolution)
    """

    if not Path(file_name).is_file():
        print("File " + file_name + " does not exist")
        sys.exit()

    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    K = fs.getNode("k-matrix").mat()
    lens_param = fs.getNode("lens-coefficient").mat()
    R = fs.getNode("rotation-matrix").mat()
    t = fs.getNode("translation-vector").mat()
    img_resolution = (
        int(fs.getNode("image-resolution").at(0).real()),
        int(fs.getNode("image-resolution").at(1).real()),
    )
    fs.release()

    return K, lens_param, R, t, img_resolution


# --------------
# Test program
# --------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Read right camera (rectified) parameters from a file
(
    K_primary,
    lens_primary,
    R_primary,
    t_primary,
    img_res_primary,
) = read_camera_configuration("./test_data/stereo_camera/primary_camera_rectified.json")

# Read the depth map (i.e. Z-axis) from an OpenEXR file
Z = cv2.imread(
    "./test_data/stereo_camera/depthMap.exr", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
)
if Z is None:
    print(
        "Did not load the file './test_data/stereo_camera/depthMap.exr' properly, did you do: git lfs pull?"
    )

# Read the primary camera image (rectified)
left_rectified = cv2.imread("./test_data/stereo_camera/primary_image_rectified.png")

# Obtain X- and Y-coordinates based on the depth map
XYZ, uv, RGB = rectilinear_camera.depthMapTo3D(Z, K_primary, left_rectified)

# Write the XYZ points, with color information, into a ply-file
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(XYZ.transpose())
pcd.colors = o3d.utility.Vector3dVector((RGB / 255).transpose())
print("Writing file: stereo_camera.ply")
o3d.io.write_point_cloud("stereo_camera.ply", pcd)

# Show the point cloud for the user
print("Visualizing the point cloud")
o3d.visualization.draw_geometries([pcd])
