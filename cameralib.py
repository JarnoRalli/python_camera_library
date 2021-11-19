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

def homogenise(points: np.array):
    """Converts non-homogeneous 2D or 3D coordinates into homogeneous coordinates.

    For example
    [x1 x2 x3]    [x1 x2 x3]
    [y1 y2 y3] -> [y1 y2 y3]
                  [1  1  1]

    Parameters
    ----------
    points : numpy.array, shape (dimensions, nr_points)

    Returns
    --------
    points : numpy.array, shape (dimensions+1, nr_points)
    """

    return np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)


def concatenateRt(R: np.array, t: np.array):
    """Concatenates a rotation matrix R and a translation vector t into a transformation matrix T.

    The outcome is as follows:
    [ R      | t]
    [[0 0 0] | 1]

    Parameters
    ----------
    R : numpy.array, shape (3, 3)
        A rotation matrix

    t : numpy.array, shape (3, 1)
        A translation vector

    Returns
    -------
    numpy.array, shape (3, 4)
        Transformation matrix
    """

    return np.concatenate((np.hstack((R, t)), np.array([[0, 0, 0, 1]])), axis=0)


def transform(T: np.array, points: np.array):
    """Apply transformation T to 3D points.

    For example, if the transformation T is defined as follows:
    [ R      | t]
    [[0 0 0] | 1]
    , where R is a 3x3 rotation matrix and t is a 3x1 translation vector,
    then the results is as follows:
    [ R      | t] * [P]
    [[0 0 0] | 1]   [1]
    , where P is a 3xn vector of n points.

    Parameters
    -----------
    T : numpy.array, shape (4, 4)
            Transformation matrix

    points : numpy.array, shape (3, num_points)
            Points to be transformed

    Returns
    -------
    numpy.array, shape (3, num_points)
            Transformed points
    """

    try:
        points = homogenise(points)
        points = np.matmul(T, points)
        return points[:3, :]
    except Exception as e:
        raise


def backproject(uv: np.array, K: np.array, normalize=False):
    """Back projects 2D pixel coordinates, defined in image coordinates, into 3D direction vectors,
    defined in the camera coordinate system. The direction vectors are defined by the origin of 
    the camera and the pixel position (u, v).
    
    If the resulting vector is not normalized, then the output vector is xi + yj + 1k.
    3D position [X, Y, Z] can be calculated, if we know the depth
    (i.e. Z-coordinate), as follows: X = x*Z, Y = y*Z and Z = Z.

    If the resulting vector is normalized, then the output vector is 
    (xn*i + yn*i + zn*i) = (x*i + y*j + z*k)/|x*i + y*j + z*k|.
    3D position [X, Y, Z] can be calculated, based on the length of the ray r, as follows:
    X = xn*r, Y = yn*r, Z = zn*r.

    Parameters
    ----------
    uv : numpy array
        Homogeneous pixel coordinates [u1, u2, u3, ...; v1, v2, v3, ...; 1, 1, 1, ...]
    K : numpy array
        Camera calibration matrix [fx, s, px; 0, fy, py; 0 0 1]

    normalize : boolean, optional (defaults to False)
        False = do not normalize the output vector
        True = normalize the output vector

    Returns
    -------
    numpy array
        If not normalized: 3D direction vector, in metric space, [x1, x2, x3, ...; y1, y2, y3, ...; 1, 1, 1, ...]
        If normalized: 3D direction vectors, in metric space, [xn1, xn2, xn3, ...; yn1, yn2, yn3, ...; zn1, zn2, zn3, ...]
    """

    # Verify that camera matrix K is a numpy matrix
    if not isinstance(K, np.ndarray):
        raise Exception("Camera matrix K is expected to be of numpy type")
    # Verify that camera matrix K is of size 3x3
    if K.shape != (3, 3):
        raise Exception("Camera matrix K needs to be of size 3x3")

    # Verify that uv vector is a numpy matrix
    if not isinstance(uv, np.ndarray):
        raise Exception("'uv' matrix is expected to be of numpy type")
    # Verify that uv vector is 'homogeneous'
    if uv.shape[0] != 3:
        raise Exception("'uv' matrix needs to be homogeneous: [u;v;1]\'")

    result = np.matmul(np.linalg.inv(K), uv)
    # Normalize homogeneous coordinates
    result[0,] = result[0,] / result[2,]
    result[1,] = result[1,] / result[2,]
    result[2,] = result[2,] / result[2,]

    if normalize:
        magnitude = np.sqrt(np.power(result[0,], 2) + np.power(result[1,], 2) + np.power(result[2,], 2))
        result[0,] = np.divide(result[0,], magnitude)
        result[1,] = np.divide(result[1,], magnitude)
        result[2,] = np.divide(result[2,], magnitude)

    return result


def forwardprojectK(points: np.array, K: np.array, image_size, image=None):
    """Project 3D points, defined by [X Y Z]', onto an image plane of a camera defined by
    a 3x3 camera matrix K.

    Forward projects 3D points onto an image plane of a camera defined by the 3x3 camera matrix K.
    Returns those 3D points that are withing the FOV of the camera (i.e. filters out those points
    that are outside of the FOV), the corresponding uv-image coordinates, and a depth map.
    Additionally, if an image is given, RGB for each 3D point is returned.

    Parameters
    ----------
    points : numpy.array, shape (3, nr_points)
            3D points
    K : numpy array, shape (3, 3)
            Camera intrinsic matrix
    image_size : tuple
            Image size, (rows, cols)
    image : numpy.array, optional
            Image used for defining colors for each point (default is None).

    Returns
    -------
    (3D points, uv-coordinates, depth map) : numpy array
        If no image is given (i.e. is None). Shapes are (3, nr_points), (3, nr_points) and (rows, cols)
    (3D points, uv-coordinates, RGB, depth map) : numpy array
        If image is given. Shapes are (3, nr_points), (3, nr_points), (3, nr_points) and (rows, cols)"""

    # Convert the image_size into a tuple. It might already be a tuple, but let's just make sure
    image_size = (int(image_size[0]), int(image_size[1]))
    depth_map = np.ones(image_size) * np.nan

    try:
        # We expect that the points are given in the camera coordinate frame, so remove points that are
        # behind the camera, i.e. where the Z-coordinate is negative
        mask = points[2, :] <= 0.0
        points = points[:, ~mask]

        # Project points to image
        uv = np.matmul(K, points)
        # Normalize coordinates
        uv[0, :] /= uv[2, :]
        uv[1, :] /= uv[2, :]
        uv[2, :] /= uv[2, :]

        # Mask out points that don't fall withing the given image (i.e. are outside of FOV)
        mask = (uv[0, :] < 0) | (uv[0, :] > (image_size[1] - 1)) | (uv[1, :] < 0) | (uv[1, :] > (image_size[0] - 1))
        points = points[:, ~mask]
        uv = uv[:, ~mask]

        # Generate a depth map
        depth_map[np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int)] = points[2, :]

        # Handle colors, if given
        if image is None:
            return points, uv, depth_map
        else:
            RGB = image[np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int), :]
            return points, uv, RGB, depth_map
    except Exception as e:
        raise


def forwardprojectP(points: np.array, P: np.array, image_size, image=None):
    """Project 3D points, defined by [X Y Z]', onto an image plane of a camera defined by
    a 3x4 projection matrix P.

    Forward projects 3D points onto an image plane of a camera defined by the 3x4 projection matrix P.
    Returns those 3D points that are withing the FOV of the camera (i.e. filters out those points
    that are outside of the FOV), the corresponding uv-image coordinates, and a depth map.
    Additionally, if an image is given, RGB for each 3D point is returned.

    Parameters
    ----------
    points : numpy.array, shape (3, nr_points)
            3D points
    P : numpy array, shape (3, 4)
            Camera projection matrix P = K[R | t]
    image_size : tuple
            Image size, (rows, cols)
    image : numpy.array, optional
            Image used for defining colors for each point (default is None).

    Returns
    -------
    (3D points, uv-coordinates, depth map) : numpy array
        If no image is given (i.e. is None). Shapes are (3, nr_points), (3, nr_points) and (rows, cols)
    (3D points, uv-coordinates, RGB, depth map) : numpy array
        If image is given. Shapes are (3, nr_points), (3, nr_points), (3, nr_points) and (rows, cols)
    """

    # Convert the image_size into a tuple. It might already be a tuple, but let's just make sure
    image_size = (int(image_size[0]), int(image_size[1]))
    depth_map = np.ones(image_size) * np.nan

    try:
        # Convert points into homogeneous form
        points = homogenise(points)

        # Project points to image
        uv = np.matmul(P, points)

        # Filter points that fall behind the camera
        mask = uv[2, :] < 0.0
        uv = uv[:, ~mask]
        points = points[:, ~mask]

        # Normalize coordinates
        uv[0, :] /= uv[2, :]
        uv[1, :] /= uv[2, :]
        uv[2, :] /= uv[2, :]

        # Mask out points that don't fall withing the given image (i.e. are outside of FOV)
        mask = (uv[0, :] < 0) | (uv[0, :] > (image_size[1] - 1)) | (uv[1, :] < 0) | (uv[1, :] > (image_size[0] - 1))
        points = points[:, ~mask]
        uv = uv[:, ~mask]

        # Generate a depth map
        depth_map[np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int)] = points[2, :]

        # Handle colors, if given
        if image is None:
            return points[:3, :], uv, depth_map
        else:
            RGB = image[np.round(uv[1, :]).astype(int), np.round(uv[0, :]).astype(int), :]
            return points[:3, :], uv, RGB, depth_map
    except Exception as e:
        raise


def depthMapTo3D(depthMap: np.array, K: np.array, image=None):
    """Backprojects a depth map, defined by the Z-coordinate values, into 3D points [X Y Z]'.

    Parameters
    -----------
    depthMap : numpy.array
            depth map containing Z-coordinate values.

    K : numpy.array, shape (3, 3)
            Camera intrinsic parameters.

    image : numpy.array, optional (defaults to None)
            RGB image. If given, RGB values corresponding to 3D points are output

    Returns
    --------
    (3D points, uv) : numpy array
        If no image is given, then the 3rd coordinates, the corresponding uv (image) coordinates are returned
    (3D points, uv, RGB)
        If an image is given, then the 3rd coordinates, the corresponding uv (image) coordinates, and the corresponding
        RGB values are returned
    """

    # Generate pixel coordinates and stack them together
    u, v = np.meshgrid(np.arange(depthMap.shape[1], dtype=np.float), np.arange(depthMap.shape[0], dtype=np.float))
    uv_coords = np.vstack((u.flatten(), v.flatten(), np.ones(u.size)))

    # Remove nan:s
    depthMap = depthMap.flatten()
    mask = np.isnan(depthMap)
    depthMap = depthMap[~mask]
    uv_coords = uv_coords[:, ~mask]

    # Calculate normalized camera coordinates
    vector_mm = backproject(uv_coords, K)
    vector_mm[0, :] = np.multiply(vector_mm[0, :], depthMap)
    vector_mm[1, :] = np.multiply(vector_mm[1, :], depthMap)
    vector_mm[2, :] = depthMap

    if image is None:
        return vector_mm, uv_coords
    else:
        RGB = image.reshape((image[:, :, 0].size, -1))
        RGB = RGB[~mask]
        return vector_mm, image, RGB.transpose()
