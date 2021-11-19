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

def dlt_homography(x: np.array, xp: np.array):
    """
    Calculates a 2D homography 3x3 matrix H, such that xp = Hx. Uses DLT, direct linear transformation, for calculating the solution.

    Parameters
    ----------
    x : numpy.array, shape (3, nr_points)
            2D points in homogeneous format
    xp : numpy.array, shape (3, nr_points)
            2D points in homogeneous format

    Returns
    -------
    H : numpy.array
        A 3x3 homography matrix that satisfies xp = Hx
    """

    if x.shape[0] != 3 or xp.shape[0] != 3:
        raise Exception("both x and xp must be in homogeneous coordinates")

    if x.shape != xp.shape:
        raise Exception("x and xp must be of the same size")

    nr_points = x.shape[1]
    
    if nr_points < 4:
        raise Exception("at least 4 points are needed")

    # Normalize the points for numerical stability
    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)
    xp_mean = np.mean(xp, axis=1)
    xp_std = np.std(xp, axis=1)

    if np.isclose(x_std[0], 0.0):
        raise Exception("standard deviation of x is zero or close to zero")

    if np.isclose(x_std[1], 0.0):
        raise Exception("standard deviation of y is zero or close to zero")

    if np.isclose(xp_std[0], 0.0):
        raise Exception("standard deviation of xp is zero or close to zero")

    if np.isclose(xp_std[1], 0.0):
        raise Exception("standard deviation of yp is zero or close to zero")

    T = np.array([
        [1.0/x_std[0], 0, -x_mean[0]/x_std[0]],
        [0, 1.0/x_std[1], -x_mean[1]/x_std[1]],
        [0, 0, 1]
    ])
    Tp = np.array([
        [1.0/xp_std[0], 0, -xp_mean[0]/xp_std[0]],
        [0, 1.0/xp_std[1], -xp_mean[1]/xp_std[1]],
        [0, 0, 1]
    ])

    x_n = T @ x
    xp_n = Tp @ xp
    A = np.empty([0, 9])

    # This is not Pythonic at all and needs to be re-written
    for i in range(nr_points):
        x = x_n[0, i]
        y = x_n[1, i]
        w = x_n[2, i]
        xp = xp_n[0, i]
        yp = xp_n[1, i]
        wp = xp_n[2, i]

        A = np.vstack((A, 
            [[0, 0, 0, -wp*x, -wp*y, -wp*w, yp*xp, yp*y, yp*w],
            [wp*x, wp*y, wp*w, 0, 0, 0, -xp*xp, -xp*y, -xp*w]]
            ))

    # DLT solution to minimizing ||Ah|| with the contraint ||h|| = 1, since h=0 is
    # not of practical use
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    H = vh[-1, :].reshape([3, 3])

    # De-normalization
    H = np.linalg.inv(Tp) @ H @ T
    H = H/H[-1, -1]

    return H

def extract_transformation(K: np.array, H: np.array, xp = None, x = None):
    """
    Extract rotation and translation (up to scale) from a homography matrix.

    Homography is defined as xp = H*x
    H = K[r1 r2 t], where R = [r1 r2 cross(r1, r2)]

    Parameters
    ----------
    K : numpy.array, shape (3, 3)
            Camera intrinsic matrix
    H : numpy.array, shape (3, 3)
            Homography matrix

    Returns
    -------
    R : numpy.array
        A 3x3 rotation matrix
    t : numpy.array
        3x1 translation vector, defined up to scale
    """

    # Multiple View Geometry in Computer Vision, page 196.
    # For a calibrated camera, the homography between a world plane at Z = 0 and the image is H = K[r1 r2 t] where ri are the columns of R.
    H_ = np.linalg.inv(K) @ H

    r1 = H_[:, 0]
    r2 = H_[:, 1]
    t = H_[:, 2]
    t /= np.linalg.norm(t)
    t = t.reshape(3,1)

    # Orthogonalize
    # We keep r1 fixed and project r2 to r1, i.e. proj = dot(r2,r1) / dot(r1, r1) * r1 and then
    # r2p = r2 - proj. Now r1 and r2 should be orthogonal
    proj = (np.dot(r2, r1) / np.dot(r1, r1)) * r1
    r2 -= proj
    r3 = np.cross(r1, r2)

    # Normalize each direction vector
    r1 /= np.linalg.norm(r1)
    r2 /= np.linalg.norm(r2)
    r3 /= np.linalg.norm(r3)
    R = np.hstack((r1.reshape(-1,1), r2.reshape(-1,1), r3.reshape(-1,1)))

    # Rank of the rotation matrix needs to be 3, otherwise it does not span a 3D space
    if not np.linalg.matrix_rank(R) == 3:
        raise Exception("rank of the rotation matrix is not 3, i.e. it does not span a 3D space")

    # Test for R*R' = I
    if not np.isclose(R @ np.transpose(R), np.eye(3) ).all():
        raise Exception("R*R' is not I, therefore R is not a rotation matrix")

    return R, t

def transformation_scale(K: np.array, R: np.array, t: np.array, uv, x):
    """
    Solves for a scale factor of a translation vector when K, R, t/|t|, uv and x are known.

    uv = K [R |t]x
    inv(K)*uv = R*x + t, where v1 = inv(K)*uv and v3 = t. v1 and v2 are vectors which lengths are not known, and they might
    not intersect due to numerical errors. The shortest 'line' between v1 and v2 is defined by a vector v3 = cross(v1, v2).
    Therefore, v1*s1 + v2*s2 - v3*s3 = R*X

    WARNING! This is version does not give the correct solution if X = [0 0 0]',
    since trivial solution s=[0 0 0] would be a solution.

    Parameters
    ----------
    K : numpy.array, shape (3, 3)
            Camera intrinsic matrix
    R : numpy.array, shape (3, 3)
            Rotation matrix
    t : numpy.array, shape(3, 1)
            Translation vector
    uv : numpy.array, shape(3, 1)
            Point uv seen in the image plane
    x : numpy.array, shape(3, 1)
            Point in the object coordinate system

    Return
    ------
    scale: float
        Scale of the translation vector
    """

    if np.isclose(x.reshape(3,1), np.zeros([3,1])).all():
        raise Exception("x = [0 0 0]' would lead to a trivial solution, where scale would be 0.0")

    if K.shape != (3, 3):
        raise Exception("K must be a 3x3 matrix")

    if R.shape != (3, 3):
        raise Exception("R must be a 3x3 matrix")

    if uv.size != 3:
        raise Exception("uv must have 3-components")

    if x.size != 3:
        raise Exception("x must have 3-components")

    # Test for R*R' = I
    if not np.isclose(R @ np.transpose(R), np.eye(3) ).all():
        raise Exception("R*R' is not I, therefore R is not a rotation matrix")

    t = t.flatten()
    uv = uv.flatten()

    v3 = t.flatten()
    v3 /= np.linalg.norm(v3)

    v1 = np.linalg.inv(K) @ uv
    v2 = np.cross(v1, v3)
    
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    
    A = np.hstack((v1.reshape(3,1), v2.reshape(3,1), -v3.reshape(3,1)))
    b = R @ x.reshape(3,1)
    s = np.linalg.solve(A, b)

    return s[-1]

