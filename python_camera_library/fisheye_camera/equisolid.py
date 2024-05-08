__author__ = "Jarno Ralli"
__copyright__ = "Copyright, 2024, Jarno Ralli"
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
import math
import cv2 as cv2
from typing import Tuple, Union


def r_to_rd(focal_length: float, r: float) -> float:
    """Projects a location r, defined in a rectilinear camera, to a location rd defined in a equisolid camera.

    Parameters
    ----------
    focal_lenght : float
        Focal length
    r : float
        Location defined in the rectilinear camera

    Returns
    -------
    float
        r projected to equidistant camera
    """

    theta = math.atan(r / focal_length)

    return math.sin(theta / 2.0) * 2 * focal_length


def rd_to_r(focal_length: float, rd: float) -> float:
    """Projects a location rp, defined in an equisolid camera, to a location r defined in a rectilinear camera.

    Parameters
    ----------
    focal_lenght : float
        Focal length
    r : float
        Location defined in the rectilinear camera

    Returns
    -------
    float
        rp projected to rectilinear camera
    """

    return math.tan(2.0 * math.asin(rd / (2.0 * focal_length))) * focal_length


def points_to_equisolid(
    points: np.ndarray, image_size: Tuple, focal_length: float
) -> np.ndarray:
    """Converts points, defined in rectilinear image, to points in an equisolid camera.

    Parameters
    ----------
    points : np.ndarray
        Points [x y]^t defined in a rectilinear camera (pixels)
    image_size : Tuple
        Image size (height, width)
    focal_length : float
        Focal length

    Returns
    -------
    np.ndarray
        Points projected to an equisolid camera (pixels)
    """

    height, width = image_size[:2]

    # Convert points to unit camera
    points_unit = points.copy()
    points_unit[0, :] = (points_unit[0, :] - width / 2.0) / focal_length
    points_unit[1, :] = (points_unit[1, :] - height / 2.0) / focal_length

    # Azimuth angle of the incoming ray of light
    phi = np.arctan2(points_unit[1, :], points_unit[0, :])

    r = np.sqrt(np.square(points_unit[0, :]) + np.square(points_unit[1, :]))
    theta = np.arctan(r)
    rd = np.sin(theta / 2.0) * 2

    # Unit camera coordinates
    u_unit = rd * np.cos(phi)
    v_unit = rd * np.sin(phi)

    # Image (pixel) coordinates
    u = u_unit * focal_length + width / 2.0
    v = v_unit * focal_length + height / 2.0

    return np.vstack((u, v))


def rectilinear_image_to_equisolid(
    image: np.ndarray, focal_length: float, scale: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Converts an image captured with rectilinear camera to equidistant angle camera image. If 'scale' is True, then
    a scaling factor is calculated so that the resulting image either covers the height, or the width, of the original
    image. The smallest scaling factor out of the two (height- and width wise) is chosen.

    Parameters
    ----------
    image : np.ndarray
        Rectilinear (pinhole) camera image
    focal_length : float
        Focal length

    Returns
    -------
    np.ndarray
        Equisolid angle camera image
    float, optional
        The scaling factor to scale the image. This value is only returned if 'scale' is True
    """

    rows, cols = image.shape[0:2]

    # Image coordinates (pixels)
    u, v = np.meshgrid(
        np.arange(cols, dtype=np.float32),
        np.arange(rows, dtype=np.float32),
    )

    scaling_factor = 1.0
    if scale:
        # Calculate the scaling factor, in both horizontal and vertical directions, so that the resulting image
        # would have either the same height or width than the original image.
        point_vertical = np.array([cols / 2, 0]).reshape(2, 1)
        point_horizontal = np.array([0, rows / 2]).reshape(2, 1)
        point_vertical_equidistant = points_to_equisolid(
            points=point_vertical, image_size=(rows, cols), focal_length=focal_length
        )
        point_horizontal_equidistant = points_to_equisolid(
            points=point_horizontal, image_size=(rows, cols), focal_length=focal_length
        )
        vertical_scaling = (rows / 2.0) / (rows / 2.0 - point_vertical_equidistant[1])
        horizontal_scaling = (cols / 2.0) / (
            cols / 2.0 - point_horizontal_equidistant[0]
        )
        scaling_factor = np.min([vertical_scaling, horizontal_scaling])

    # Unit camera coordinates
    u_unit_sp = (u - cols / 2.0) / focal_length
    v_unit_sp = (v - rows / 2.0) / focal_length

    # Radial distance in the unit camera
    r_d = np.sqrt(np.square(u_unit_sp) + np.square(v_unit_sp)) / scaling_factor
    r_d[r_d > (math.pi / 2.0) - 0.1] = np.NAN
    theta = np.arcsin(r_d / 2.0) * 2.0

    # Azimuth angle of the incoming ray of light
    phi = np.arctan2(v_unit_sp, u_unit_sp)

    # Radial distance in the equisolid camera
    r = np.tan(theta)

    # Unit camera coordinates
    u_unit = r * np.cos(phi)
    v_unit = r * np.sin(phi)

    # Image (pixel) coordinates
    u = u_unit * focal_length + cols / 2.0
    v = v_unit * focal_length + rows / 2.0

    equisolid_image = cv2.remap(image, u, v, cv2.INTER_LINEAR)

    if scale:
        return (equisolid_image, scaling_factor)
    else:
        return equisolid_image


def equisolid_image_to_rectilinear(
    image: np.ndarray, focal_length: float, scaling_factor: float = 1.0
) -> np.ndarray:
    """Converts an image captured with an equisolid angle camera to rectilinear camera image.

    Parameters
    ----------
    image : np.ndarray
        Equisolid angle camera image
    focal_length : float
        Focal length
    scaling_factor : float
        A scaling factor used to up- or downscale the projected image. This should be set to the same
        value as what is obtained from the rectilinear_image_to_equisolid function.

    Returns
    -------
    np.ndarray
        Rectilinear image
    """

    rows, cols = image.shape[0:2]

    # Image coordinates (pixels)
    u, v = np.meshgrid(
        np.arange(cols, dtype=np.float32),
        np.arange(rows, dtype=np.float32),
    )

    # Unit camera coordinates
    u_unit_rect = (u - cols / 2.0) / focal_length
    v_unit_rect = (v - rows / 2.0) / focal_length

    # Radial distance in the unit camera
    r = np.sqrt(np.square(u_unit_rect) + np.square(v_unit_rect))
    theta = np.arctan(r)

    # Azimuth angle of the incoming ray of light
    phi = np.arctan2(v_unit_rect, u_unit_rect)

    # Corrected radial distance in the hemispherical camera
    r_d = np.sin(theta / 2.0) * 2.0 * scaling_factor

    # Unit camera coordinates
    u_unit = r_d * np.cos(phi)
    v_unit = r_d * np.sin(phi)

    # Image (pixel) coordinates
    u = u_unit * focal_length + cols / 2.0
    v = v_unit * focal_length + rows / 2.0

    rectilinear_image = cv2.remap(image, u, v, cv2.INTER_LINEAR)

    return rectilinear_image
