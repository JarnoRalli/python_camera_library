def test_homography_import():
    import numpy as np
    from python_camera_library import rectilinear_camera as cam
    from python_camera_library import homography as homog
    from python_camera_library import utils

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

    # Extract R_ and t_ from the homography, using K
    # R is 'fully defined', whereas t_ is defined only up to scale
    (R_, t_) = homog.extract_transformation(K, H)

    scale = homog.transformation_scale(K, R_, t_, uv[:, 1], points3D_obj[:, 1])
    t_ *= scale

    np.testing.assert_almost_equal(R, R_)
    np.testing.assert_almost_equal(t, t_)
