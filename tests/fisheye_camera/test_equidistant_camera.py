import pytest
from python_camera_library.fisheye_camera import equidistant


def test_equidistant_camera():
    focal_length = 10
    r = 10
    # Test projecting distance r (defined in rectilinear camera) to rd (defined in equidistant camera) and back to r
    assert r == pytest.approx(
        equidistant.rd_to_r(
            focal_length=focal_length,
            rd=equidistant.r_to_rd(focal_length=focal_length, r=r),
        ),
        0.000001,
    )
