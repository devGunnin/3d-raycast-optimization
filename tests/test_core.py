"""
Tests for tactica.core module.

These tests verify the core visibility computation and sensor models.
"""

import pytest
import numpy as np


def _cuda_available():
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TestCameraResolution:
    """Tests for CameraResolution class."""

    def test_default_values(self):
        """Test default 4K resolution."""
        from tactica.core.sensors import CameraResolution

        res = CameraResolution()
        assert res.horizontal_pixels == 3840
        assert res.vertical_pixels == 2160
        assert res.pixels_per_meter == 30.0

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        from tactica.core.sensors import CameraResolution

        res = CameraResolution(1920, 1080, 30.0)
        assert abs(res.aspect_ratio - 16/9) < 0.01

    def test_presets(self):
        """Test resolution presets."""
        from tactica.core.sensors import CameraResolution

        res_4k = CameraResolution.preset_4k()
        assert res_4k.horizontal_pixels == 3840

        res_1080 = CameraResolution.preset_1080p()
        assert res_1080.horizontal_pixels == 1920

    def test_serialization(self):
        """Test to_dict and from_dict."""
        from tactica.core.sensors import CameraResolution

        res = CameraResolution(1920, 1080, 25.0)
        data = res.to_dict()
        res2 = CameraResolution.from_dict(data)

        assert res2.horizontal_pixels == res.horizontal_pixels
        assert res2.pixels_per_meter == res.pixels_per_meter


class TestCamera:
    """Tests for Camera class."""

    def test_camera_creation(self):
        """Test basic camera creation."""
        from tactica.core.sensors import Camera

        cam = Camera(
            x=100.0, y=100.0, z=10.0,
            yaw=0.0, pitch=-0.1,
            hfov=1.57, vfov=1.0,
            max_range=100.0
        )

        assert cam.x == 100.0
        assert cam.position == (100.0, 100.0, 10.0)

    def test_to_array(self):
        """Test conversion to numpy array."""
        from tactica.core.sensors import Camera

        cam = Camera(1, 2, 3, 0.1, 0.2, 0.3, 0.4, 50.0)
        arr = cam.to_array()

        assert arr.shape == (8,)
        assert arr.dtype == np.float32
        assert arr[0] == 1.0
        assert arr[7] == 50.0

    def test_from_fov_and_resolution(self):
        """Test camera creation with auto-computed vfov and range."""
        from tactica.core.sensors import Camera, CameraResolution

        res = CameraResolution(3840, 2160, 30.0)
        cam = Camera.from_fov_and_resolution(
            x=100, y=100, z=10,
            yaw=0, pitch=0,
            hfov=np.pi/2,  # 90 degrees
            resolution=res
        )

        assert cam.max_range > 0
        assert cam.vfov > 0
        assert cam.vfov < cam.hfov  # VFOV should be smaller for landscape

    def test_serialization(self):
        """Test to_dict and from_dict."""
        from tactica.core.sensors import Camera

        cam = Camera(1, 2, 3, 0.1, 0.2, 0.3, 0.4, 50.0)
        data = cam.to_dict()
        cam2 = Camera.from_dict(data)

        assert cam2.x == cam.x
        assert cam2.max_range == cam.max_range

    def test_degree_properties(self):
        """Test degree conversion properties."""
        from tactica.core.sensors import Camera

        cam = Camera(0, 0, 0, np.pi/4, -np.pi/6, np.pi/2, np.pi/3, 100)

        assert abs(cam.yaw_degrees - 45) < 0.1
        assert abs(cam.pitch_degrees - (-30)) < 0.1
        assert abs(cam.hfov_degrees - 90) < 0.1


class TestMaxRangeComputation:
    """Tests for max range computation."""

    def test_narrower_fov_longer_range(self):
        """Narrower FOV should give longer range."""
        from tactica.core.sensors import CameraResolution, compute_max_range_from_fov

        res = CameraResolution(3840, 2160, 30.0)

        narrow_range = compute_max_range_from_fov(np.deg2rad(30), np.deg2rad(20), res)
        wide_range = compute_max_range_from_fov(np.deg2rad(120), np.deg2rad(80), res)

        assert narrow_range > wide_range

    def test_higher_ppm_shorter_range(self):
        """Higher PPM requirement should give shorter range."""
        from tactica.core.sensors import CameraResolution, compute_max_range_from_fov

        res_low = CameraResolution(3840, 2160, 20.0)
        res_high = CameraResolution(3840, 2160, 50.0)

        fov = np.deg2rad(90)
        vfov = np.deg2rad(60)

        range_low = compute_max_range_from_fov(fov, vfov, res_low)
        range_high = compute_max_range_from_fov(fov, vfov, res_high)

        assert range_low > range_high


class TestCreateRandomCameras:
    """Tests for random camera generation."""

    def test_creates_correct_count(self):
        """Test that correct number of cameras are created."""
        from tactica.core.sensors import create_random_cameras

        dem = np.zeros((100, 100), dtype=np.float32)
        cameras = create_random_cameras(5, dem, seed=42)

        assert len(cameras) == 5

    def test_respects_bounds(self):
        """Test that cameras are within DEM bounds."""
        from tactica.core.sensors import create_random_cameras

        dem = np.zeros((100, 100), dtype=np.float32)
        cameras = create_random_cameras(10, dem, margin=10.0, seed=42)

        for cam in cameras:
            assert 10 <= cam.x <= 90
            assert 10 <= cam.y <= 90

    def test_avoids_walls(self):
        """Test that cameras are not placed on walls."""
        from tactica.core.sensors import create_random_cameras

        dem = np.zeros((100, 100), dtype=np.float32)
        # Make half the DEM walls
        dem[:, 50:] = 1e9

        cameras = create_random_cameras(5, dem, wall_threshold=1e6, seed=42)

        for cam in cameras:
            assert cam.x < 50  # All cameras should be in non-wall area


class TestVisibilityComputation:
    """Tests for GPU visibility computation."""

    @pytest.fixture
    def simple_dem(self):
        """Create a simple flat DEM for testing."""
        return np.zeros((64, 64), dtype=np.float32)

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_basic_visibility(self, simple_dem):
        """Test basic visibility computation."""
        from tactica.core.visibility import compute_visibility
        from tactica.core.sensors import Camera

        # Camera in center looking down
        cam = Camera(32, 32, 10, 0, -0.5, np.pi/2, np.pi/3, 50)

        visible_any, vis_count = compute_visibility(simple_dem, [cam])

        assert visible_any.shape == simple_dem.shape
        assert vis_count.shape == simple_dem.shape
        assert visible_any.sum() > 0  # Some cells should be visible

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_walls_block_visibility(self, simple_dem):
        """Test that walls block visibility."""
        from tactica.core.visibility import compute_visibility
        from tactica.core.sensors import Camera

        # Add a wall
        dem = simple_dem.copy()
        dem[30:35, :] = 1e9  # Horizontal wall

        # Camera on one side, looking toward wall
        cam = Camera(32, 20, 10, np.pi/2, 0, np.pi/2, np.pi/3, 50)

        visible_any, vis_count = compute_visibility(dem, [cam], wall_threshold=1e6)

        # Cells behind wall should not be visible
        assert visible_any[40, 32] == 0  # Behind wall
