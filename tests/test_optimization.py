"""
Tests for tactica.optimization module.

These tests verify the optimization problem definition and constraints.
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


class TestOptimizationConstraints:
    """Tests for OptimizationConstraints class."""

    def test_default_constraints(self):
        """Test default constraint values."""
        from tactica.optimization.constraints import OptimizationConstraints

        constraints = OptimizationConstraints()

        assert constraints.min_coverage == 0.0
        assert constraints.placement_mask is None
        assert constraints.fixed_cameras is None

    def test_validate_shape_mismatch(self):
        """Test that validation catches shape mismatches."""
        from tactica.optimization.constraints import OptimizationConstraints

        constraints = OptimizationConstraints(
            placement_mask=np.ones((50, 50), dtype=bool)
        )

        # Should raise for different shape
        with pytest.raises(ValueError):
            constraints.validate((100, 100))

        # Should pass for matching shape
        constraints.validate((50, 50))

    def test_get_valid_placement_mask(self):
        """Test combined valid placement mask computation."""
        from tactica.optimization.constraints import OptimizationConstraints

        dem = np.zeros((50, 50), dtype=np.float32)
        dem[20:25, :] = 1e9  # Wall

        # Exclusion zone
        exclusion = np.zeros((50, 50), dtype=bool)
        exclusion[:10, :] = True  # Exclude top rows

        constraints = OptimizationConstraints(exclusion_mask=exclusion)

        valid = constraints.get_valid_placement_mask(dem, wall_threshold=1e6)

        # Wall cells should be invalid
        assert not valid[22, 25]

        # Excluded cells should be invalid
        assert not valid[5, 25]

        # Other cells should be valid
        assert valid[35, 25]


class TestCameraPlacementProblem:
    """Tests for CameraPlacementProblem class."""

    @pytest.fixture
    def simple_problem(self):
        """Create a simple problem for testing."""
        from tactica.optimization.problem import CameraPlacementProblem

        dem = np.zeros((64, 64), dtype=np.float32)
        return CameraPlacementProblem(
            dem=dem,
            num_cameras=2,
            wall_threshold=1e6,
        )

    def test_problem_creation(self, simple_problem):
        """Test basic problem creation."""
        assert simple_problem.num_cameras == 2
        assert simple_problem.dim == 12  # 2 cameras * 6 params
        assert simple_problem.bounds.shape == (12, 2)

    def test_decode_encode_roundtrip(self, simple_problem):
        """Test that encode/decode are inverses."""
        x = simple_problem.get_initial_solution(seed=42)
        cameras = simple_problem.decode_cameras(x)

        assert len(cameras) == 2

        x2 = simple_problem.encode_cameras(cameras)

        # Should be close (some difference due to terrain height)
        np.testing.assert_allclose(x[:2], x2[:2], rtol=0.1)  # x, y
        np.testing.assert_allclose(x[3:6], x2[3:6], rtol=0.1)  # yaw, pitch, fov

    def test_get_initial_solution(self, simple_problem):
        """Test initial solution generation."""
        x = simple_problem.get_initial_solution(seed=42)

        assert x.shape == (12,)

        # Should be within bounds
        lower = simple_problem.bounds[:, 0]
        upper = simple_problem.bounds[:, 1]
        assert np.all(x >= lower)
        assert np.all(x <= upper)

    def test_reproducibility(self, simple_problem):
        """Test that same seed produces same initial solution."""
        x1 = simple_problem.get_initial_solution(seed=42)
        x2 = simple_problem.get_initial_solution(seed=42)

        np.testing.assert_array_equal(x1, x2)

    def test_reset_tracking(self, simple_problem):
        """Test tracking state reset."""
        simple_problem.best_coverage = 0.5
        simple_problem.eval_count = 100

        simple_problem.reset_tracking()

        assert simple_problem.best_coverage == 0.0
        assert simple_problem.eval_count == 0
        assert len(simple_problem.trace_coverage) == 0


class TestCreatePriorityMask:
    """Tests for priority mask creation utilities."""

    def test_create_from_zones(self):
        """Test priority mask creation from zones."""
        from tactica.optimization.constraints import create_priority_mask_from_zones

        zones = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [2, 2, 2],
        ])

        priorities = create_priority_mask_from_zones(
            zones,
            {0: 1.0, 1: 2.0, 2: 0.5}
        )

        assert priorities[0, 0] == 1.0
        assert priorities[0, 2] == 2.0
        assert priorities[2, 0] == 0.5


class TestExclusionFromBuffer:
    """Tests for exclusion buffer creation."""

    def test_buffer_around_walls(self):
        """Test that buffer is created around walls."""
        from tactica.optimization.constraints import create_exclusion_from_buffer

        dem = np.zeros((50, 50), dtype=np.float32)
        dem[25, 25] = 1e9  # Single wall cell

        exclusion = create_exclusion_from_buffer(dem, buffer_cells=3)

        # Wall cell should be excluded
        assert exclusion[25, 25]

        # Cells within buffer should be excluded
        assert exclusion[24, 25]
        assert exclusion[25, 24]
        assert exclusion[26, 25]

        # Cells far from wall should not be excluded
        assert not exclusion[0, 0]
        assert not exclusion[49, 49]


class TestObjectiveFunctions:
    """Tests for objective functions."""

    @pytest.fixture
    def mock_cameras(self):
        """Create mock cameras for testing."""
        from tactica.core.sensors import Camera

        return [
            Camera(32, 32, 10, 0, -0.5, 1.57, 1.0, 50),
            Camera(32, 48, 10, np.pi, -0.5, 1.57, 1.0, 50),
        ]

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_coverage_objective(self, mock_cameras):
        """Test coverage objective computation."""
        from tactica.optimization.objectives import compute_coverage_objective

        dem = np.zeros((64, 64), dtype=np.float32)
        coverage = compute_coverage_objective(dem, mock_cameras)

        assert 0 <= coverage <= 1

    @pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )
    def test_weighted_coverage(self, mock_cameras):
        """Test weighted coverage computation."""
        from tactica.optimization.objectives import compute_weighted_coverage

        dem = np.zeros((64, 64), dtype=np.float32)
        weights = np.ones((64, 64), dtype=np.float32)
        weights[32:, :] = 2.0  # Higher priority in bottom half

        coverage = compute_weighted_coverage(dem, mock_cameras, weights)

        assert 0 <= coverage <= 1
