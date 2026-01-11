"""
Tests for tactica.dem module.

These tests verify DEM generation, loading, and coordinate handling.
"""

import pytest
import numpy as np


class TestSyntheticDEM:
    """Tests for synthetic DEM generation."""

    def test_generate_flat(self):
        """Test flat terrain generation."""
        from tactica.dem.synthetic import generate_synthetic_dem

        dem = generate_synthetic_dem(100, 100, mode='flat')

        assert dem.shape == (100, 100)
        assert dem.dtype == np.float32
        assert np.allclose(dem, 0)

    def test_generate_hills(self):
        """Test hills terrain generation."""
        from tactica.dem.synthetic import generate_synthetic_dem

        dem = generate_synthetic_dem(100, 100, mode='hills', seed=42)

        assert dem.shape == (100, 100)
        assert dem.max() > dem.min()  # Should have variation

    def test_reproducibility(self):
        """Test that seed produces reproducible results."""
        from tactica.dem.synthetic import generate_synthetic_dem

        dem1 = generate_synthetic_dem(100, 100, mode='hills', seed=42)
        dem2 = generate_synthetic_dem(100, 100, mode='hills', seed=42)

        np.testing.assert_array_equal(dem1, dem2)

    def test_unknown_mode_raises(self):
        """Test that unknown mode raises ValueError."""
        from tactica.dem.synthetic import generate_synthetic_dem

        with pytest.raises(ValueError):
            generate_synthetic_dem(100, 100, mode='unknown')


class TestRandomDEM:
    """Tests for random DEM generation."""

    def test_generate_random(self):
        """Test random terrain generation."""
        from tactica.dem.synthetic import generate_random_dem

        dem = generate_random_dem(128, 128, seed=42)

        assert dem.shape == (128, 128)
        assert dem.dtype == np.float32
        assert dem.min() >= 0

    def test_height_range(self):
        """Test that height range is respected."""
        from tactica.dem.synthetic import generate_random_dem

        dem = generate_random_dem(128, 128, height_range=30.0, seed=42)

        assert dem.max() <= 30.0
        assert dem.min() >= 0


class TestFloorplanDEM:
    """Tests for indoor floorplan generation."""

    def test_create_floorplan(self):
        """Test floorplan creation."""
        from tactica.dem.synthetic import create_floorplan_dem

        dem = create_floorplan_dem(256, 256, seed=42)

        assert dem.shape == (256, 256)

        # Should have walls (high values)
        wall_count = (dem >= 1e6).sum()
        assert wall_count > 0

        # Should have floor space (low values)
        floor_count = (dem < 1e6).sum()
        assert floor_count > wall_count  # More floor than walls


class TestWallFunctions:
    """Tests for wall manipulation functions."""

    def test_add_walls_to_dem(self):
        """Test adding individual wall cells."""
        from tactica.dem.synthetic import add_walls_to_dem

        dem = np.zeros((50, 50), dtype=np.float32)
        wall_cells = [(10, 10), (20, 20), (30, 30)]

        dem_with_walls = add_walls_to_dem(dem, wall_cells, wall_height=1e9)

        assert dem_with_walls[10, 10] == 1e9
        assert dem_with_walls[20, 20] == 1e9
        assert dem_with_walls[0, 0] == 0  # Unchanged

    def test_add_wall_lines(self):
        """Test adding wall lines."""
        from tactica.dem.synthetic import add_wall_lines

        dem = np.zeros((50, 50), dtype=np.float32)
        lines = [((10, 10), (10, 40))]  # Horizontal line

        dem_with_lines = add_wall_lines(dem, lines, wall_height=1e9, thickness=1)

        # Line should exist
        assert dem_with_lines[10, 25] == 1e9
        # Off the line should be empty
        assert dem_with_lines[20, 25] == 0


class TestDEMMetadata:
    """Tests for DEMMetadata class."""

    def test_basic_metadata(self):
        """Test basic metadata creation."""
        from tactica.dem.loader import DEMMetadata

        meta = DEMMetadata(
            crs="EPSG:32610",
            bounds=(0, 0, 1000, 1000),
            resolution=(1.0, 1.0),
        )

        assert meta.crs == "EPSG:32610"
        assert meta.cell_size == 1.0
        assert meta.width_meters == 1000

    def test_serialization(self):
        """Test to_dict and from_dict."""
        from tactica.dem.loader import DEMMetadata

        meta = DEMMetadata(
            crs="EPSG:4326",
            bounds=(-122, 37, -121, 38),
            resolution=(0.001, 0.001),
        )

        data = meta.to_dict()
        meta2 = DEMMetadata.from_dict(data)

        assert meta2.crs == meta.crs
        assert meta2.bounds == meta.bounds


class TestCoordinates:
    """Tests for coordinate transformations."""

    def test_grid_to_world(self):
        """Test grid to world coordinate conversion."""
        from tactica.dem.loader import DEMMetadata
        from tactica.dem.coordinates import grid_to_world

        meta = DEMMetadata(
            bounds=(0, 0, 100, 100),
            resolution=(1.0, 1.0),
        )

        # Center of grid should map to center of world
        x, y = grid_to_world(50, 50, meta)
        assert abs(x - 50.5) < 0.1
        assert abs(y - 49.5) < 0.1  # Note: y is inverted in grids

    def test_world_to_grid(self):
        """Test world to grid coordinate conversion."""
        from tactica.dem.loader import DEMMetadata
        from tactica.dem.coordinates import world_to_grid

        meta = DEMMetadata(
            bounds=(0, 0, 100, 100),
            resolution=(1.0, 1.0),
        )

        row, col = world_to_grid(50, 50, meta)
        assert abs(row - 49.5) < 0.1
        assert abs(col - 49.5) < 0.1

    def test_roundtrip_coordinates(self):
        """Test that grid->world->grid is consistent."""
        from tactica.dem.loader import DEMMetadata
        from tactica.dem.coordinates import grid_to_world, world_to_grid

        meta = DEMMetadata(
            bounds=(0, 0, 256, 256),
            resolution=(1.0, 1.0),
        )

        # Original grid coordinates
        orig_row, orig_col = 100.0, 150.0

        # Convert to world and back
        x, y = grid_to_world(orig_row, orig_col, meta)
        new_row, new_col = world_to_grid(x, y, meta)

        assert abs(new_row - orig_row) < 0.01
        assert abs(new_col - orig_col) < 0.01
