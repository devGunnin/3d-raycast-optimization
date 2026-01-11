#!/usr/bin/env python3
"""
Basic usage example for the Tactica camera placement optimization engine.

This script demonstrates the core functionality of the tactica package:
1. Generating synthetic terrain
2. Setting up optimization constraints
3. Running optimization
4. Visualizing results
"""

import sys
import os

# Add tactica package to path (for development)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pathlib import Path

# Import from tactica package
from tactica import (
    optimize_camera_placement,
    TacticaConfig,
)
from tactica.dem import (
    generate_synthetic_dem,
    generate_random_dem,
    create_floorplan_dem,
)
from tactica.core import (
    Camera,
    CameraResolution,
    compute_visibility,
)
from tactica.optimization import (
    CameraPlacementProblem,
    OptimizationConstraints,
)
from tactica.visualization import (
    plot_camera_configuration,
    create_optimization_gif,
)


def example_outdoor_optimization():
    """Example: Optimize camera placement on outdoor terrain."""
    print("=" * 60)
    print("OUTDOOR TERRAIN OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Generate synthetic terrain
    print("\n1. Generating synthetic terrain...")
    dem = generate_synthetic_dem(256, 256, mode='hills', seed=42)
    print(f"   DEM shape: {dem.shape}")
    print(f"   Height range: [{dem.min():.1f}, {dem.max():.1f}]")

    # Run optimization
    print("\n2. Running optimization...")
    result = optimize_camera_placement(
        dem=dem,
        num_cameras=4,
        optimizer="cma",
        budget=500,
        seed=42,
        verbose=True,
    )

    # Print results
    print("\n3. Results:")
    print(f"   Coverage: {result.coverage:.1%}")
    print(f"   Runtime: {result.runtime_seconds:.2f}s")
    print(f"   Evaluations: {result.num_evaluations}")

    print("\n   Camera configurations:")
    for i, cam in enumerate(result.cameras):
        print(f"   Camera {i+1}:")
        print(f"     Position: ({cam.x:.1f}, {cam.y:.1f}, {cam.z:.1f})")
        print(f"     Yaw: {cam.yaw_degrees:.0f}°, Pitch: {cam.pitch_degrees:.0f}°")
        print(f"     FOV: {cam.hfov_degrees:.0f}° x {cam.vfov_degrees:.0f}°")
        print(f"     Max Range: {cam.max_range:.1f}")

    return dem, result


def example_indoor_optimization():
    """Example: Optimize camera placement in indoor floorplan."""
    print("\n" + "=" * 60)
    print("INDOOR FLOORPLAN OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Generate floorplan
    print("\n1. Generating indoor floorplan...")
    dem = create_floorplan_dem(256, 256, seed=42)
    wall_count = (dem >= 1e6).sum()
    print(f"   DEM shape: {dem.shape}")
    print(f"   Wall cells: {wall_count}")

    # Run optimization with indoor settings
    print("\n2. Running optimization...")
    result = optimize_camera_placement(
        dem=dem,
        num_cameras=6,
        optimizer="cma",
        budget=500,
        wall_threshold=1e6,
        z_bounds=(2, 6),  # Lower ceiling
        fov_bounds=(np.deg2rad(60), np.deg2rad(120)),  # Wide FOV for indoor
        seed=42,
        verbose=True,
    )

    print("\n3. Results:")
    print(f"   Coverage: {result.coverage:.1%}")
    print(f"   Runtime: {result.runtime_seconds:.2f}s")

    return dem, result


def example_with_constraints():
    """Example: Optimization with placement constraints."""
    print("\n" + "=" * 60)
    print("CONSTRAINED OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Generate terrain
    print("\n1. Generating terrain with priority zones...")
    dem = generate_synthetic_dem(256, 256, mode='hills', seed=42)

    # Create priority weights (higher priority in center)
    y, x = np.mgrid[0:256, 0:256]
    center = 128
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    priority_weights = 1 + 2 * np.exp(-distance**2 / (2 * 50**2))
    priority_weights = priority_weights.astype(np.float32)

    print(f"   Priority range: [{priority_weights.min():.1f}, {priority_weights.max():.1f}]")

    # Create constraints
    constraints = OptimizationConstraints(
        priority_weights=priority_weights,
        min_coverage=0.5,  # Require at least 50% coverage
    )

    # Run optimization
    print("\n2. Running constrained optimization...")
    result = optimize_camera_placement(
        dem=dem,
        num_cameras=4,
        constraints=constraints,
        budget=500,
        seed=42,
        verbose=True,
    )

    print("\n3. Results:")
    print(f"   Coverage: {result.coverage:.1%}")

    return dem, result


def example_compare_optimizers():
    """Example: Compare different optimization algorithms."""
    print("\n" + "=" * 60)
    print("OPTIMIZER COMPARISON EXAMPLE")
    print("=" * 60)

    # Generate terrain
    dem = generate_random_dem(128, 128, seed=42)

    optimizers = ["cma", "pso", "de"]
    results = {}

    for opt in optimizers:
        print(f"\nRunning {opt}...")
        try:
            result = optimize_camera_placement(
                dem=dem,
                num_cameras=3,
                optimizer=opt,
                budget=300,
                seed=42,
                verbose=False,
            )
            results[opt] = result
            print(f"  Coverage: {result.coverage:.1%}, Time: {result.runtime_seconds:.2f}s")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    for name, res in sorted(results.items(), key=lambda x: -x[1].coverage):
        print(f"{name:12s}: {res.coverage:6.1%} ({res.runtime_seconds:.2f}s)")


def main():
    """Run all examples."""
    print("TACTICA CAMERA PLACEMENT OPTIMIZATION EXAMPLES")
    print("=" * 60)

    # Example 1: Outdoor
    dem1, result1 = example_outdoor_optimization()

    # Example 2: Indoor
    dem2, result2 = example_indoor_optimization()

    # Example 3: Constraints
    dem3, result3 = example_with_constraints()

    # Example 4: Comparison (only if OptimizationFramework available)
    try:
        example_compare_optimizers()
    except ImportError:
        print("\nSkipping optimizer comparison (OptimizationFramework not installed)")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
