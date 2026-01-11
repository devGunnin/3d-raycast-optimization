#!/usr/bin/env python3
"""
Outdoor camera placement optimization using Dual Annealing.

This example demonstrates how to use the dual annealing optimizer
from OptimizationFramework for camera placement optimization.

Dual Annealing combines simulated annealing with local search minimization,
making it effective for global optimization problems with many local minima.

Usage:
    python dual_annealing_outdoor.py
    python dual_annealing_outdoor.py --cameras 6 --budget 2000
    python dual_annealing_outdoor.py --compare-optimizers
"""

import sys
import os
import argparse

# Add tactica package to path (for development)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tactica import optimize_camera_placement
from tactica.dem import generate_synthetic_dem


def run_dual_annealing_optimization(
    dem_size: int = 256,
    num_cameras: int = 4,
    budget: int = 1000,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Run dual annealing optimization for outdoor camera placement.

    Args:
        dem_size: Size of the DEM (dem_size x dem_size)
        num_cameras: Number of cameras to place
        budget: Maximum function evaluations
        seed: Random seed for reproducibility
        verbose: Print progress information
    """
    print("=" * 60)
    print("DUAL ANNEALING - OUTDOOR OPTIMIZATION")
    print("=" * 60)

    # Generate terrain
    print("\n1. Generating synthetic outdoor terrain...")
    dem = generate_synthetic_dem(dem_size, dem_size, mode='hills', seed=seed)
    print(f"   DEM shape: {dem.shape}")
    print(f"   Height range: [{dem.min():.1f}, {dem.max():.1f}]")

    # Print parameters
    print("\n2. Optimization Parameters:")
    print(f"   Budget: {budget} evaluations")
    print(f"   Cameras: {num_cameras}")
    print(f"   Seed: {seed}")

    # Run optimization
    print("\n3. Running optimization...")
    result = optimize_camera_placement(
        dem=dem,
        num_cameras=num_cameras,
        optimizer="dual",
        budget=budget,
        seed=seed,
        verbose=verbose,
    )

    # Print results
    print("\n4. Results:")
    print(f"   Coverage: {result.coverage:.1%}")
    print(f"   Evaluations: {result.num_evaluations}")
    print(f"   Runtime: {result.runtime_seconds:.2f}s")

    print("\n   Camera configurations:")
    for i, cam in enumerate(result.cameras):
        print(f"   Camera {i+1}:")
        print(f"     Position: ({cam.x:.1f}, {cam.y:.1f}, {cam.z:.1f})")
        print(f"     Yaw: {cam.yaw_degrees:.0f} deg, Pitch: {cam.pitch_degrees:.0f} deg")
        print(f"     FOV: {cam.hfov_degrees:.0f} deg x {cam.vfov_degrees:.0f} deg")

    return dem, result


def compare_optimizers(dem_size: int = 128, budget: int = 500, seed: int = 42):
    """Compare dual annealing with other optimizers."""
    print("\n" + "=" * 60)
    print("OPTIMIZER COMPARISON")
    print("=" * 60)

    dem = generate_synthetic_dem(dem_size, dem_size, mode='hills', seed=seed)

    optimizers = ["cma", "dual", "de", "pso"]
    results = {}

    for opt in optimizers:
        print(f"\nTesting {opt}...")
        try:
            result = optimize_camera_placement(
                dem=dem,
                num_cameras=4,
                optimizer=opt,
                budget=budget,
                seed=seed,
                verbose=False,
            )
            results[opt] = result
            print(f"  Coverage: {result.coverage:.1%}, Time: {result.runtime_seconds:.1f}s")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    for opt, res in sorted(results.items(), key=lambda x: -x[1].coverage):
        print(f"  {opt:8s}: {res.coverage:6.1%} ({res.runtime_seconds:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Dual Annealing camera placement optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with defaults
  python dual_annealing_outdoor.py

  # More cameras with larger budget
  python dual_annealing_outdoor.py --cameras 6 --budget 2000

  # Compare with other optimizers
  python dual_annealing_outdoor.py --compare-optimizers
        """
    )

    parser.add_argument('--dem-size', type=int, default=256,
                        help='DEM size (default: 256)')
    parser.add_argument('--cameras', type=int, default=4,
                        help='Number of cameras (default: 4)')
    parser.add_argument('--budget', type=int, default=1000,
                        help='Evaluation budget (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--compare-optimizers', action='store_true',
                        help='Compare dual annealing with other optimizers')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if args.compare_optimizers:
        compare_optimizers(args.dem_size, args.budget, args.seed)
    else:
        run_dual_annealing_optimization(
            dem_size=args.dem_size,
            num_cameras=args.cameras,
            budget=args.budget,
            seed=args.seed,
            verbose=not args.quiet,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
