"""
Optimization runner for camera placement.

This module provides high-level functions for running optimization
algorithms on camera placement problems.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import time
import numpy as np

from tactica.core.sensors import Camera, CameraResolution
from tactica.optimization.problem import CameraPlacementProblem
from tactica.optimization.constraints import OptimizationConstraints


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.

    Attributes:
        cameras: Optimized camera configurations
        coverage: Final coverage fraction
        fitness: Final fitness value (negative of objective)
        num_evaluations: Number of function evaluations used
        runtime_seconds: Wall-clock time for optimization
        trace_coverage: Coverage values during optimization
        trace_x: Parameter vectors during optimization
        optimizer_name: Name of the optimizer used
        success: Whether optimization succeeded
        message: Status message
    """
    cameras: List[Camera]
    coverage: float
    fitness: float
    num_evaluations: int
    runtime_seconds: float
    trace_coverage: List[float] = field(default_factory=list)
    trace_x: List[np.ndarray] = field(default_factory=list)
    optimizer_name: str = "unknown"
    success: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cameras": [cam.to_dict() for cam in self.cameras],
            "coverage": self.coverage,
            "fitness": self.fitness,
            "num_evaluations": self.num_evaluations,
            "runtime_seconds": self.runtime_seconds,
            "optimizer_name": self.optimizer_name,
            "success": self.success,
            "message": self.message,
        }


def optimize_camera_placement(
    dem: np.ndarray,
    num_cameras: int = 4,
    optimizer: str = "cma",
    budget: int = 1000,
    camera_resolution: Optional[CameraResolution] = None,
    constraints: Optional[OptimizationConstraints] = None,
    wall_threshold: float = 1e6,
    z_bounds: tuple = (2, 15),
    fov_bounds: tuple = (np.deg2rad(30), np.deg2rad(140)),
    seed: Optional[int] = None,
    verbose: bool = True,
    **optimizer_kwargs,
) -> OptimizationResult:
    """
    Run camera placement optimization.

    This is the main entry point for optimizing camera placement on a DEM.

    Args:
        dem: Digital Elevation Model array (H, W)
        num_cameras: Number of cameras to place
        optimizer: Optimization algorithm. Options:
            - "cma": CMA-ES (default, recommended)
            - "pso": Particle Swarm Optimization
            - "de": Differential Evolution
            - "ga": Genetic Algorithm with SBX
            - "abc": Artificial Bee Colony
            - "dual": Dual Annealing
            - "pgo": Population Graph Optimization
        budget: Maximum number of function evaluations
        camera_resolution: Camera sensor configuration (default: 4K @ 30 PPM)
        constraints: Optimization constraints (placement zones, priorities, etc.)
        wall_threshold: Height above which cells are walls
        z_bounds: (min, max) camera height above ground
        fov_bounds: (min, max) horizontal FOV in radians
        seed: Random seed for reproducibility
        verbose: Print progress information
        **optimizer_kwargs: Additional arguments for the specific optimizer

    Returns:
        OptimizationResult with optimized cameras and metrics

    Example:
        >>> from tactica.dem import generate_synthetic_dem
        >>> dem = generate_synthetic_dem(256, 256, mode='hills')
        >>> result = optimize_camera_placement(dem, num_cameras=4, budget=500)
        >>> print(f"Coverage: {result.coverage:.1%}")
    """
    if camera_resolution is None:
        camera_resolution = CameraResolution(3840, 2160, 30.0)

    # Create problem
    problem = CameraPlacementProblem(
        dem=dem,
        num_cameras=num_cameras,
        camera_resolution=camera_resolution,
        wall_threshold=wall_threshold,
        constraints=constraints,
        z_bounds=z_bounds,
        fov_bounds=fov_bounds,
    )

    if verbose:
        print(f"Camera Placement Optimization")
        print(f"  DEM shape: {dem.shape}")
        print(f"  Cameras: {num_cameras}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Budget: {budget}")
        print(f"  Parameters: {problem.dim}")

    t0 = time.time()

    try:
        if optimizer == "cma":
            result = _run_cma_es(problem, budget, seed, verbose, **optimizer_kwargs)
        elif optimizer in ["pso", "de", "ga", "abc", "dual", "pgo", "nelder", "powell", "bho"]:
            result = _run_optframework(problem, optimizer, budget, seed, verbose, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        runtime = time.time() - t0

        # Get final cameras
        if problem.best_x is not None:
            cameras = problem.decode_cameras(problem.best_x)
        else:
            cameras = problem.decode_cameras(result.x)

        return OptimizationResult(
            cameras=cameras,
            coverage=problem.best_coverage,
            fitness=-result.fx,
            num_evaluations=result.nfev,
            runtime_seconds=runtime,
            trace_coverage=problem.trace_coverage.copy(),
            trace_x=[x.copy() for x in problem.trace_x],
            optimizer_name=optimizer,
            success=True,
            message="Optimization completed successfully",
        )

    except Exception as e:
        runtime = time.time() - t0
        return OptimizationResult(
            cameras=[],
            coverage=0.0,
            fitness=0.0,
            num_evaluations=problem.eval_count,
            runtime_seconds=runtime,
            optimizer_name=optimizer,
            success=False,
            message=str(e),
        )


@dataclass
class _OptResult:
    """Internal optimization result."""
    x: np.ndarray
    fx: float
    nfev: int


def _run_cma_es(
    problem: CameraPlacementProblem,
    budget: int,
    seed: Optional[int],
    verbose: bool,
    sigma0: float = 0.3,
    **kwargs,
) -> _OptResult:
    """Run CMA-ES optimization."""
    import cma

    lower, upper = problem.get_bounds_tuple()
    x0 = problem.get_initial_solution(seed)

    # Compute initial sigma
    sigma = sigma0 * np.mean(upper - lower)

    opts = {
        'bounds': [lower.tolist(), upper.tolist()],
        'maxfevals': budget,
        'verb_disp': 0,
        'verb_log': 0,
        'seed': seed if seed is not None else np.random.randint(1e6),
    }

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [problem.objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        if verbose and es.countiter % 10 == 0:
            print(f"  Gen {es.countiter}: coverage={problem.best_coverage:.1%}")

    return _OptResult(
        x=es.result.xbest,
        fx=es.result.fbest,
        nfev=es.result.evaluations,
    )


def _run_optframework(
    problem: CameraPlacementProblem,
    optimizer: str,
    budget: int,
    seed: Optional[int],
    verbose: bool,
    **kwargs,
) -> _OptResult:
    """
    Run optimization using OptimizationFramework.

    OptimizationFramework is a submodule that provides various optimization
    algorithms including PSO, DE, GA, ABC, Dual Annealing, and more.

    Args:
        problem: The camera placement problem
        optimizer: Name of the optimizer to use
        budget: Maximum function evaluations
        seed: Random seed for reproducibility
        verbose: Print progress (passed to some optimizers)
        **kwargs: Additional arguments passed to the optimizer
    """
    import sys
    from pathlib import Path

    # Add OptimizationFramework to path if needed
    of_path = Path(__file__).parent.parent.parent / 'OptimizationFramework' / 'src'
    if str(of_path) not in sys.path:
        sys.path.insert(0, str(of_path))

    try:
        from OptimizationFramework.optimizers import (
            run_pso,
            run_de_scipy,
            run_ga_sbx,
            run_abc,
            run_dual_annealing,
            run_nelder_mead,
            run_powell,
            run_pgo,
            run_bho,
        )
    except ImportError as e:
        raise ImportError(
            f"OptimizationFramework is required for '{optimizer}' optimizer. "
            f"Please ensure OptimizationFramework submodule is initialized. "
            f"Error: {e}"
        )

    runners = {
        "pso": run_pso,
        "de": run_de_scipy,
        "ga": run_ga_sbx,
        "abc": run_abc,
        "dual": run_dual_annealing,
        "nelder": run_nelder_mead,
        "powell": run_powell,
        "pgo": run_pgo,
        "bho": run_bho,
    }

    if optimizer not in runners:
        raise ValueError(f"Unknown optimizer: {optimizer}. Available: {list(runners.keys())}")

    runner = runners[optimizer]

    # Run optimization
    # Note: OptimizationFramework functions only accept (f, bounds, budget, seed)
    # Extra kwargs are ignored to avoid TypeError
    result = runner(problem.objective, problem.bounds, budget, seed=seed)

    if verbose:
        print(f"  Completed: {result.nfev} evaluations")

    return _OptResult(
        x=result.x,
        fx=result.fx,
        nfev=result.nfev,
    )


def run_comparison(
    dem: np.ndarray,
    num_cameras: int = 4,
    optimizers: Optional[List[str]] = None,
    budget: int = 1000,
    seed: int = 42,
    verbose: bool = True,
    **problem_kwargs,
) -> Dict[str, OptimizationResult]:
    """
    Run multiple optimizers and compare results.

    Args:
        dem: DEM array
        num_cameras: Number of cameras
        optimizers: List of optimizer names (default: all available)
        budget: Evaluation budget per optimizer
        seed: Random seed
        verbose: Print progress
        **problem_kwargs: Additional arguments for CameraPlacementProblem

    Returns:
        Dict mapping optimizer name to OptimizationResult
    """
    if optimizers is None:
        optimizers = ["cma", "pso", "de", "ga"]

    results = {}

    for opt_name in optimizers:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running {opt_name}")
            print(f"{'='*50}")

        result = optimize_camera_placement(
            dem=dem,
            num_cameras=num_cameras,
            optimizer=opt_name,
            budget=budget,
            seed=seed,
            verbose=verbose,
            **problem_kwargs,
        )

        results[opt_name] = result

        if verbose:
            print(f"  Final coverage: {result.coverage:.1%}")
            print(f"  Runtime: {result.runtime_seconds:.2f}s")

    # Print summary
    if verbose:
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        sorted_results = sorted(results.items(), key=lambda x: x[1].coverage, reverse=True)
        for i, (name, res) in enumerate(sorted_results, 1):
            print(f"  {i}. {name:12s} {res.coverage:6.1%}  ({res.runtime_seconds:.2f}s)")

    return results
