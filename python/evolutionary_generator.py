"""Evolutionary Minecraft house generator with smart mutations.

This module starts from a valid base house structure and evolves it through
intelligent architectural modifications rather than random block chaos.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np


class MutationType(Enum):
    """Types of architectural mutations that can be applied."""

    ADD_WINDOW = "add_window"
    REMOVE_WINDOW = "remove_window"
    ADD_COLUMN = "add_column"
    REMOVE_COLUMN = "remove_column"
    MODIFY_ROOF = "modify_roof"
    ADD_DOOR = "add_door"
    EXTEND_WALL = "extend_wall"
    REDUCE_WALL = "reduce_wall"
    ADD_BALCONY = "add_balcony"


@dataclass
class HouseConstraints:
    """Defines architectural requirements for a valid house."""

    min_floor_coverage: float = 0.9
    min_wall_height: int = 3
    max_wall_height: int = 6
    requires_roof: bool = True
    min_doors: int = 1
    max_doors: int = 2
    min_windows: int = 2
    max_windows: int = 8
    min_enclosed_ratio: float = 0.75
    max_floating_blocks: int = 10
    prefer_symmetry: bool = True

    def __str__(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"Floor: {self.min_floor_coverage*100:.0f}%+, "
            f"Walls: {self.min_wall_height}-{self.max_wall_height}, "
            f"Doors: {self.min_doors}-{self.max_doors}, "
            f"Windows: {self.min_windows}-{self.max_windows}"
        )


@dataclass
class FitnessScore:
    """Breakdown of how well a configuration meets constraints."""

    floor_score: float = 0.0
    wall_score: float = 0.0
    roof_score: float = 0.0
    door_score: float = 0.0
    window_score: float = 0.0
    enclosure_score: float = 0.0
    connectivity_score: float = 0.0
    symmetry_score: float = 0.0
    aesthetic_score: float = 0.0

    @property
    def total(self) -> float:
        """Calculate weighted total fitness score."""
        weights = {
            'floor': 2.0,
            'wall': 2.5,
            'roof': 1.5,
            'door': 2.0,
            'window': 1.5,
            'enclosure': 1.5,
            'connectivity': 2.0,
            'symmetry': 1.0,
            'aesthetic': 1.0,
        }

        score = (
            self.floor_score * weights['floor'] +
            self.wall_score * weights['wall'] +
            self.roof_score * weights['roof'] +
            self.door_score * weights['door'] +
            self.window_score * weights['window'] +
            self.enclosure_score * weights['enclosure'] +
            self.connectivity_score * weights['connectivity'] +
            self.symmetry_score * weights['symmetry'] +
            self.aesthetic_score * weights['aesthetic']
        )

        return score / sum(weights.values())

    def __str__(self) -> str:
        """Format fitness breakdown for display."""
        return (
            f"Total: {self.total:.2f} | "
            f"Floor: {self.floor_score:.2f}, "
            f"Walls: {self.wall_score:.2f}, "
            f"Roof: {self.roof_score:.2f}, "
            f"Doors: {self.door_score:.2f}, "
            f"Windows: {self.window_score:.2f}, "
            f"Enclosed: {self.enclosure_score:.2f}, "
            f"Connected: {self.connectivity_score:.2f}, "
            f"Symmetry: {self.symmetry_score:.2f}, "
            f"Aesthetic: {self.aesthetic_score:.2f}"
        )


@dataclass
class Generation:
    """Represents one iteration of the evolutionary process."""

    number: int
    grid: np.ndarray
    fitness: FitnessScore
    constraints: HouseConstraints
    parent_generation: int | None = None
    mutations_applied: list[str] = None

    def __post_init__(self):
        """Initialize mutations list if not provided."""
        if self.mutations_applied is None:
            self.mutations_applied = []

    def to_dict(self) -> dict:
        """Serialize generation data for storage."""
        return {
            'number': self.number,
            'grid': self.grid.tolist(),
            'fitness': self.fitness.total,
            'fitness_breakdown': {
                'floor': self.fitness.floor_score,
                'wall': self.fitness.wall_score,
                'roof': self.fitness.roof_score,
                'door': self.fitness.door_score,
                'window': self.fitness.window_score,
                'enclosure': self.fitness.enclosure_score,
                'connectivity': self.fitness.connectivity_score,
                'symmetry': self.fitness.symmetry_score,
                'aesthetic': self.fitness.aesthetic_score,
            },
            'parent_generation': self.parent_generation,
            'mutations_applied': self.mutations_applied,
        }

    @classmethod
    def from_dict(cls, data: dict, constraints: HouseConstraints) -> Generation:
        """Deserialize generation data from storage."""
        fitness = FitnessScore(
            floor_score=data['fitness_breakdown']['floor'],
            wall_score=data['fitness_breakdown']['wall'],
            roof_score=data['fitness_breakdown']['roof'],
            door_score=data['fitness_breakdown']['door'],
            window_score=data['fitness_breakdown']['window'],
            enclosure_score=data['fitness_breakdown']['enclosure'],
            connectivity_score=data['fitness_breakdown']['connectivity'],
            symmetry_score=data['fitness_breakdown']['symmetry'],
            aesthetic_score=data['fitness_breakdown']['aesthetic'],
        )

        return cls(
            number=data['number'],
            grid=np.array(data['grid'], dtype=np.uint8),
            fitness=fitness,
            constraints=constraints,
            parent_generation=data.get('parent_generation'),
            mutations_applied=data.get('mutations_applied', []),
        )


class EvolutionaryHouseGenerator:
    """Generates houses through iterative intelligent mutations."""

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        constraints: HouseConstraints | None = None,
    ):
        """Initialize the evolutionary generator.

        Args:
            width: X dimension of the building volume.
            height: Y dimension of the building volume.
            depth: Z dimension of the building volume.
            constraints: Optional custom constraint set.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.constraints = constraints or HouseConstraints()

        self.current_generation = 0
        self.best_generation: Generation | None = None
        self.history: list[Generation] = []

        self.checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def generate_base_house(self) -> np.ndarray:
        """Create a solid, valid starting structure.

        Returns:
            3D numpy array with a complete rectangular house.
        """
        grid = np.zeros((self.width, self.height, self.depth), dtype=np.uint8)

        # Solid floor
        grid[:, 0, :] = 1

        # Complete walls (all four sides)
        wall_height = min(self.constraints.min_wall_height + 1, self.height - 2)

        # Front and back walls
        grid[:, 1:wall_height+1, 0] = 1
        grid[:, 1:wall_height+1, self.depth-1] = 1

        # Left and right walls
        grid[0, 1:wall_height+1, :] = 1
        grid[self.width-1, 1:wall_height+1, :] = 1

        # Add one door in the middle of front wall
        door_x = self.width // 2
        grid[door_x, 1:3, 0] = 0
        if door_x > 0:
            grid[door_x-1, 1:3, 0] = 0

        # Add a few windows symmetrically
        window_y = min(3, wall_height)

        # Front wall windows
        if self.width >= 6:
            window_x1 = self.width // 4
            window_x2 = 3 * self.width // 4
            grid[window_x1, window_y, 0] = 0
            grid[window_x2, window_y, 0] = 0

        # Side walls windows
        if self.depth >= 6:
            window_z1 = self.depth // 3
            window_z2 = 2 * self.depth // 3
            grid[0, window_y, window_z1] = 0
            grid[self.width-1, window_y, window_z1] = 0

        # Flat roof
        roof_y = wall_height + 1
        if roof_y < self.height:
            grid[:, roof_y, :] = 1

        return grid

    def apply_mutation(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Apply one intelligent architectural mutation.

        Args:
            grid: Current block configuration.

        Returns:
            Tuple of (mutated grid, mutation description).
        """
        mutated = grid.copy()

        # Choose a random mutation type
        mutation_type = random.choice(list(MutationType))

        if mutation_type == MutationType.ADD_WINDOW:
            return self._add_window(mutated)
        elif mutation_type == MutationType.REMOVE_WINDOW:
            return self._remove_window(mutated)
        elif mutation_type == MutationType.ADD_COLUMN:
            return self._add_column(mutated)
        elif mutation_type == MutationType.REMOVE_COLUMN:
            return self._remove_column(mutated)
        elif mutation_type == MutationType.MODIFY_ROOF:
            return self._modify_roof(mutated)
        elif mutation_type == MutationType.ADD_DOOR:
            return self._add_door(mutated)
        elif mutation_type == MutationType.EXTEND_WALL:
            return self._extend_wall(mutated)
        elif mutation_type == MutationType.REDUCE_WALL:
            return self._reduce_wall(mutated)
        elif mutation_type == MutationType.ADD_BALCONY:
            return self._add_balcony(mutated)

        return mutated, "no_change"

    def _add_window(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Add a window to a wall."""
        # Find a solid wall block that could become a window
        candidates = []
        for y in range(2, min(5, self.height)):
            for x in range(1, self.width-1):
                for z in [0, self.depth-1]:
                    if grid[x, y, z] == 1:
                        candidates.append((x, y, z))
            for z in range(1, self.depth-1):
                for x in [0, self.width-1]:
                    if grid[x, y, z] == 1:
                        candidates.append((x, y, z))

        if candidates:
            x, y, z = random.choice(candidates)
            grid[x, y, z] = 0
            return grid, f"add_window({x},{y},{z})"

        return grid, "add_window_failed"

    def _remove_window(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Fill in a window."""
        candidates = []
        for y in range(1, min(5, self.height)):
            for x in range(self.width):
                for z in [0, self.depth-1]:
                    if grid[x, y, z] == 0 and y > 0 and grid[x, y-1, z] == 1:
                        candidates.append((x, y, z))
            for z in range(self.depth):
                for x in [0, self.width-1]:
                    if grid[x, y, z] == 0 and y > 0 and grid[x, y-1, z] == 1:
                        candidates.append((x, y, z))

        if candidates:
            x, y, z = random.choice(candidates)
            grid[x, y, z] = 1
            return grid, f"remove_window({x},{y},{z})"

        return grid, "remove_window_failed"

    def _add_column(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Add a decorative column."""
        corners = [
            (0, self.depth-1),
            (self.width-1, 0),
            (self.width-1, self.depth-1),
        ]

        if random.random() < 0.5:
            x, z = random.choice(corners)
            height = random.randint(3, min(6, self.height))
            grid[x, 1:height, z] = 1
            return grid, f"add_column({x},{z},h={height})"

        return grid, "add_column_skipped"

    def _remove_column(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Remove a decorative element."""
        # Find columns (vertical stacks on edges)
        for x in [0, self.width-1]:
            for z in [0, self.depth-1]:
                if grid[x, 2, z] == 1:
                    height = 0
                    for y in range(1, self.height):
                        if grid[x, y, z] == 1:
                            height = y
                        else:
                            break
                    if height > 0 and random.random() < 0.3:
                        grid[x, 1:height+1, z] = 0
                        return grid, f"remove_column({x},{z})"

        return grid, "remove_column_failed"

    def _modify_roof(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Change roof configuration."""
        # Find roof level
        roof_y = 0
        for y in range(self.height-1, 0, -1):
            if np.any(grid[:, y, :] > 0):
                roof_y = y
                break

        if roof_y > 2:
            # Try to make it more interesting
            if random.random() < 0.3:
                # Add a peak
                peak_x = self.width // 2
                for dz in range(self.depth // 3, 2 * self.depth // 3):
                    if roof_y + 1 < self.height:
                        grid[peak_x, roof_y+1, dz] = 1
                return grid, "roof_add_peak"
            else:
                # Remove some roof blocks for variation
                remove_count = random.randint(1, 3)
                for _ in range(remove_count):
                    x = random.randint(1, self.width-2)
                    z = random.randint(1, self.depth-2)
                    grid[x, roof_y, z] = 0
                return grid, "roof_remove_blocks"

        return grid, "modify_roof_skipped"

    def _add_door(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Add an additional entrance."""
        # Try back wall
        door_x = self.width // 2 + random.randint(-1, 1)
        door_x = max(1, min(door_x, self.width-2))

        if grid[door_x, 1, self.depth-1] == 1:
            grid[door_x, 1:3, self.depth-1] = 0
            return grid, f"add_door_back({door_x})"

        return grid, "add_door_failed"

    def _extend_wall(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Make walls taller."""
        max_wall_height = 0
        for y in range(self.height):
            if np.any(grid[:, y, 0] > 0) or np.any(grid[:, y, self.depth-1] > 0):
                max_wall_height = y

        if max_wall_height < self.constraints.max_wall_height and max_wall_height + 1 < self.height:
            new_height = max_wall_height + 1
            # Extend front and back
            grid[:, new_height, 0] = 1
            grid[:, new_height, self.depth-1] = 1
            # Extend sides
            grid[0, new_height, :] = 1
            grid[self.width-1, new_height, :] = 1
            return grid, f"extend_wall(to_y={new_height})"

        return grid, "extend_wall_failed"

    def _reduce_wall(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Make walls shorter."""
        max_wall_height = 0
        for y in range(self.height):
            if np.any(grid[:, y, 0] > 0) or np.any(grid[:, y, self.depth-1] > 0):
                max_wall_height = y

        if max_wall_height > self.constraints.min_wall_height:
            # Remove top layer
            grid[:, max_wall_height, 0] = 0
            grid[:, max_wall_height, self.depth-1] = 0
            grid[0, max_wall_height, :] = 0
            grid[self.width-1, max_wall_height, :] = 0
            return grid, f"reduce_wall(from_y={max_wall_height})"

        return grid, "reduce_wall_failed"

    def _add_balcony(self, grid: np.ndarray) -> tuple[np.ndarray, str]:
        """Add a balcony extension."""
        if random.random() < 0.3 and self.width < 18:
            # Add blocks extending from a wall
            wall_side = random.choice([0, self.depth-1])
            balcony_x = random.randint(self.width // 3, 2 * self.width // 3)
            balcony_y = random.randint(2, min(4, self.height-1))

            if wall_side == 0:
                grid[balcony_x, balcony_y, 1] = 1
            else:
                grid[balcony_x, balcony_y, self.depth-2] = 1

            return grid, f"add_balcony({balcony_x},{balcony_y})"

        return grid, "add_balcony_skipped"

    def evaluate_fitness(self, grid: np.ndarray) -> FitnessScore:
        """Calculate how well a configuration meets all constraints.

        Args:
            grid: Block configuration to evaluate.

        Returns:
            Detailed fitness breakdown.
        """
        score = FitnessScore()

        # Floor coverage
        floor = grid[:, 0, :]
        floor_coverage = np.sum(floor > 0) / floor.size
        score.floor_score = min(floor_coverage / self.constraints.min_floor_coverage, 1.0)

        # Wall structure - check height and completeness
        wall_heights = []
        perimeter_cells = 0
        solid_perimeter = 0

        for y in range(1, self.height):
            for x in range(self.width):
                for z in range(self.depth):
                    if x == 0 or x == self.width - 1 or z == 0 or z == self.depth - 1:
                        perimeter_cells += 1
                        if grid[x, y, z] > 0:
                            solid_perimeter += 1
                            wall_heights.append(y)

        avg_wall_height = max(wall_heights) if wall_heights else 0

        # Prefer walls in the target range
        if avg_wall_height < self.constraints.min_wall_height:
            score.wall_score = avg_wall_height / self.constraints.min_wall_height
        elif avg_wall_height > self.constraints.max_wall_height:
            score.wall_score = max(0, 1.0 - (avg_wall_height - self.constraints.max_wall_height) * 0.2)
        else:
            score.wall_score = 1.0

        # Roof coverage
        if self.constraints.requires_roof and avg_wall_height > 0:
            roof_y = min(self.height - 1, avg_wall_height + 1)
            if roof_y < self.height:
                roof = grid[:, roof_y, :]
                roof_coverage = np.sum(roof > 0) / roof.size
                score.roof_score = roof_coverage
        else:
            score.roof_score = 1.0

        # Door count
        door_count = self._count_doors(grid)
        if door_count < self.constraints.min_doors:
            score.door_score = door_count / self.constraints.min_doors
        elif door_count > self.constraints.max_doors:
            score.door_score = max(0, 1.0 - (door_count - self.constraints.max_doors) * 0.3)
        else:
            score.door_score = 1.0

        # Window count
        window_count = self._count_windows(grid)
        if window_count < self.constraints.min_windows:
            score.window_score = window_count / self.constraints.min_windows
        elif window_count > self.constraints.max_windows:
            score.window_score = max(0, 1.0 - (window_count - self.constraints.max_windows) * 0.1)
        else:
            score.window_score = 1.0

        # Enclosure ratio
        enclosure_ratio = solid_perimeter / perimeter_cells if perimeter_cells > 0 else 0
        score.enclosure_score = min(enclosure_ratio / self.constraints.min_enclosed_ratio, 1.0)

        # Connectivity (check for floating blocks)
        floating_blocks = self._count_floating_blocks(grid)
        score.connectivity_score = max(
            0.0,
            1.0 - (floating_blocks / max(self.constraints.max_floating_blocks, 1))
        )

        # Symmetry check (Z-axis reflection)
        if self.constraints.prefer_symmetry:
            symmetry = self._calculate_symmetry(grid)
            score.symmetry_score = symmetry
        else:
            score.symmetry_score = 1.0

        # Aesthetic score (variety, interesting features)
        score.aesthetic_score = self._calculate_aesthetics(grid, window_count, door_count)

        return score

    def _count_doors(self, grid: np.ndarray) -> int:
        """Count door openings in the walls."""
        door_count = 0

        # Check all walls at ground level
        for y in range(1, min(4, self.height)):
            # Front and back walls
            for x in range(1, self.width-1):
                if grid[x, y, 0] == 0 and grid[x, 0, 0] > 0:
                    if y == 1:  # Only count at y=1 to avoid double counting
                        door_count += 1
                if grid[x, y, self.depth-1] == 0 and grid[x, 0, self.depth-1] > 0:
                    if y == 1:
                        door_count += 1

        return door_count

    def _count_windows(self, grid: np.ndarray) -> int:
        """Count window openings in the walls."""
        window_count = 0

        for y in range(2, min(6, self.height)):
            # Front and back walls
            for x in range(self.width):
                if grid[x, y, 0] == 0 and grid[x, y-1, 0] > 0:
                    window_count += 1
                if grid[x, y, self.depth-1] == 0 and grid[x, y-1, self.depth-1] > 0:
                    window_count += 1

            # Side walls
            for z in range(self.depth):
                if grid[0, y, z] == 0 and grid[0, y-1, z] > 0:
                    window_count += 1
                if grid[self.width-1, y, z] == 0 and grid[self.width-1, y-1, z] > 0:
                    window_count += 1

        return window_count

    def _count_floating_blocks(self, grid: np.ndarray) -> int:
        """Count blocks that have no support below them."""
        floating = 0
        for y in range(1, self.height):
            for x in range(self.width):
                for z in range(self.depth):
                    if grid[x, y, z] > 0:
                        if grid[x, y - 1, z] == 0:
                            floating += 1
        return floating

    def _calculate_symmetry(self, grid: np.ndarray) -> float:
        """Calculate Z-axis symmetry score."""
        flipped = np.flip(grid, axis=2)
        matching = np.sum(grid == flipped)
        total = grid.size
        return matching / total

    def _calculate_aesthetics(self, grid: np.ndarray, window_count: int, door_count: int) -> float:
        """Evaluate aesthetic qualities."""
        score = 0.5  # Base score

        # Reward variety in features
        if window_count >= 2 and window_count <= 6:
            score += 0.2

        if door_count == 1:
            score += 0.2
        elif door_count == 2:
            score += 0.1

        # Reward vertical variety (different heights)
        max_heights = []
        for x in range(self.width):
            for z in range(self.depth):
                for y in range(self.height - 1, -1, -1):
                    if grid[x, y, z] > 0:
                        max_heights.append(y)
                        break

        if max_heights:
            height_variance = np.std(max_heights)
            if height_variance > 0.5:
                score += 0.1

        return min(score, 1.0)

    def evolve_generation(self, parent_grid: np.ndarray | None = None) -> Generation:
        """Create and evaluate one generation.

        Args:
            parent_grid: Optional parent configuration to evolve from.

        Returns:
            New generation with fitness evaluation.
        """
        mutations_applied = []

        if parent_grid is None:
            # Start from base structure
            grid = self.generate_base_house()
            parent_gen = None
            mutations_applied.append("base_house_generated")
        else:
            # Apply 1-3 mutations to parent
            grid = parent_grid.copy()
            parent_gen = self.current_generation - 1

            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                grid, mutation_desc = self.apply_mutation(grid)
                if mutation_desc != "no_change":
                    mutations_applied.append(mutation_desc)

        fitness = self.evaluate_fitness(grid)

        generation = Generation(
            number=self.current_generation,
            grid=grid,
            fitness=fitness,
            constraints=self.constraints,
            parent_generation=parent_gen,
            mutations_applied=mutations_applied,
        )

        self.current_generation += 1
        self.history.append(generation)

        # Update best if this is better
        if self.best_generation is None or fitness.total > self.best_generation.fitness.total:
            self.best_generation = generation
            self.save_checkpoint(generation)
            print(f"  🌟 New best! Fitness: {fitness.total:.3f} (mutations: {', '.join(mutations_applied)})")

        return generation

    def run_evolution(
        self,
        max_generations: int = 100,
        target_fitness: float = 0.85,
        callback=None,
    ) -> Generation:
        """Run the evolutionary process until target is reached.

        Args:
            max_generations: Maximum iterations to run.
            target_fitness: Stop when this fitness level is reached.
            callback: Optional function called after each generation.

        Returns:
            Best generation found.
        """
        # Try to load previous best
        loaded = self.load_latest_checkpoint()

        if loaded:
            parent = self.best_generation.grid
            print(f"  Continuing from checkpoint with fitness {self.best_generation.fitness.total:.3f}")
        else:
            parent = None
            print("  Starting from fresh base house")

        for i in range(max_generations):
            generation = self.evolve_generation(parent)

            if callback:
                callback(generation)

            if generation.number % 10 == 0:
                print(f"Generation {generation.number}: {generation.fitness}")

            if generation.fitness.total >= target_fitness:
                print(f"\n🎉 Target fitness reached in {i + 1} generations!")
                break

            # Use best as parent for next generation
            parent = self.best_generation.grid

        print(f"\n📊 Final stats:")
        print(f"   Best fitness: {self.best_generation.fitness.total:.3f}")
        print(f"   Mutations: {', '.join(self.best_generation.mutations_applied[-5:])}")

        return self.best_generation

    def save_checkpoint(self, generation: Generation) -> None:
        """Save a successful generation to disk.

        Args:
            generation: Generation to persist.
        """
        checkpoint_file = self.checkpoint_dir / f"gen_{generation.number:04d}.json"
        with checkpoint_file.open('w', encoding='utf-8') as f:
            json.dump(generation.to_dict(), f, indent=2)

        # Also save as "best.json" for easy loading
        best_file = self.checkpoint_dir / "best.json"
        with best_file.open('w', encoding='utf-8') as f:
            json.dump(generation.to_dict(), f, indent=2)

    def load_latest_checkpoint(self) -> bool:
        """Load the most recent successful generation.

        Returns:
            True if checkpoint was loaded successfully.
        """
        best_file = self.checkpoint_dir / "best.json"
        if not best_file.exists():
            return False

        try:
            with best_file.open('r', encoding='utf-8') as f:
                data = json.load(f)

            generation = Generation.from_dict(data, self.constraints)
            self.best_generation = generation
            self.current_generation = generation.number + 1

            print(f"📁 Loaded checkpoint from generation {generation.number}")
            print(f"   Fitness: {generation.fitness}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            return False

    def export_to_viewer_format(self, generation: Generation) -> dict:
        """Convert a generation to the frontend JSON format.

        Args:
            generation: Generation to export.

        Returns:
            Dictionary compatible with the Three.js viewer.
        """
        blocks = []
        grid = generation.grid

        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    if grid[x, y, z] > 0:
                        blocks.append({
                            'x': int(x),
                            'y': int(y),
                            'z': int(z),
                            'type': 'block',
                        })

        return {
            'blocks': blocks,
            'dimensions': {
                'width': self.width,
                'height': self.height,
                'depth': self.depth,
            },
            'metadata': {
                'generation': generation.number,
                'fitness': generation.fitness.total,
                'fitness_breakdown': {
                    'floor': generation.fitness.floor_score,
                    'wall': generation.fitness.wall_score,
                    'roof': generation.fitness.roof_score,
                    'door': generation.fitness.door_score,
                    'window': generation.fitness.window_score,
                    'enclosure': generation.fitness.enclosure_score,
                    'connectivity': generation.fitness.connectivity_score,
                    'symmetry': generation.fitness.symmetry_score,
                    'aesthetic': generation.fitness.aesthetic_score,
                },
                'mutations_applied': generation.mutations_applied,
            }
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart evolutionary house generator with intelligent mutations"
    )

    parser.add_argument('--width', type=int, default=10, help='Building width')
    parser.add_argument('--height', type=int, default=8, help='Building height')
    parser.add_argument('--depth', type=int, default=10, help='Building depth')
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Maximum generations to evolve'
    )
    parser.add_argument(
        '--target-fitness',
        type=float,
        default=0.85,
        help='Stop when this fitness is reached (0-1)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for the final house JSON'
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for evolutionary generation."""
    args = parse_args()

    print("🏠 Smart Evolutionary House Generator")
    print("=" * 60)

    constraints = HouseConstraints()
    print(f"Constraints: {constraints}")
    print()

    generator = EvolutionaryHouseGenerator(
        width=args.width,
        height=args.height,
        depth=args.depth,
        constraints=constraints,
    )

    def progress_callback(gen: Generation):
        """Display progress after each generation."""
        pass  # Main loop handles printing

    print("Starting evolution from base house structure...")
    print()

    best = generator.run_evolution(
        max_generations=args.generations,
        target_fitness=args.target_fitness,
        callback=progress_callback,
    )

    print("\n" + "=" * 60)
    print(f"🏆 Best result: Generation {best.number}")
    print(f"   {best.fitness}")
    print()

    # Export to viewer format
    output_path = args.output or Path(__file__).parent.parent / "output" / "house.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    viewer_data = generator.export_to_viewer_format(best)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(viewer_data, f, indent=2)

    print(f"💾 Exported to: {output_path}")
    print(f"   Total blocks: {len(viewer_data['blocks'])}")
    print(f"   Dimensions: {viewer_data['dimensions']}")
    print(f"   Applied mutations: {', '.join(best.mutations_applied)}")


if __name__ == '__main__':
    main()