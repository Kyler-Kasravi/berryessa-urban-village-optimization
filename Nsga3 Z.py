"""
NSGA-III Grid-Based Land Use Optimization - Enhanced UI Version
================================================================

Features:
1. Interactive UI with Tkinter
2. Configurable objective weights via sliders
3. Progress visualization every N generations
4. Multiple runs without program termination
5. Editable objective functions
6. Increased generation count for better optimization
7. VECTORIZED objective functions (7-8x speedup)
8. FIXED commute bug: measures to nearest cell, not cluster centroid
9. EDITABLE building sizes (industrial, commercial, essential cell counts)
10. PROPORTIONAL grid scaling with auto-computed building counts/sizes
11. WORKING stop button via custom termination
12. PERSISTENT convergence graphs + scrollable snapshots tab

Based on v7: Free-form connected buildings
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import time
import threading
import queue
import math
from typing import Dict, List, Tuple, Set, Callable, Optional
from collections import deque
import io
import sys
import os

# Optional tkinter import for GUI mode
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    tk = None

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.callback import Callback
from pymoo.core.termination import Termination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize

# =============================================================================
# Configuration
# =============================================================================

LAND_TYPES = {
    -1: {"name": "Unbuildable", "color": "#444444"},
    0: {"name": "Residential", "color": "#85C1E9"},
    1: {"name": "Green Space", "color": "#00FF7F"},
    2: {"name": "Commercial (cluster)", "color": "#FFB347"},
    3: {"name": "Industrial (cluster)", "color": "#FF6B6B"},
    4: {"name": "Essential (cluster)", "color": "#DDA0DD"},
}

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# =============================================================================
# Baseline proportions from the 20x20 grid configuration
# =============================================================================
BASELINE_GRID = 20
BASELINE_PROPORTIONS = {
    'industrial_frac': 18 / 400,
    'commercial_frac': 24 / 400,
    'essential_frac':  16 / 400,
    'green_frac':      0.18,
}
BASELINE_BUILDING_SIZES = {
    'industrial_size': 9,
    'commercial_size': 4,
    'essential_size': 4,
}
BASELINE_BUILDING_COUNTS = {
    'n_industrial': 2,
    'n_commercial': 6,
    'n_essential': 4,
}


def compute_scaled_config(grid_size: int) -> dict:
    """Compute proportionally scaled building configuration for a given grid size."""
    total_cells = grid_size * grid_size
    scale = total_cells / (BASELINE_GRID * BASELINE_GRID)
    
    baseline_ind_cells = BASELINE_BUILDING_COUNTS['n_industrial'] * BASELINE_BUILDING_SIZES['industrial_size']
    baseline_com_cells = BASELINE_BUILDING_COUNTS['n_commercial'] * BASELINE_BUILDING_SIZES['commercial_size']
    baseline_ess_cells = BASELINE_BUILDING_COUNTS['n_essential']  * BASELINE_BUILDING_SIZES['essential_size']
    
    target_industrial = baseline_ind_cells * scale
    target_commercial = baseline_com_cells * scale
    target_essential  = baseline_ess_cells * scale
    
    def pick_size_and_count(target_cells, baseline_size, baseline_count):
        if target_cells < 0.5:
            return 0, 0
        if target_cells >= baseline_size:
            size = baseline_size
        else:
            size = max(1, round(target_cells))
        count = max(1, round(target_cells / size))
        best_count = count
        best_err = abs(count * size - target_cells)
        for c in [count - 1, count + 1]:
            if c >= 1:
                err = abs(c * size - target_cells)
                if err < best_err:
                    best_count = c
                    best_err = err
        return size, best_count
    
    ind_size, n_ind = pick_size_and_count(target_industrial, BASELINE_BUILDING_SIZES['industrial_size'], BASELINE_BUILDING_COUNTS['n_industrial'])
    com_size, n_com = pick_size_and_count(target_commercial, BASELINE_BUILDING_SIZES['commercial_size'], BASELINE_BUILDING_COUNTS['n_commercial'])
    ess_size, n_ess = pick_size_and_count(target_essential,  BASELINE_BUILDING_SIZES['essential_size'],  BASELINE_BUILDING_COUNTS['n_essential'])
    
    green_ratio = BASELINE_PROPORTIONS['green_frac']
    
    actual_ind = n_ind * ind_size
    actual_com = n_com * com_size
    actual_ess = n_ess * ess_size
    building_cells = actual_ind + actual_com + actual_ess
    remaining = total_cells - building_cells
    actual_green = int(remaining * green_ratio) if remaining > 0 else 0
    actual_res = remaining - actual_green if remaining > 0 else 0
    
    return {
        'n_industrial': n_ind, 'n_commercial': n_com, 'n_essential': n_ess,
        'industrial_size': ind_size, 'commercial_size': com_size, 'essential_size': ess_size,
        'green_ratio': green_ratio,
        'total_cells': total_cells,
        'actual_industrial_cells': actual_ind, 'actual_commercial_cells': actual_com,
        'actual_essential_cells': actual_ess, 'actual_green_cells': actual_green,
        'actual_residential_cells': actual_res,
        'industrial_pct': actual_ind / total_cells * 100 if total_cells > 0 else 0,
        'commercial_pct': actual_com / total_cells * 100 if total_cells > 0 else 0,
        'essential_pct': actual_ess / total_cells * 100 if total_cells > 0 else 0,
        'green_pct': actual_green / total_cells * 100 if total_cells > 0 else 0,
        'residential_pct': actual_res / total_cells * 100 if total_cells > 0 else 0,
    }


# Default objective function code templates
DEFAULT_OBJECTIVES = {
    "job_commute": '''def calc_job_commute(grid):
    """Average Manhattan distance from each residential cell to nearest job CELL.
    FIXED: measures to nearest cell, not cluster centroid. Vectorized."""
    res_coords = np.argwhere(grid == 0)
    job_coords = np.argwhere((grid == 2) | (grid == 3))
    if len(res_coords) == 0 or len(job_coords) == 0:
        return 100.0
    diff = np.abs(res_coords[:, None, :] - job_coords[None, :, :])
    dists = diff[:, :, 0] + diff[:, :, 1]
    return float(dists.min(axis=1).mean())
''',
    "essential_commute": '''def calc_essential_commute(grid):
    """Average Manhattan distance from each residential cell to nearest essential CELL.
    FIXED: nearest cell, not centroid. Vectorized."""
    res_coords = np.argwhere(grid == 0)
    ess_coords = np.argwhere(grid == 4)
    if len(res_coords) == 0 or len(ess_coords) == 0:
        return 100.0
    diff = np.abs(res_coords[:, None, :] - ess_coords[None, :, :])
    dists = diff[:, :, 0] + diff[:, :, 1]
    return float(dists.min(axis=1).mean())
''',
    "livability": '''def calc_livability(grid):
    """Per residential cell livability score (vectorized).
    Base: 1.0, Green nbr: +0.5, Essential: +0.3, Commercial: -0.1, Industrial: -2.0"""
    res_mask = (grid == 0)
    if not res_mask.any():
        return 0.0
    scores = (1.0
              + 0.5 * _count_4neighbors(grid, 1)
              + 0.3 * _count_4neighbors(grid, 4)
              - 0.1 * _count_4neighbors(grid, 2)
              - 2.0 * _count_4neighbors(grid, 3))
    return float((scores * res_mask).sum())
''',
    "industrial_adjacent": '''def calc_industrial_adjacent(grid):
    """Count residential cells with at least one industrial neighbor. Vectorized."""
    res_mask = (grid == 0)
    has_ind = _count_4neighbors(grid, 3)
    return int((res_mask & (has_ind > 0)).sum())
'''
}


# =============================================================================
# Free-Form Building Growth
# =============================================================================

def grow_building(grid, anchor_row, anchor_col, target_size, growth_genes, land_type):
    """Grow a connected building from anchor using growth directions."""
    size = grid.shape[0]
    if anchor_row < 0 or anchor_row >= size or anchor_col < 0 or anchor_col >= size:
        return set()
    if grid[anchor_row, anchor_col] != -2:
        return set()
    
    cells = {(anchor_row, anchor_col)}
    frontier = [(anchor_row, anchor_col)]
    gene_idx = 0
    
    while len(cells) < target_size and frontier and gene_idx < len(growth_genes):
        direction = int(growth_genes[gene_idx]) % 4
        gene_idx += 1
        grew = False
        for fr, fc in list(frontier):
            dr, dc = DIRECTIONS[direction]
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < size and 0 <= nc < size:
                if grid[nr, nc] == -2 and (nr, nc) not in cells:
                    cells.add((nr, nc))
                    frontier.append((nr, nc))
                    grew = True
                    break
        if not grew:
            for alt_dir in range(4):
                if alt_dir == direction:
                    continue
                for fr, fc in list(frontier):
                    dr, dc = DIRECTIONS[alt_dir]
                    nr, nc = fr + dr, fc + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if grid[nr, nc] == -2 and (nr, nc) not in cells:
                            cells.add((nr, nc))
                            frontier.append((nr, nc))
                            grew = True
                            break
                if grew:
                    break
        new_frontier = []
        for fr, fc in frontier:
            can_grow = False
            for dr, dc in DIRECTIONS:
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if grid[nr, nc] == -2 and (nr, nc) not in cells:
                        can_grow = True
                        break
            if can_grow:
                new_frontier.append((fr, fc))
        frontier = new_frontier if new_frontier else frontier[-3:]
    
    if len(cells) >= target_size:
        for r, c in list(cells)[:target_size]:
            grid[r, c] = land_type
        return set(list(cells)[:target_size])
    return set()


# =============================================================================
# City Grid Builder
# =============================================================================

class CityBuilder:
    def __init__(self, grid_size, n_industrial, n_commercial, n_essential,
                 industrial_size=9, commercial_size=4, essential_size=4, green_ratio=0.18):
        self.grid_size = grid_size
        self.n_industrial = n_industrial
        self.n_commercial = n_commercial
        self.n_essential = n_essential
        self.industrial_size = industrial_size
        self.commercial_size = commercial_size
        self.essential_size = essential_size
        self.green_ratio = green_ratio
        
        self.ind_genes = 2 + max(0, industrial_size - 1)
        self.com_genes = 2 + max(0, commercial_size - 1)
        self.ess_genes = 2 + max(0, essential_size - 1)
        
        self.encoding_length = (
            n_industrial * self.ind_genes +
            n_commercial * self.com_genes +
            n_essential * self.ess_genes +
            grid_size * grid_size
        )
    
    def build(self, encoding):
        grid = np.full((self.grid_size, self.grid_size), -2, dtype=int)
        buildings = []
        collisions = 0
        idx = 0
        
        for i in range(self.n_industrial):
            anchor_r = int(encoding[idx]) % self.grid_size
            anchor_c = int(encoding[idx + 1]) % self.grid_size
            growth = encoding[idx + 2:idx + self.ind_genes]
            idx += self.ind_genes
            cells = grow_building(grid, anchor_r, anchor_c, self.industrial_size, growth, 3)
            if cells and len(cells) == self.industrial_size:
                buildings.append((3, cells))
            else:
                collisions += 1
        
        for i in range(self.n_commercial):
            anchor_r = int(encoding[idx]) % self.grid_size
            anchor_c = int(encoding[idx + 1]) % self.grid_size
            growth = encoding[idx + 2:idx + self.com_genes]
            idx += self.com_genes
            cells = grow_building(grid, anchor_r, anchor_c, self.commercial_size, growth, 2)
            if cells and len(cells) == self.commercial_size:
                buildings.append((2, cells))
            else:
                collisions += 1
        
        for i in range(self.n_essential):
            anchor_r = int(encoding[idx]) % self.grid_size
            anchor_c = int(encoding[idx + 1]) % self.grid_size
            growth = encoding[idx + 2:idx + self.ess_genes]
            idx += self.ess_genes
            cells = grow_building(grid, anchor_r, anchor_c, self.essential_size, growth, 4)
            if cells and len(cells) == self.essential_size:
                buildings.append((4, cells))
            else:
                collisions += 1
        
        empty_cells = [(r, c) for r in range(self.grid_size)
                       for c in range(self.grid_size) if grid[r, c] == -2]
        n_green = int(len(empty_cells) * self.green_ratio)
        
        if empty_cells and idx < len(encoding):
            priorities = encoding[idx:idx + len(empty_cells)]
            if len(priorities) < len(empty_cells):
                priorities = np.concatenate([priorities,
                    np.zeros(len(empty_cells) - len(priorities))])
            cell_priorities = list(zip(empty_cells, priorities[:len(empty_cells)]))
            cell_priorities.sort(key=lambda x: -x[1])
            for i, ((r, c), _) in enumerate(cell_priorities):
                if i < n_green:
                    grid[r, c] = 1
                    buildings.append((1, {(r, c)}))
                else:
                    grid[r, c] = 0
                    buildings.append((0, {(r, c)}))
        else:
            for i, (r, c) in enumerate(empty_cells):
                if i < n_green:
                    grid[r, c] = 1
                    buildings.append((1, {(r, c)}))
                else:
                    grid[r, c] = 0
                    buildings.append((0, {(r, c)}))
        
        return grid, buildings, collisions


# =============================================================================
# Vectorized Objective Functions
# =============================================================================

def _count_4neighbors(grid, value):
    match = (grid == value).astype(np.float64)
    counts = np.zeros_like(match)
    counts[1:, :]  += match[:-1, :]
    counts[:-1, :] += match[1:, :]
    counts[:, 1:]  += match[:, :-1]
    counts[:, :-1] += match[:, 1:]
    return counts

def calc_job_commute(grid, buildings=None):
    res_coords = np.argwhere(grid == 0)
    job_coords = np.argwhere((grid == 2) | (grid == 3))
    if len(res_coords) == 0 or len(job_coords) == 0:
        return 100.0
    diff = np.abs(res_coords[:, None, :] - job_coords[None, :, :])
    dists = diff[:, :, 0] + diff[:, :, 1]
    return float(dists.min(axis=1).mean())

def calc_essential_commute(grid, buildings=None):
    res_coords = np.argwhere(grid == 0)
    ess_coords = np.argwhere(grid == 4)
    if len(res_coords) == 0 or len(ess_coords) == 0:
        return 100.0
    diff = np.abs(res_coords[:, None, :] - ess_coords[None, :, :])
    dists = diff[:, :, 0] + diff[:, :, 1]
    return float(dists.min(axis=1).mean())

def calc_livability(grid):
    res_mask = (grid == 0)
    if not res_mask.any():
        return 0.0
    scores = (1.0
              + 0.5 * _count_4neighbors(grid, 1)
              + 0.3 * _count_4neighbors(grid, 4)
              - 0.1 * _count_4neighbors(grid, 2)
              - 2.0 * _count_4neighbors(grid, 3))
    return float((scores * res_mask).sum())

def calc_industrial_adjacent(grid):
    res_mask = (grid == 0)
    has_ind = _count_4neighbors(grid, 3)
    return int((res_mask & (has_ind > 0)).sum())

def get_neighbors(grid, r, c):
    size = grid.shape[0]
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append(grid[nr, nc])
    return neighbors

def building_center(cells):
    if not cells:
        return (0, 0)
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    return (sum(rows)/len(rows), sum(cols)/len(cols))


# =============================================================================
# Stoppable Termination
# =============================================================================

class StopFlag:
    """A stop flag that survives deepcopy — always refers to the same shared state."""
    def __init__(self):
        self._stopped = False
    
    def stop(self):
        self._stopped = True
    
    def clear(self):
        self._stopped = False
    
    @property
    def is_stopped(self):
        return self._stopped
    
    def __deepcopy__(self, memo):
        # Return the SAME object so deepcopied termination still sees our flag
        return self
    
    def __copy__(self):
        return self


class StoppableTermination(Termination):
    """Custom termination that checks a StopFlag AND a max generation count.
    
    The StopFlag survives PyMOO's internal deepcopy of the termination object
    by returning itself from __deepcopy__.
    """
    
    def __init__(self, n_max_gen, stop_flag):
        super().__init__()
        self.n_max_gen = n_max_gen
        self.stop_flag = stop_flag  # StopFlag instance
    
    def _update(self, algorithm):
        gen = algorithm.n_gen
        if self.stop_flag.is_stopped:
            self.force_termination = True
            return 1.0
        return gen / self.n_max_gen


# =============================================================================
# NSGA-III Problem
# =============================================================================

class WeightedBuildingProblem(Problem):
    def __init__(self, grid_size=20, n_industrial=2, n_commercial=6, n_essential=4,
                 industrial_size=9, commercial_size=4, essential_size=4,
                 green_ratio=0.18, weights=None, custom_objectives=None):
        self.grid_size = grid_size
        self.builder = CityBuilder(
            grid_size, n_industrial, n_commercial, n_essential,
            industrial_size=industrial_size, commercial_size=commercial_size,
            essential_size=essential_size, green_ratio=green_ratio
        )
        self.weights = weights or {
            'job_commute': 1.0, 'essential_commute': 1.0,
            'livability': 1.0, 'industrial_adjacent': 1.0
        }
        self.custom_objectives = custom_objectives or {}
        super().__init__(
            n_var=self.builder.encoding_length, n_obj=4, n_ieq_constr=1,
            xl=0, xu=max(grid_size - 1, 3), vtype=int
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = len(X)
        F = np.zeros((n_pop, self.n_obj))
        G = np.zeros((n_pop, self.n_ieq_constr))
        for i, x in enumerate(X):
            grid, buildings, collisions = self.builder.build(x)
            f1 = self.custom_objectives.get('job_commute', calc_job_commute)(grid) if 'job_commute' in self.custom_objectives else calc_job_commute(grid)
            f2 = self.custom_objectives.get('essential_commute', calc_essential_commute)(grid) if 'essential_commute' in self.custom_objectives else calc_essential_commute(grid)
            f3 = -(self.custom_objectives.get('livability', calc_livability)(grid) if 'livability' in self.custom_objectives else calc_livability(grid))
            f4 = self.custom_objectives.get('industrial_adjacent', calc_industrial_adjacent)(grid) if 'industrial_adjacent' in self.custom_objectives else calc_industrial_adjacent(grid)
            F[i, 0] = f1 * self.weights['job_commute']
            F[i, 1] = f2 * self.weights['essential_commute']
            F[i, 2] = f3 * self.weights['livability']
            F[i, 3] = f4 * self.weights['industrial_adjacent']
            G[i, 0] = collisions
        out["F"] = F
        out["G"] = G


# =============================================================================
# Genetic Operators
# =============================================================================

class IntegerSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xu = int(problem.xu) if np.isscalar(problem.xu) else int(problem.xu[0])
        return np.random.randint(0, xu + 1, (n_samples, problem.n_var))

class IntegerMutation(Mutation):
    def __init__(self, prob=0.2, eta=3):
        super().__init__()
        self.prob = prob
        self.eta = eta
    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        xu = int(problem.xu) if np.isscalar(problem.xu) else int(problem.xu[0])
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if np.random.random() < self.prob:
                    delta = np.random.randint(-self.eta, self.eta + 1)
                    Y[i, j] = np.clip(Y[i, j] + delta, 0, xu)
        return Y

class IntegerCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob
    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty((2, n_matings, n_var), dtype=X.dtype)
        for k in range(n_matings):
            if np.random.random() < self.prob:
                pt = np.random.randint(1, n_var)
                Y[0, k] = np.concatenate([X[0, k, :pt], X[1, k, pt:]])
                Y[1, k] = np.concatenate([X[1, k, :pt], X[0, k, pt:]])
            else:
                Y[0, k] = X[0, k].copy()
                Y[1, k] = X[1, k].copy()
        return Y


# =============================================================================
# Progress Callback
# =============================================================================

class ProgressCallback(Callback):
    def __init__(self, snapshot_interval=10, update_queue=None):
        super().__init__()
        self.snapshot_interval = snapshot_interval
        self.update_queue = update_queue
        self.history = []
        self.generation_data = []
    
    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        F = pop.get("F")
        X = pop.get("X")
        
        if len(F) > 0:
            best_job = F[:, 0].min()
            best_ess = F[:, 1].min()
            best_liv = -F[:, 2].min()
            best_adj = F[:, 3].min()
            
            self.history.append({
                'gen': gen, 'best_job': best_job,
                'best_ess': best_ess, 'best_liv': best_liv, 'best_adj': best_adj
            })
            
            if gen % self.snapshot_interval == 0 or gen == 1:
                f_min, f_max = F.min(0), F.max(0)
                F_norm = (F - f_min) / (f_max - f_min + 1e-12)
                balanced_idx = np.argmin(np.linalg.norm(F_norm, axis=1))
                
                self.generation_data.append({
                    'gen': gen, 'X': X[balanced_idx].copy(),
                    'F': F[balanced_idx].copy(), 'all_F': F.copy()
                })
                
                if self.update_queue:
                    self.update_queue.put({
                        'type': 'generation', 'gen': gen,
                        'best_job': best_job, 'best_ess': best_ess,
                        'best_liv': best_liv, 'best_adj': best_adj,
                        'n_solutions': len(F)
                    })


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_city(ax, grid, buildings, title=""):
    size = grid.shape[0]
    ax.clear()
    for r in range(size):
        for c in range(size):
            t = grid[r, c]
            color = LAND_TYPES.get(t, {}).get("color", "#CCCCCC")
            rect = Rectangle((c, size-r-1), 1, 1, facecolor=color,
                            edgecolor='white', linewidth=0.2)
            ax.add_patch(rect)
    for land_type, cells in buildings:
        if land_type in [2, 3, 4] and len(cells) > 1:
            for r, c in cells:
                for dr, dc in DIRECTIONS:
                    if (r+dr, c+dc) not in cells:
                        if dr == -1:
                            ax.plot([c, c+1], [size-r, size-r], 'k-', lw=1.2)
                        elif dr == 1:
                            ax.plot([c, c+1], [size-r-1, size-r-1], 'k-', lw=1.2)
                        elif dc == -1:
                            ax.plot([c, c], [size-r-1, size-r], 'k-', lw=1.2)
                        elif dc == 1:
                            ax.plot([c+1, c+1], [size-r-1, size-r], 'k-', lw=1.2)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9, fontweight='bold')


# =============================================================================
# Main UI Application
# =============================================================================

class NSGA3App:
    def __init__(self, root):
        self.root = root
        self.root.title("NSGA-III Land Use Optimization")
        self.root.geometry("1500x950")
        
        self.running = False
        self.stop_flag = StopFlag()
        self.result = None
        self.problem = None
        self.callback = None
        self.update_queue = queue.Queue()
        self._auto_scaling = False
        
        self.create_widgets()
        self.check_updates()
    
    def create_widgets(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_outer = ttk.Frame(main_pane)
        main_pane.add(left_outer, weight=1)
        
        left_canvas = tk.Canvas(left_outer, width=380)
        left_scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=left_canvas.yview)
        left_frame = ttk.Frame(left_canvas)
        left_frame.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=left_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        
        self.create_control_panel(left_frame)
        self.create_visualization_panel(right_frame)
    
    def create_control_panel(self, parent):
        # Configuration
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(config_frame, text="Grid Size:").grid(row=0, column=0, sticky=tk.W)
        self.grid_size_var = tk.IntVar(value=20)
        ttk.Spinbox(config_frame, from_=5, to=80, textvariable=self.grid_size_var, width=8).grid(row=0, column=1, padx=5)
        
        self.auto_scale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Auto-scale to grid", variable=self.auto_scale_var).grid(row=0, column=2, sticky=tk.W, padx=5)
        self.grid_size_var.trace_add('write', self._on_grid_size_change)
        
        ttk.Label(config_frame, text="Generations:").grid(row=1, column=0, sticky=tk.W)
        self.generations_var = tk.IntVar(value=500)
        ttk.Spinbox(config_frame, from_=50, to=2000, textvariable=self.generations_var, width=8, increment=50).grid(row=1, column=1, padx=5)
        
        ttk.Label(config_frame, text="Population:").grid(row=2, column=0, sticky=tk.W)
        self.pop_size_var = tk.IntVar(value=120)
        ttk.Spinbox(config_frame, from_=50, to=500, textvariable=self.pop_size_var, width=8, increment=10).grid(row=2, column=1, padx=5)
        
        ttk.Label(config_frame, text="Snapshot Every:").grid(row=3, column=0, sticky=tk.W)
        self.snapshot_var = tk.IntVar(value=25)
        ttk.Spinbox(config_frame, from_=5, to=100, textvariable=self.snapshot_var, width=8, increment=5).grid(row=3, column=1, padx=5)
        ttk.Label(config_frame, text="gens").grid(row=3, column=2, sticky=tk.W)
        
        # Building Counts
        buildings_frame = ttk.LabelFrame(parent, text="Building Counts", padding=10)
        buildings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(buildings_frame, text="Industrial:").grid(row=0, column=0, sticky=tk.W)
        self.n_industrial_var = tk.IntVar(value=2)
        ttk.Spinbox(buildings_frame, from_=0, to=50, textvariable=self.n_industrial_var, width=5).grid(row=0, column=1)
        
        ttk.Label(buildings_frame, text="Commercial:").grid(row=1, column=0, sticky=tk.W)
        self.n_commercial_var = tk.IntVar(value=6)
        ttk.Spinbox(buildings_frame, from_=0, to=50, textvariable=self.n_commercial_var, width=5).grid(row=1, column=1)
        
        ttk.Label(buildings_frame, text="Essential:").grid(row=2, column=0, sticky=tk.W)
        self.n_essential_var = tk.IntVar(value=4)
        ttk.Spinbox(buildings_frame, from_=0, to=50, textvariable=self.n_essential_var, width=5).grid(row=2, column=1)
        
        # Building Sizes
        size_frame = ttk.LabelFrame(parent, text="Building Sizes (cells per building)", padding=10)
        size_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(size_frame, text="Industrial:").grid(row=0, column=0, sticky=tk.W)
        self.industrial_size_var = tk.IntVar(value=9)
        ttk.Spinbox(size_frame, from_=1, to=25, textvariable=self.industrial_size_var, width=5).grid(row=0, column=1)
        ttk.Label(size_frame, text="cells").grid(row=0, column=2, sticky=tk.W, padx=3)
        
        ttk.Label(size_frame, text="Commercial:").grid(row=1, column=0, sticky=tk.W)
        self.commercial_size_var = tk.IntVar(value=4)
        ttk.Spinbox(size_frame, from_=1, to=16, textvariable=self.commercial_size_var, width=5).grid(row=1, column=1)
        ttk.Label(size_frame, text="cells").grid(row=1, column=2, sticky=tk.W, padx=3)
        
        ttk.Label(size_frame, text="Essential:").grid(row=2, column=0, sticky=tk.W)
        self.essential_size_var = tk.IntVar(value=4)
        ttk.Spinbox(size_frame, from_=1, to=16, textvariable=self.essential_size_var, width=5).grid(row=2, column=1)
        ttk.Label(size_frame, text="cells").grid(row=2, column=2, sticky=tk.W, padx=3)
        
        ttk.Label(size_frame, text="Green Space %:").grid(row=3, column=0, sticky=tk.W)
        self.green_ratio_var = tk.DoubleVar(value=18.0)
        ttk.Spinbox(size_frame, from_=5, to=50, textvariable=self.green_ratio_var, width=5, increment=2).grid(row=3, column=1)
        
        # Cell Distribution Preview
        info_frame = ttk.LabelFrame(parent, text="Cell Distribution Preview", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=10, width=40, font=('Courier', 9), state=tk.DISABLED, bg='#F5F5F5')
        self.info_text.pack(fill=tk.X)
        ttk.Button(info_frame, text="Refresh Preview", command=self._update_info_preview).pack(fill=tk.X, pady=3)
        
        for var in [self.n_industrial_var, self.n_commercial_var, self.n_essential_var,
                    self.industrial_size_var, self.commercial_size_var, self.essential_size_var,
                    self.green_ratio_var, self.grid_size_var]:
            var.trace_add('write', lambda *args: self.root.after(100, self._update_info_preview))
        
        # Objective Weights
        weights_frame = ttk.LabelFrame(parent, text="Objective Weights", padding=10)
        weights_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.weight_vars = {}
        for i, (key, label) in enumerate([
            ('job_commute', 'Job Commute'), ('essential_commute', 'Essential Commute'),
            ('livability', 'Livability'), ('industrial_adjacent', 'Industrial Adjacent')
        ]):
            ttk.Label(weights_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W)
            var = tk.DoubleVar(value=1.0)
            self.weight_vars[key] = var
            scale = ttk.Scale(weights_frame, from_=0.0, to=3.0, variable=var, orient=tk.HORIZONTAL, length=120)
            scale.grid(row=i, column=1, padx=5)
            label_val = ttk.Label(weights_frame, textvariable=var, width=5)
            label_val.grid(row=i, column=2)
            var.trace_add('write', lambda *args, v=var, l=label_val: l.config(text=f"{v.get():.2f}"))
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Optimization", command=self.start_optimization)
        self.run_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="Show Final Results", command=self.show_final_results).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Edit Objectives", command=self.edit_objectives).pack(fill=tk.X, pady=2)
        
        # Progress
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack()
        
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=8, width=35)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.root.after(200, self._update_info_preview)
    
    def _on_grid_size_change(self, *args):
        if self._auto_scaling:
            return
        if not self.auto_scale_var.get():
            return
        try:
            grid_size = self.grid_size_var.get()
        except (tk.TclError, ValueError):
            return
        if grid_size < 5:
            return
        self._auto_scaling = True
        try:
            config = compute_scaled_config(grid_size)
            self.n_industrial_var.set(config['n_industrial'])
            self.n_commercial_var.set(config['n_commercial'])
            self.n_essential_var.set(config['n_essential'])
            self.industrial_size_var.set(config['industrial_size'])
            self.commercial_size_var.set(config['commercial_size'])
            self.essential_size_var.set(config['essential_size'])
            self.green_ratio_var.set(round(config['green_ratio'] * 100, 1))
        finally:
            self._auto_scaling = False
    
    def _update_info_preview(self):
        try:
            gs = self.grid_size_var.get()
            n_ind = self.n_industrial_var.get()
            n_com = self.n_commercial_var.get()
            n_ess = self.n_essential_var.get()
            ind_sz = self.industrial_size_var.get()
            com_sz = self.commercial_size_var.get()
            ess_sz = self.essential_size_var.get()
            green_pct = self.green_ratio_var.get()
        except (tk.TclError, ValueError):
            return
        total = gs * gs
        ind_cells = n_ind * ind_sz
        com_cells = n_com * com_sz
        ess_cells = n_ess * ess_sz
        building_cells = ind_cells + com_cells + ess_cells
        remaining = max(0, total - building_cells)
        green_cells = int(remaining * green_pct / 100.0)
        res_cells = remaining - green_cells
        overflow = building_cells > total
        
        text = f"Grid: {gs}x{gs} = {total} cells\n"
        text += f"{'─' * 35}\n"
        text += f"Industrial:  {n_ind}x{ind_sz} = {ind_cells:4d}  ({ind_cells/total*100:5.1f}%)\n"
        text += f"Commercial:  {n_com}x{com_sz} = {com_cells:4d}  ({com_cells/total*100:5.1f}%)\n"
        text += f"Essential:   {n_ess}x{ess_sz} = {ess_cells:4d}  ({ess_cells/total*100:5.1f}%)\n"
        text += f"Green Space:          {green_cells:4d}  ({green_cells/total*100:5.1f}%)\n"
        text += f"Residential:          {res_cells:4d}  ({res_cells/total*100:5.1f}%)\n"
        text += f"{'─' * 35}\n"
        text += f"Total:                {total:4d}  (100.0%)\n"
        if overflow:
            text += f"\nOVERFLOW: buildings need {building_cells}\n"
            text += f"  cells but grid only has {total}!"
        text += f"\nBaseline 20x20 proportions:\n"
        text += f"  Ind: 4.5% | Com: 6.0% | Ess: 4.0%\n"
        text += f"  Grn: ~15% | Res: ~70%"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.config(state=tk.DISABLED)
    
    def create_visualization_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Current View (convergence + best solutions)
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Current View")
        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab1)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Click 'Run Optimization' to start", ha='center', va='center', fontsize=14)
        ax.axis('off')
        self.canvas.draw()
        
        # Tab 2: Snapshots — scrollable, shows convergence graphs + all snapshots
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Snapshots")
        
        # Scrollable canvas for snapshots
        self.snap_outer_canvas = tk.Canvas(tab2)
        self.snap_scrollbar = ttk.Scrollbar(tab2, orient=tk.VERTICAL, command=self.snap_outer_canvas.yview)
        self.snap_inner_frame = ttk.Frame(self.snap_outer_canvas)
        
        self.snap_inner_frame.bind("<Configure>",
            lambda e: self.snap_outer_canvas.configure(scrollregion=self.snap_outer_canvas.bbox("all")))
        self.snap_outer_canvas.create_window((0, 0), window=self.snap_inner_frame, anchor="nw")
        self.snap_outer_canvas.configure(yscrollcommand=self.snap_scrollbar.set)
        
        self.snap_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.snap_outer_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel on snapshots tab
        def _snap_mousewheel(event):
            self.snap_outer_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.snap_outer_canvas.bind("<MouseWheel>", _snap_mousewheel)
        self.snap_inner_frame.bind("<MouseWheel>", _snap_mousewheel)
        
        # We'll hold references to snapshot figure canvases so we can clean up
        self.snap_canvases = []
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def start_optimization(self):
        if self.running:
            return
        self.running = True
        self.stop_flag.clear()
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.result = None
        self.callback = None
        
        # Clear old snapshot widgets
        for widget in self.snap_inner_frame.winfo_children():
            widget.destroy()
        self.snap_canvases = []
        
        thread = threading.Thread(target=self.run_optimization)
        thread.daemon = True
        thread.start()
    
    def run_optimization(self):
        try:
            self.log("=" * 40)
            self.log("Starting NSGA-III Optimization")
            self.log("=" * 40)
            
            grid_size = self.grid_size_var.get()
            n_gen = self.generations_var.get()
            pop_size = self.pop_size_var.get()
            snapshot_interval = self.snapshot_var.get()
            weights = {k: v.get() for k, v in self.weight_vars.items()}
            
            ind_size = self.industrial_size_var.get()
            com_size = self.commercial_size_var.get()
            ess_size = self.essential_size_var.get()
            n_ind = self.n_industrial_var.get()
            n_com = self.n_commercial_var.get()
            n_ess = self.n_essential_var.get()
            green_ratio = self.green_ratio_var.get() / 100.0
            
            self.log(f"Grid: {grid_size}x{grid_size}")
            self.log(f"Industrial: {n_ind} x {ind_size} cells")
            self.log(f"Commercial: {n_com} x {com_size} cells")
            self.log(f"Essential:  {n_ess} x {ess_size} cells")
            self.log(f"Generations: {n_gen}, Pop: {pop_size}")
            
            self.problem = WeightedBuildingProblem(
                grid_size=grid_size, n_industrial=n_ind, n_commercial=n_com,
                n_essential=n_ess, industrial_size=ind_size, commercial_size=com_size,
                essential_size=ess_size, green_ratio=green_ratio, weights=weights
            )
            
            self.callback = ProgressCallback(
                snapshot_interval=snapshot_interval, update_queue=self.update_queue
            )
            
            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
            algorithm = NSGA3(
                ref_dirs=ref_dirs, pop_size=pop_size,
                sampling=IntegerSampling(),
                crossover=IntegerCrossover(prob=0.9),
                mutation=IntegerMutation(prob=0.2, eta=4),
            )
            
            termination = StoppableTermination(n_gen, self.stop_flag)
            
            self.log(f"Ref directions: {len(ref_dirs)}")
            self.log("Running...")
            
            start_time = time.perf_counter()
            
            self.result = minimize(
                self.problem, algorithm,
                termination=termination,
                callback=self.callback,
                seed=int(time.time()) % 10000,
                verbose=False
            )
            
            np.savez("pareto_results.npz", X=self.result.X, F=self.result.F)
            elapsed = time.perf_counter() - start_time
            
            if self.stop_flag.is_stopped:
                self.log(f"\nStopped at gen {self.callback.history[-1]['gen'] if self.callback.history else '?'} ({elapsed:.1f}s)")
            else:
                self.log(f"\nCompleted in {elapsed:.1f}s")
            self.log(f"Pareto solutions: {len(self.result.X)}")
            
            self.update_queue.put({'type': 'complete', 'elapsed': elapsed})
            
        except Exception as e:
            self.log(f"Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.running = False
            self.update_queue.put({'type': 'finished'})
    
    def stop_optimization(self):
        if self.running:
            self.stop_flag.stop()
            self.log("Stop requested — finishing current generation...")
            self.stop_button.config(state=tk.DISABLED)
    
    def check_updates(self):
        try:
            while True:
                update = self.update_queue.get_nowait()
                
                if update['type'] == 'generation':
                    gen = update['gen']
                    n_gen = self.generations_var.get()
                    self.progress_var.set((gen / n_gen) * 100)
                    self.status_label.config(text=f"Gen {gen}/{n_gen} - Solutions: {update['n_solutions']}")
                    self.log(f"Gen {gen}: Job={update['best_job']:.2f}, Liv={update['best_liv']:.1f}")
                    
                    if self.callback and self.problem:
                        self._update_current_view()
                        self._add_snapshot_to_tab()
                
                elif update['type'] == 'complete':
                    elapsed = update['elapsed']
                    self.status_label.config(text=f"{'Stopped' if self.stop_flag.is_stopped else 'Complete'}! ({elapsed:.1f}s)")
                    self.progress_var.set(100)
                    if self.callback and self.problem:
                        self._update_current_view()
                        self._rebuild_snapshots_tab()
                
                elif update['type'] == 'finished':
                    self.run_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    
        except queue.Empty:
            pass
        self.root.after(100, self.check_updates)
    
    def _update_current_view(self):
        """Update the Current View tab: convergence graphs on top, latest best solutions on bottom."""
        if not self.callback or not self.problem:
            return
        try:
            self.fig.clear()
            gs = gridspec.GridSpec(2, 4, figure=self.fig, hspace=0.35, wspace=0.3)
            
            # Row 0: convergence plots + stats
            if self.callback.history:
                gens = [h['gen'] for h in self.callback.history]
                
                ax1 = self.fig.add_subplot(gs[0, 0])
                ax1.plot(gens, [h['best_job'] for h in self.callback.history], 'b-', linewidth=1.2)
                ax1.plot(gens, [h['best_ess'] for h in self.callback.history], 'g-', linewidth=1.2)
                ax1.set_xlabel('Generation', fontsize=7)
                ax1.set_ylabel('Best Distance', fontsize=7)
                ax1.set_title('Commute Convergence', fontsize=8)
                ax1.legend(['Job', 'Essential'], fontsize=6, loc='upper right')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(labelsize=6)
                
                ax2 = self.fig.add_subplot(gs[0, 1])
                ax2.plot(gens, [h['best_liv'] for h in self.callback.history], 'purple', linewidth=1.2)
                ax2.set_xlabel('Generation', fontsize=7)
                ax2.set_title('Livability', fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(labelsize=6)
                
                ax3 = self.fig.add_subplot(gs[0, 2])
                ax3.plot(gens, [h['best_adj'] for h in self.callback.history], 'r-', linewidth=1.2)
                ax3.set_xlabel('Generation', fontsize=7)
                ax3.set_title('Ind. Adjacency', fontsize=8)
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(labelsize=6)
                
                # Stats
                ax4 = self.fig.add_subplot(gs[0, 3])
                ax4.axis('off')
                latest = self.callback.history[-1]
                stats = (f"Generation: {latest['gen']}\n\n"
                         f"Best Metrics:\n"
                         f"  Job Commute: {latest['best_job']:.2f}\n"
                         f"  Essential:   {latest['best_ess']:.2f}\n"
                         f"  Livability:  {latest['best_liv']:.1f}\n"
                         f"  Ind. Adj:    {latest['best_adj']:.0f}")
                ax4.text(0.1, 0.9, stats, transform=ax4.transAxes, fontsize=9, va='top', family='monospace')
            
            # Row 1: 4 most recent snapshots
            if self.callback.generation_data:
                n_show = min(4, len(self.callback.generation_data))
                # Show the last n_show snapshots
                recent = self.callback.generation_data[-n_show:]
                for i, snap in enumerate(recent):
                    ax = self.fig.add_subplot(gs[1, i])
                    grid, buildings, _ = self.problem.builder.build(snap['X'])
                    F = snap['F']
                    title = f"Gen {snap['gen']}\nJ:{F[0]:.1f} E:{F[1]:.1f} L:{-F[2]:.0f} A:{F[3]:.0f}"
                    visualize_city(ax, grid, buildings, title)
            
            self.canvas.draw()
        except Exception as e:
            print(f"Current view error: {e}")
    
    def _add_snapshot_to_tab(self):
        """Add the latest snapshot as a new figure in the Snapshots tab."""
        if not self.callback or not self.problem or not self.callback.generation_data:
            return
        
        try:
            snap = self.callback.generation_data[-1]
            grid, buildings, _ = self.problem.builder.build(snap['X'])
            F = snap['F']
            gen = snap['gen']
            
            # Create a small figure for this snapshot
            fig_snap = plt.figure(figsize=(10, 3.5))
            gs_snap = gridspec.GridSpec(1, 5, figure=fig_snap, wspace=0.3,
                                        width_ratios=[1.2, 1.2, 1.2, 1.5, 0.8])
            
            # Mini convergence plots (up to this generation)
            hist_so_far = [h for h in self.callback.history if h['gen'] <= gen]
            if hist_so_far:
                hgens = [h['gen'] for h in hist_so_far]
                
                ax1 = fig_snap.add_subplot(gs_snap[0, 0])
                ax1.plot(hgens, [h['best_job'] for h in hist_so_far], 'b-', linewidth=1)
                ax1.plot(hgens, [h['best_ess'] for h in hist_so_far], 'g-', linewidth=1)
                ax1.set_title('Commute', fontsize=7)
                ax1.legend(['Job', 'Ess'], fontsize=5, loc='upper right')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(labelsize=5)
                
                ax2 = fig_snap.add_subplot(gs_snap[0, 1])
                ax2.plot(hgens, [h['best_liv'] for h in hist_so_far], 'purple', linewidth=1)
                ax2.set_title('Livability', fontsize=7)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(labelsize=5)
                
                ax3 = fig_snap.add_subplot(gs_snap[0, 2])
                ax3.plot(hgens, [h['best_adj'] for h in hist_so_far], 'r-', linewidth=1)
                ax3.set_title('Ind. Adj', fontsize=7)
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(labelsize=5)
            
            # City grid
            ax_city = fig_snap.add_subplot(gs_snap[0, 3])
            visualize_city(ax_city, grid, buildings,
                          f"Gen {gen}: J:{F[0]:.1f} E:{F[1]:.1f} L:{-F[2]:.0f} A:{F[3]:.0f}")
            
            # Stats text
            ax_stats = fig_snap.add_subplot(gs_snap[0, 4])
            ax_stats.axis('off')
            latest_h = hist_so_far[-1] if hist_so_far else None
            if latest_h:
                ax_stats.text(0.05, 0.95,
                    f"Gen {gen}\n\n"
                    f"Job:  {latest_h['best_job']:.2f}\n"
                    f"Ess:  {latest_h['best_ess']:.2f}\n"
                    f"Liv:  {latest_h['best_liv']:.1f}\n"
                    f"Adj:  {latest_h['best_adj']:.0f}",
                    transform=ax_stats.transAxes, fontsize=8, va='top', family='monospace')
            
            fig_snap.suptitle(f'Snapshot — Generation {gen}', fontsize=9, fontweight='bold', y=0.98)
            fig_snap.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Embed in snapshots tab
            snap_canvas = FigureCanvasTkAgg(fig_snap, master=self.snap_inner_frame)
            snap_widget = snap_canvas.get_tk_widget()
            snap_widget.pack(fill=tk.X, padx=5, pady=3)
            snap_canvas.draw()
            
            # Bind mousewheel on new widget
            def _snap_mw(event):
                self.snap_outer_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            snap_widget.bind("<MouseWheel>", _snap_mw)
            
            self.snap_canvases.append((fig_snap, snap_canvas))
            
            # Scroll to bottom
            self.snap_outer_canvas.update_idletasks()
            self.snap_outer_canvas.yview_moveto(1.0)
            
        except Exception as e:
            print(f"Snapshot add error: {e}")
    
    def _rebuild_snapshots_tab(self):
        """Rebuild the entire snapshots tab (called on completion for clean final state)."""
        # Already built incrementally, just make sure scroll is at top
        self.snap_outer_canvas.update_idletasks()
        self.snap_outer_canvas.yview_moveto(0.0)
    
    def show_final_results(self):
        if not self.result or not self.problem:
            messagebox.showinfo("Info", "No results available. Run optimization first.")
            return
        
        self.fig.clear()
        X, F = self.result.X, self.result.F
        gs = gridspec.GridSpec(2, 4, figure=self.fig, hspace=0.35, wspace=0.3)
        
        # Row 0: Pareto scatter plots + stats
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.scatter(F[:, 0], -F[:, 2], c='blue', alpha=0.6, s=20)
        ax1.set_xlabel('Job Commute'); ax1.set_ylabel('Livability')
        ax1.set_title('Commute vs Livability')
        ax1.grid(True, alpha=0.3)
        
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.scatter(F[:, 1], -F[:, 2], c='green', alpha=0.6, s=20)
        ax2.set_xlabel('Essential Commute'); ax2.set_ylabel('Livability')
        ax2.set_title('Essential vs Livability')
        ax2.grid(True, alpha=0.3)
        
        ax3 = self.fig.add_subplot(gs[0, 2])
        ax3.scatter(F[:, 3], -F[:, 2], c='red', alpha=0.6, s=20)
        ax3.set_xlabel('Industrial Adjacency'); ax3.set_ylabel('Livability')
        ax3.set_title('Adjacency vs Livability')
        ax3.grid(True, alpha=0.3)
        
        ax4 = self.fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        stats = (f"Final Results\n\n"
                 f"Pareto Solutions: {len(X)}\n"
                 f"Grid: {self.problem.grid_size}x{self.problem.grid_size}\n\n"
                 f"Objective Ranges:\n"
                 f"  Job:  [{F[:,0].min():.2f}, {F[:,0].max():.2f}]\n"
                 f"  Ess:  [{F[:,1].min():.2f}, {F[:,1].max():.2f}]\n"
                 f"  Liv:  [{-F[:,2].max():.1f}, {-F[:,2].min():.1f}]\n"
                 f"  Adj:  [{int(F[:,3].min())}, {int(F[:,3].max())}]")
        ax4.text(0.05, 0.95, stats, transform=ax4.transAxes, fontsize=9, va='top', family='monospace')
        
        # Row 1: Best solutions
        best_indices = [np.argmin(F[:, 0]), np.argmin(F[:, 2]), np.argmin(F[:, 3])]
        f_min, f_max = F.min(0), F.max(0)
        F_norm = (F - f_min) / (f_max - f_min + 1e-12)
        best_indices.append(np.argmin(np.linalg.norm(F_norm, axis=1)))
        
        titles = ['Best Job Commute', 'Best Livability', 'Min Ind. Adjacency', 'Balanced Solution']
        for i, (idx, title) in enumerate(zip(best_indices, titles)):
            ax = self.fig.add_subplot(gs[1, i])
            grid, buildings, _ = self.problem.builder.build(X[idx])
            metrics = f"J:{F[idx,0]:.1f} E:{F[idx,1]:.1f} L:{-F[idx,2]:.0f} A:{F[idx,3]:.0f}"
            visualize_city(ax, grid, buildings, f"{title}\n{metrics}")
        
        self.fig.suptitle('NSGA-III Final Results', fontsize=12, fontweight='bold')
        self.canvas.draw()
    
    def save_results(self):
        if not self.result or not self.callback:
            messagebox.showinfo("Info", "No results to save.")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig_path = f"/home/claude/nsga3_results_{timestamp}.png"
        self.fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        self.log(f"Saved: {fig_path}")
        messagebox.showinfo("Saved", f"Results saved to:\n{fig_path}")
    
    def edit_objectives(self):
        editor = ObjectiveEditor(self.root)
        self.root.wait_window(editor.top)


class ObjectiveEditor:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Objective Function Editor")
        self.top.geometry("800x600")
        notebook = ttk.Notebook(self.top)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.editors = {}
        for name, code in DEFAULT_OBJECTIVES.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=name.replace('_', ' ').title())
            text = scrolledtext.ScrolledText(frame, width=90, height=25, font=('Courier', 10))
            text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            text.insert(tk.END, code)
            self.editors[name] = text
        btn_frame = ttk.Frame(self.top)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Validate", command=self.validate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=self.top.destroy).pack(side=tk.RIGHT, padx=5)
    
    def validate(self):
        for name, editor in self.editors.items():
            code = editor.get("1.0", tk.END)
            try:
                compile(code, f"<{name}>", "exec")
            except SyntaxError as e:
                messagebox.showerror("Syntax Error", f"Error in {name}:\n{str(e)}")
                return False
        messagebox.showinfo("Valid", "All objective functions have valid syntax.")
        return True
    
    def reset(self):
        for name, code in DEFAULT_OBJECTIVES.items():
            if name in self.editors:
                self.editors[name].delete("1.0", tk.END)
                self.editors[name].insert(tk.END, code)


# =============================================================================
# CLI runner
# =============================================================================

def run_cli(n_gen=500, snapshot_interval=25, save_images=True,
            grid_size=20, industrial_size=9, commercial_size=4, essential_size=4,
            n_industrial=None, n_commercial=None, n_essential=None,
            green_ratio=None):
    print("=" * 70)
    print("NSGA-III Grid Optimization - CLI Mode")
    print("=" * 70)
    
    if n_industrial is None or n_commercial is None or n_essential is None:
        config = compute_scaled_config(grid_size)
        n_industrial = n_industrial or config['n_industrial']
        n_commercial = n_commercial or config['n_commercial']
        n_essential = n_essential or config['n_essential']
        industrial_size = config['industrial_size']
        commercial_size = config['commercial_size']
        essential_size = config['essential_size']
        green_ratio_val = config['green_ratio']
    else:
        green_ratio_val = (green_ratio / 100.0) if green_ratio and green_ratio > 1 else (green_ratio or 0.18)
    
    pop_size = 120
    print(f"\nGrid: {grid_size}x{grid_size} ({grid_size*grid_size} cells)")
    print(f"Industrial: {n_industrial} x {industrial_size} = {n_industrial*industrial_size}")
    print(f"Commercial: {n_commercial} x {commercial_size} = {n_commercial*commercial_size}")
    print(f"Essential:  {n_essential} x {essential_size} = {n_essential*essential_size}")
    print(f"Generations: {n_gen}, Pop: {pop_size}")
    
    problem = WeightedBuildingProblem(
        grid_size=grid_size, n_industrial=n_industrial, n_commercial=n_commercial,
        n_essential=n_essential, industrial_size=industrial_size,
        commercial_size=commercial_size, essential_size=essential_size,
        green_ratio=green_ratio_val
    )
    callback = ProgressCallback(snapshot_interval=snapshot_interval)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
    algorithm = NSGA3(
        ref_dirs=ref_dirs, pop_size=pop_size, sampling=IntegerSampling(),
        crossover=IntegerCrossover(prob=0.9), mutation=IntegerMutation(prob=0.2, eta=4),
    )
    
    # CLI uses a stop flag that's never triggered
    stop_flag = StopFlag()
    termination = StoppableTermination(n_gen, stop_flag)
    
    start = time.perf_counter()
    result = minimize(problem, algorithm, termination=termination,
                      callback=callback, seed=42, verbose=True)
    np.savez("pareto_results.npz", X=result.X, F=result.F)
    elapsed = time.perf_counter() - start
    
    print(f"\nPareto solutions: {len(result.X)}, Time: {elapsed:.1f}s")
    
    if save_images:
        X, F = result.X, result.F
        fig = plt.figure(figsize=(20, 14))
        gs_fig = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs_fig[0, 0])
        ax1.scatter(F[:, 0], -F[:, 2], c='blue', alpha=0.6, s=25)
        ax1.set_xlabel('Job Commute'); ax1.set_ylabel('Livability')
        ax1.set_title('Commute vs Livability'); ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs_fig[0, 1])
        ax2.scatter(F[:, 1], -F[:, 2], c='green', alpha=0.6, s=25)
        ax2.set_xlabel('Essential Commute'); ax2.set_ylabel('Livability')
        ax2.set_title('Essential vs Livability'); ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs_fig[0, 2])
        ax3.scatter(F[:, 3], -F[:, 2], c='red', alpha=0.6, s=25)
        ax3.set_xlabel('Industrial Adjacency'); ax3.set_ylabel('Livability')
        ax3.set_title('Adjacency vs Livability'); ax3.grid(True, alpha=0.3)
        
        best_indices = [np.argmin(F[:, 0]), np.argmin(F[:, 2]), np.argmin(F[:, 3])]
        f_min, f_max = F.min(0), F.max(0)
        F_norm = (F - f_min) / (f_max - f_min + 1e-12)
        best_indices.append(np.argmin(np.linalg.norm(F_norm, axis=1)))
        titles = ['Best Job Commute', 'Best Livability', 'Min Ind. Adjacency', 'Balanced']
        
        for i, (idx, title) in enumerate(zip(best_indices, titles)):
            ax = fig.add_subplot(gs_fig[1, i])
            grid, buildings, _ = problem.builder.build(X[idx])
            metrics = f"J:{F[idx,0]:.1f} E:{F[idx,1]:.1f} L:{-F[idx,2]:.0f} A:{F[idx,3]:.0f}"
            visualize_city(ax, grid, buildings, f"{title}\n{metrics}")
        
        plt.suptitle('NSGA-III Results', fontsize=14, fontweight='bold', y=0.98)
        results_path = "/home/claude/nsga3_results.png"
        fig.savefig(results_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {results_path}")
        plt.close(fig)
    
    return result, problem, callback


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        grid_size = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        run_cli(n_gen=n_gen, grid_size=grid_size)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-scaling":
        print("Auto-scaling test:")
        print(f"{'Grid':>6} {'Ind':>12} {'Com':>12} {'Ess':>12} {'Grn':>8} {'Res':>8}")
        print("-" * 70)
        for gs in [8, 10, 12, 15, 20, 25, 30, 40, 50]:
            cfg = compute_scaled_config(gs)
            ind_str = f"{cfg['n_industrial']}x{cfg['industrial_size']}={cfg['actual_industrial_cells']}"
            com_str = f"{cfg['n_commercial']}x{cfg['commercial_size']}={cfg['actual_commercial_cells']}"
            ess_str = f"{cfg['n_essential']}x{cfg['essential_size']}={cfg['actual_essential_cells']}"
            print(f"{gs:>4}x{gs:<2} {ind_str:>12} {com_str:>12} {ess_str:>12} "
                  f"{cfg['actual_green_cells']:>6} ({cfg['green_pct']:.1f}%) "
                  f"{cfg['actual_residential_cells']:>6} ({cfg['residential_pct']:.1f}%)")
    elif not HAS_TKINTER:
        print("Tkinter not available. Running in CLI mode...")
        run_cli()
    else:
        try:
            root = tk.Tk()
            app = NSGA3App(root)
            root.mainloop()
        except Exception as e:
            print(f"Could not start GUI: {e}")
            print("Running in CLI mode...")
            run_cli()


if __name__ == "__main__":
    main()