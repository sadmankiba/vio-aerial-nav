import random
import logging
import heapq
import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

from utils.cam_util import (
    CamProp,
    get_cam_axes,
    get_cam_intrinsic_params,
    get_cam_intrinsic_params_diff_aspect,
)
from utils.pos_util import Point3D, local_to_world, perpendicular_distance, yaw_deg_to_vector

"""
States:
•	Free: The agent has sensed this area and confirmed no obstacles exist.
•	Occupied: The agent has sensed an obstacle (wall, furniture) here.
•	Unknown: The agent has not yet scanned this area with its sensors.

In literature, a "Frontier Cell" is any Free cell that is adjacent to at least one unknown cell.

But, here we are defining a frontier cell that can be adjacent to unknown cells or free cells.
"""

### CONFIGURABLE ###
# Two dim map
FRONTIER_SCORING_KERNEL_SIZE = (11, 11)
MAP_MAX_RANGE = 420  # meters
TARGET_DIR_SIGMA = 16  # beam width

# If drone is on course, then all cells in nearby region will have good score between 0.7 - 1.0.
# We want to balance it for obstacle / free / visited / unknown cells.
# Already visited direction -> less unknown and not visited -> more unknown
# high unknown score (and higher than free) will encourage drone to visit new places.
UNKNOWN_CELL_WEIGHT = 1.50
FREE_CELL_WEIGHT = 0.75
VISITED_CELL_WEIGHT = -4
OBSTACLE_CELL_WEIGHT = -0.5
TARGET_DIR_SCORE_WEIGHT = 40.0
TARGET_MASK_VALUE_THRESHOLD = 0.9  # About 5 meters from target direction line
EXP_SAMPLING_TEMPERATURE = 10.0

# Grid Height Ranges for Obstacle Marking (NED Z: positive downwards)
LOWER_GRID_Z_RANGE = (-10, 3)
MID_GRID_Z_RANGE = (-20, -5)
UPPER_GRID_Z_RANGE = (-35, -16)
UPPER_2_GRID_Z_RANGE = (-50, -30)
UPPER_3_GRID_Z_RANGE = (-70, -45)

# Grid Selection Thresholds
LOWER_MID_Z_THRESHOLD = -7
MID_UPPER_Z_THRESHOLD = -18
UPPER_UPPER_2_Z_THRESHOLD = -33
UPPER_2_UPPER_3_Z_THRESHOLD = -48

# 3D Occup Map and Target Direction Follow
DEPTH_IMG_SAMPLE_STEP = 4

FREE_MIN_DISTANCE = 3  # meters
FREE_MAX_DISTANCE = 40  # meters
OBS_MIN_DISTANCE = 3  # meters
OBS_MAX_DISTANCE = 25  # meters
TARGET_DIR_DEST_INTERVAL = 5 # meters

TARGET_DIR_LINE_DIST_MAX_NEAR = 10.0
TARGET_DIR_LINE_DIST_MAX_FAR = 16.0
TARGET_DIR_LINE_DIST_MAX_NEARBY_OBJ = 45.0
####################


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GridMark(Enum):
    UNKNOWN = 0
    OBSTACLE = 2
    FREE = 3


class TargetDirMark(Enum):
    UNVISITED = 0
    VISITED = 1
    OBSTACLE = 2
    UNREACHED = 3


@dataclass
class MapConfig:
    # mapping
    free_min_distance: float
    free_max_distance: float
    obs_min_distance: float
    obs_max_distance: float
    free_on_unknown_only: bool

    # path planning
    max_path_plan_iterations: int
    path_idx_jump: int
    dist_near_obstacle_thr: float
    dist_near_visited_thr: float
    frac_obstacle_radius: float
    frac_free_radius: float
    obs_penalty_constant_factor: float
    free_bonus_constant_factor: float
    target_dir_dest_interval: int

    # target direction
    target_dir_line_max_nearby_obj: float
    target_dir_line_max_near: float
    target_dir_line_max_far: float


OPENUAV_MAP_CONFIG = MapConfig(
    # mapping
    free_min_distance=FREE_MIN_DISTANCE,
    free_max_distance=FREE_MAX_DISTANCE,
    obs_min_distance=OBS_MIN_DISTANCE,
    obs_max_distance=OBS_MAX_DISTANCE,
    free_on_unknown_only=True,
    
    # path planning
    max_path_plan_iterations=20000,
    path_idx_jump=2,
    dist_near_obstacle_thr=0.4,
    dist_near_visited_thr=0.4,
    frac_obstacle_radius=1.5,
    frac_free_radius=3.0,
    obs_penalty_constant_factor=1.5,
    free_bonus_constant_factor=3.0,
    target_dir_dest_interval=TARGET_DIR_DEST_INTERVAL,
    
    # target direction
    target_dir_line_max_near=TARGET_DIR_LINE_DIST_MAX_NEAR,
    target_dir_line_max_far=TARGET_DIR_LINE_DIST_MAX_FAR,
    target_dir_line_max_nearby_obj=TARGET_DIR_LINE_DIST_MAX_NEARBY_OBJ,
)


class TwoDimMap:
    def __init__(self):
        self.x_range = (-400, 400)
        self.y_range = (-400, 400)
        self.res = 1

        # Calculate grid dimensions
        self.nx = int((self.x_range[1] - self.x_range[0]) // self.res)
        self.ny = int((self.y_range[1] - self.y_range[0]) // self.res)

        # State grids: with GridMark enum values for different height ranges
        self.grid_lower = np.zeros((self.nx, self.ny), dtype=np.uint8)
        self.grid_mid = np.zeros((self.nx, self.ny), dtype=np.uint8)
        self.grid_upper = np.zeros((self.nx, self.ny), dtype=np.uint8)
        self.grid_upper_2 = np.zeros((self.nx, self.ny), dtype=np.uint8)
        self.grid_upper_3 = np.zeros((self.nx, self.ny), dtype=np.uint8)
        self.target_yaw = None

        self.init_cells()

        # target_value is a float grid for beam-like scoring
        self.target_value = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Visited mask for persistence
        self.visited_grid = np.zeros((self.nx, self.ny), dtype=bool)

        self.frontier_candidate_indices = None
        self.frontier_scores = None
        self.frontier_probs = None
        self.frontier_indices = None

    def init_cells(self):
        # Mark all cells as unknown
        self.grid_lower.fill(GridMark.UNKNOWN.value)
        self.grid_mid.fill(GridMark.UNKNOWN.value)
        self.grid_upper.fill(GridMark.UNKNOWN.value)
        self.grid_upper_2.fill(GridMark.UNKNOWN.value)
        self.grid_upper_3.fill(GridMark.UNKNOWN.value)

        # Mark cells outside 400m circle as obstacles (vectorized)
        x = np.linspace(self.x_range[0], self.x_range[1], self.nx, endpoint=False)
        y = np.linspace(self.y_range[0], self.y_range[1], self.ny, endpoint=False)
        xv, yv = np.meshgrid(x, y, indexing="ij")
        dist = np.sqrt(xv**2 + yv**2)
        circle_mask = dist > 400
        self.grid_lower[circle_mask] = GridMark.OBSTACLE.value
        self.grid_mid[circle_mask] = GridMark.OBSTACLE.value
        self.grid_upper[circle_mask] = GridMark.OBSTACLE.value
        self.grid_upper_2[circle_mask] = GridMark.OBSTACLE.value
        self.grid_upper_3[circle_mask] = GridMark.OBSTACLE.value

    def mark_visited(self, point):
        ix, iy = self._get_idx(*point[:2])
        if self._is_in_bounds(ix, iy):
            self.visited_grid[ix, iy] = True
            # Also ensure it's marked as FREE (at least not UNKNOWN) in the appropriate grid
            grid = self._get_grid_for_z(point[2])
            if grid[ix, iy] == GridMark.UNKNOWN.value:
                grid[ix, iy] = GridMark.FREE.value

    def mark_unvisited(self, point):
        ix, iy = self._get_idx(*point[:2])
        if self._is_in_bounds(ix, iy):
            grid = self._get_grid_for_z(point[2])
            grid[ix, iy] = GridMark.FREE.value
            self.visited_grid[ix, iy] = False

    def mark_obstacle(self, point):
        ix, iy = self._get_idx(*point[:2])
        if self._is_in_bounds(ix, iy):
            grid = self._get_grid_for_z(point[2])
            grid[ix, iy] = GridMark.OBSTACLE.value

    def _get_idx(self, x, y):
        """Convert point coordinates to grid indices."""
        ix = int((x - self.x_range[0]) // self.res)
        iy = int((y - self.y_range[0]) // self.res)
        return ix, iy

    def _get_idx_vectorized(self, points):
        """Convert point coordinates to grid indices."""
        ix = ((points[:, 0] - self.x_range[0]) // self.res).astype(int)
        iy = ((points[:, 1] - self.y_range[0]) // self.res).astype(int)
        return ix, iy

    def _get_center(self, ix, iy) -> tuple[float, float]:
        """Convert grid indices to center coordinates."""
        cx = self.x_range[0] + (ix + 0.5) * self.res
        cy = self.y_range[0] + (iy + 0.5) * self.res
        return cx, cy

    def _is_in_bounds(self, ix, iy):
        """Check if grid indices are within bounds."""
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    def is_nearby_area_visited(self, point, radius) -> bool:
        """
        Check if any cell in the occupancy grid within radius of the point has been visited.
        Returns True if visited, False otherwise.
        """
        px, py = point[:2]
        ix, iy = self._get_idx(px, py)
        radius_cells = int(np.ceil(radius / self.res))

        # Define the search range within bounds
        x_start = max(0, ix - radius_cells)
        x_end = min(self.nx, ix + radius_cells + 1)
        y_start = max(0, iy - radius_cells)
        y_end = min(self.ny, iy + radius_cells + 1)

        # Get grid indices and corresponding world coordinates
        ix_range = np.arange(x_start, x_end)
        iy_range = np.arange(y_start, y_end)
        ixv, iyv = np.meshgrid(ix_range, iy_range, indexing="ij")

        # Convert grid indices back to centers
        cx = self.x_range[0] + (ixv + 0.5) * self.res
        cy = self.y_range[0] + (iyv + 0.5) * self.res

        # Calculate distance to point
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2

        # Mask for radius
        mask = dist_sq <= radius**2

        # Check for visited cells in the masked sub-grid
        sub_visited = self.visited_grid[x_start:x_end, y_start:y_end]
        return np.any(sub_visited[mask])

    def mark_path_visited(self, path):
        """
        Mark all cells along the path as visited.
        path: list of points (N, 3) or (N, 2)
        """
        # logger.debug(f"Marking path visited: {path}")
        if path is None or len(path) < 2:
            return

        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            dist = np.linalg.norm(p2[:2] - p1[:2])
            num_points = int(dist / (self.res / 2)) + 2
            pts = np.linspace(p1[:2], p2[:2], num_points)
            self.update_visited_grid(pts)

    def update_grid_vectorized(self, points, mark_value, grid, only_unknown=False):
        """
        Update grid cells at given points with mark_value.
        points: (M, 2) or (M, 3) numpy array
        """
        if len(points) == 0:
            return

        ix, iy = self._get_idx_vectorized(points)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

        if only_unknown:
            # Only update cells that are currently UNKNOWN
            target_indices = (ix[valid], iy[valid])
            unknown_mask = grid[target_indices] == GridMark.UNKNOWN.value
            grid[target_indices[0][unknown_mask], target_indices[1][unknown_mask]] = (
                mark_value
            )
        else:
            grid[ix[valid], iy[valid]] = mark_value

        # Invalidate frontier scores and indices
        self.frontier_candidate_indices = None
        self.frontier_scores = None
        self.frontier_probs = None
        self.frontier_indices = None

    def update_visited_grid(self, points):
        if len(points) == 0:
            return

        ix, iy = self._get_idx_vectorized(points)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        self.visited_grid[ix[valid], iy[valid]] = True

    def _get_grid_for_z(self, z):
        if z > LOWER_MID_Z_THRESHOLD:
            return self.grid_lower
        elif MID_UPPER_Z_THRESHOLD <= z <= LOWER_MID_Z_THRESHOLD:
            return self.grid_mid
        elif UPPER_UPPER_2_Z_THRESHOLD <= z < MID_UPPER_Z_THRESHOLD:
            return self.grid_upper
        elif UPPER_2_UPPER_3_Z_THRESHOLD <= z < UPPER_UPPER_2_Z_THRESHOLD:
            return self.grid_upper_2
        else:
            return self.grid_upper_3

    def update_from_depth_images(
        self, current_pos, rot_mat, depth_images: dict[str, np.ndarray], sample_step=2
    ):
        """
        Vectorized update of the occupancy grid from 5 depth images.

        - Samples rays from all 5 cameras.
        - Projects rays into world coordinates using current UAV pose.
        - Marks free space along rays (up to 25m).
        - Marks obstacles at terminating points in the appropriate height-based grid(s).
        """
        cameras = get_cam_axes()
        max_depth_meters = 100.0

        for cam_name, depth_img in depth_images.items():
            # Skip down camera
            if cam_name == "down":
                continue

            h, w = depth_img.shape

            # Subsample pixels
            u_grid, v_grid = np.meshgrid(
                np.arange(0, w, sample_step), np.arange(0, h, sample_step)
            )
            u = u_grid.flatten()
            v = v_grid.flatten()

            pixel_vals = depth_img[v, u]
            depths = (pixel_vals / 255.0) * max_depth_meters  # depth in meters

            # Ray directions in camera local frame (Z=fwd, X=right, Y=down)

            z_dir_unit = np.ones_like(depths)
            x_dir_unit = (u - cx) / f
            y_dir_unit = (v - cy) / f
            norms = np.sqrt(x_dir_unit**2 + y_dir_unit**2 + z_dir_unit**2)
            x_dir_unit, y_dir_unit, z_dir_unit = (
                x_dir_unit / norms,
                y_dir_unit / norms,
                z_dir_unit / norms,
            )

            # Camera axes in UAV local NED
            axis_f, axis_r, axis_d = cameras[cam_name]
            local_ray_dirs = (
                z_dir_unit[:, None] * np.array(axis_f)
                + x_dir_unit[:, None] * np.array(axis_r)
                + y_dir_unit[:, None] * np.array(axis_d)
            )

            # World ray directions
            world_ray_dirs = (np.array(rot_mat) @ local_ray_dirs.T).T

            # Calculate absolute world Z of all points at these depths
            z_world_samples = current_pos[2] + depths * world_ray_dirs[:, 2]

            # 1. Collect FREE points along rays
            free_dist_eligible = depths >= FREE_MIN_DISTANCE
            d_max = np.minimum(depths[free_dist_eligible], FREE_MAX_DISTANCE)
            sample_d = np.arange(1.0, FREE_MAX_DISTANCE + 1.0, self.res / 2)
            mask = sample_d[None, :] < d_max[:, None]
            ray_indices, sample_indices = np.where(mask)

            if len(ray_indices) > 0:
                free_world_dirs = world_ray_dirs[free_dist_eligible][ray_indices]
                free_depths = sample_d[sample_indices]
                free_world_pts = current_pos + free_world_dirs * free_depths[:, None]
                free_z_world = current_pos[2] + free_depths * free_world_dirs[:, 2]

                # Update all five grids for FREE points if they are in range
                for grid, z_range in [
                    (self.grid_lower, LOWER_GRID_Z_RANGE),
                    (self.grid_mid, MID_GRID_Z_RANGE),
                    (self.grid_upper, UPPER_GRID_Z_RANGE),
                    (self.grid_upper_2, UPPER_2_GRID_Z_RANGE),
                    (self.grid_upper_3, UPPER_3_GRID_Z_RANGE),
                ]:
                    mask_in_range = (free_z_world >= z_range[0]) & (
                        free_z_world < z_range[1]
                    )
                    if np.any(mask_in_range):
                        self.update_grid_vectorized(
                            free_world_pts[mask_in_range],
                            GridMark.FREE.value,
                            grid,
                            only_unknown=True,
                        )

            # 2. Collect OBSTACLE points
            obs_mask = (depths > OBS_MIN_DISTANCE) & (depths < OBS_MAX_DISTANCE)

            for grid, z_range in [
                (self.grid_lower, LOWER_GRID_Z_RANGE),
                (self.grid_mid, MID_GRID_Z_RANGE),
                (self.grid_upper, UPPER_GRID_Z_RANGE),
                (self.grid_upper_2, UPPER_2_GRID_Z_RANGE),
                (self.grid_upper_3, UPPER_3_GRID_Z_RANGE),
            ]:
                in_height_range = (z_world_samples >= z_range[0]) & (
                    z_world_samples < z_range[1]
                )

                should_be_obs = obs_mask & in_height_range
                should_be_free = obs_mask & (~in_height_range)

                if np.any(should_be_obs):
                    obs_depths = depths[should_be_obs]
                    obs_world_dirs = world_ray_dirs[should_be_obs]
                    self.update_grid_vectorized(
                        current_pos + obs_world_dirs * obs_depths[:, None],
                        GridMark.OBSTACLE.value,
                        grid,
                    )

                if np.any(should_be_free):
                    free_depths = depths[should_be_free]
                    free_world_dirs = world_ray_dirs[should_be_free]
                    # Note: for terminating points not in range, we mark as FREE only if UNKNOWN
                    self.update_grid_vectorized(
                        current_pos + free_world_dirs * free_depths[:, None],
                        GridMark.FREE.value,
                        grid,
                        only_unknown=True,
                    )

    def set_target_dir(self, yaw_degree, sigma=TARGET_DIR_SIGMA):
        """
        Set target direction and prepare target_value grid (2D grid of float)
        with values between 0 and 1 forming a "beam" in the target direction.
        """
        self.target_yaw = yaw_degree

        # 1. Create a coordinate grid centered at (0, 0) in world coordinates
        # Map indices to world distances relative to (0,0)
        x_centers = np.linspace(
            self.x_range[0] + self.res / 2, self.x_range[1] - self.res / 2, self.nx
        )
        y_centers = np.linspace(
            self.y_range[0] + self.res / 2, self.y_range[1] - self.res / 2, self.ny
        )
        xv, yv = np.meshgrid(x_centers, y_centers, indexing="ij")

        # 2. Convert target yaw to unit vector u = (ux, uy)
        # yaw 0 means towards +ve x, yaw 90 means towards +ve y (NED)
        theta = np.radians(yaw_degree)
        ux, uy = np.cos(theta), np.sin(theta)

        # 3. Calculate Perpendicular Distance using Cross Product magnitude
        # In 2D: |v x u| = |x*uy - y*ux|
        dist_perp = np.abs(xv * uy - yv * ux)

        # 4. Calculate Distance along the line using Dot Product
        # v . u = x*ux + y*uy
        dist_parallel = xv * ux + yv * uy

        # 5. Scoring Logic
        # Gaussian decay for perpendicular distance (keeps the "beam" constant width)
        score_perp = np.exp(-(dist_perp**2) / (2 * sigma**2))

        # Directional weighting:
        # Points in front (dist_parallel > 0) get full score.
        # Points behind (dist_parallel < 0) get penalized.
        score_dir = np.where(dist_parallel >= 0, 1.0, np.exp(dist_parallel / sigma))

        # Total score
        self.target_value = (score_perp * score_dir).astype(np.float32)

    def get_frontier_v1(self, current_pos):
        """
        Frontier Cell: Free unvisited cell adjacent to at least one Unknown cell.
        Score: Number of unknown cells in a 5x5 kernel.
        Sampling: Sample based on probability proportional to score.
        Returns: (x, y) center coordinate of selected frontier cell or None.
        """
        grid = self._get_grid_for_z(current_pos[2])
        unknown_mask = (grid == GridMark.UNKNOWN.value).astype(np.float32)
        free_unvisited_mask = (
            (grid == GridMark.FREE.value) & (~self.visited_grid)
        ).astype(np.uint8)

        # Find cells adjacent to Unknown (using 3x3 dilation)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        has_unknown_neighbor = cv2.dilate(unknown_mask.astype(np.uint8), kernel_3x3)
        candidates_mask = (free_unvisited_mask == 1) & (has_unknown_neighbor == 1)

        indices = np.argwhere(candidates_mask)
        if len(indices) == 0:
            return None

        # Scoring with 5x5 kernel
        kernel_5x5 = np.ones((5, 5), np.float32)
        scores = cv2.filter2D(
            unknown_mask, -1, kernel_5x5, borderType=cv2.BORDER_CONSTANT
        )

        candidate_scores = scores[candidates_mask]
        total_score = np.sum(candidate_scores)

        if total_score == 0:
            # If all candidates have 0 score (unlikely if they are adjacent to unknown), pick random
            idx = indices[random.randint(0, len(indices) - 1)]
        else:
            probs = candidate_scores / total_score
            idx_choice = np.random.choice(len(indices), p=probs)
            idx = indices[idx_choice]

        return self._get_center(idx[0], idx[1])

    def get_frontier_v2(self, current_pos, radius=25.0, min_radius=0.0):
        """
        Frontier Cell: Free unvisited cell within 25m radius.
        Score: 7x7 kernel with weights (Unknown: 2x, Free Unvisited: 1x, Free Visited: -1x, Obstacle: -2x).
        Sampling: Sample based on max(0, score).
        Returns: (x, y) center coordinate of selected frontier cell or None.
        """
        px, py = current_pos[:2]
        grid = self._get_grid_for_z(current_pos[2])

        # Weight grid
        weight_grid = np.zeros_like(grid, dtype=np.float32)
        weight_grid[grid == GridMark.UNKNOWN.value] += UNKNOWN_CELL_WEIGHT
        weight_grid[grid == GridMark.FREE.value] += FREE_CELL_WEIGHT
        weight_grid[self.visited_grid] += VISITED_CELL_WEIGHT
        weight_grid[grid == GridMark.OBSTACLE.value] += OBSTACLE_CELL_WEIGHT

        # Scoring
        kernel = np.ones(FRONTIER_SCORING_KERNEL_SIZE, np.float32)
        scores = cv2.filter2D(weight_grid, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Candidates: Free unvisited within 25m
        free_unvisited_mask = (grid == GridMark.FREE.value) & (~self.visited_grid)

        # Filter by radius
        x_centers = np.linspace(
            self.x_range[0] + self.res / 2, self.x_range[1] - self.res / 2, self.nx
        )
        y_centers = np.linspace(
            self.y_range[0] + self.res / 2, self.y_range[1] - self.res / 2, self.ny
        )
        xv, yv = np.meshgrid(x_centers, y_centers, indexing="ij")
        dist_sq = (xv - px) ** 2 + (yv - py) ** 2
        radius_mask = dist_sq <= radius**2
        min_radius_mask = dist_sq >= min_radius**2

        # Priority 1: Free unvisited within radius
        candidates_mask = free_unvisited_mask & radius_mask & min_radius_mask
        indices = np.argwhere(candidates_mask)

        if len(indices) == 0:
            # Priority 2: Free within radius
            candidates_mask = free_unvisited_mask & radius_mask & min_radius_mask
            indices = np.argwhere(candidates_mask)
            if len(indices) == 0:
                # Priority 3: Any point within radius
                candidates_mask = radius_mask & min_radius_mask
                indices = np.argwhere(candidates_mask)
                if len(indices) == 0:
                    return None

        candidate_scores = scores[candidates_mask]
        # Use max(0, score) for probability
        candidate_weights = np.maximum(0, candidate_scores)
        total_weight = np.sum(candidate_weights)

        if total_weight == 0:
            idx = indices[random.randint(0, len(indices) - 1)]
        else:
            probs = candidate_weights / total_weight
            idx_choice = np.random.choice(len(indices), p=probs)
            idx = indices[idx_choice]

        return self._get_center(idx[0], idx[1])

    def get_frontier_v3(
        self, current_pos, radius=25.0, min_radius=0.0, n=1
    ) -> list[tuple[float, float]]:
        """
        Returns a list of n frontier points.

        A frontier cell is defined as a free unvisited cell that is in the target direction area.

        Consider points within radius. If none found in target direction,
        return any free unvisited point within radius as a fallback.
        Score: 7x7 kernel with weights similar to get_frontier_v2, plus a bonus
               counting target direction points in a 15x15 neighborhood.
        Sampling: Sample based on max(0, score).

        Returns:
            A list of (x, y) center coordinates of selected frontier cells.
        """
        px, py = current_pos[:2]
        grid = self._get_grid_for_z(current_pos[2])

        # 1. Weight grid
        weight_grid = np.zeros_like(grid, dtype=np.float32)
        weight_grid[grid == GridMark.UNKNOWN.value] += UNKNOWN_CELL_WEIGHT
        weight_grid[grid == GridMark.FREE.value] += FREE_CELL_WEIGHT
        weight_grid[self.visited_grid] += VISITED_CELL_WEIGHT
        weight_grid[grid == GridMark.OBSTACLE.value] += OBSTACLE_CELL_WEIGHT

        # 2. Base scoring with a kernel
        kernel = np.ones(FRONTIER_SCORING_KERNEL_SIZE, np.float32)
        scores = cv2.filter2D(weight_grid, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # 3. Total scoring: Add target_value grid values directly
        total_scores = scores + TARGET_DIR_SCORE_WEIGHT * self.target_value

        # 4. Filter by radius
        x_centers = np.linspace(
            self.x_range[0] + self.res / 2, self.x_range[1] - self.res / 2, self.nx
        )
        y_centers = np.linspace(
            self.y_range[0] + self.res / 2, self.y_range[1] - self.res / 2, self.ny
        )
        xv, yv = np.meshgrid(x_centers, y_centers, indexing="ij")
        px, py = current_pos[0], current_pos[1]
        dist_sq = (xv - px) ** 2 + (yv - py) ** 2
        radial_mask = dist_sq <= radius**2
        min_radial_mask = dist_sq >= min_radius**2

        free_unvisited_mask = (grid == GridMark.FREE.value) & (~self.visited_grid)
        target_dir_mask = self.target_value > TARGET_MASK_VALUE_THRESHOLD
        # import pdb; pdb.set_trace()

        # Priority 1: Free Unvisited within radial range
        target_candidates_mask = free_unvisited_mask & radial_mask & min_radial_mask
        indices = np.argwhere(target_candidates_mask)

        if len(indices) == 0:
            # Priority 2: Any point within radius range (might be obstacle marked on map, but are navigable actually)
            any_candidates_mask = radial_mask & min_radial_mask
            indices = np.argwhere(any_candidates_mask)
            if len(indices) == 0:
                return []
            selected_mask = any_candidates_mask
            logger_prefix = "Any frontier"
        else:
            selected_mask = target_candidates_mask
            logger_prefix = "Free unvisited frontier"

        # 6. Sampling based on total_scores
        candidate_scores = total_scores[selected_mask]
        idxs, all_probs = self.sample_frontier_points(
            indices, candidate_scores, n, temperature=EXP_SAMPLING_TEMPERATURE
        )

        results = [self._get_center(idx[0], idx[1]) for idx in idxs]

        print_debug = False
        if print_debug:
            top_5_score_idx = np.argsort(total_scores.flatten())[-5:][::-1]
            top_5_score_coords = [
                np.unravel_index(idx, total_scores.shape) for idx in top_5_score_idx
            ]
            top_5_score_centers = [
                self._get_center(r, c) for r, c in top_5_score_coords
            ]
            top_5_score_centers = [(float(x), float(y)) for x, y in top_5_score_centers]
            top_5_score_values = [
                float(total_scores[r, c]) for r, c in top_5_score_coords
            ]

            top_5_cand_idx = np.argsort(candidate_scores)[-5:][::-1]
            top_5_cand_centers = [
                self._get_center(indices[i][0], indices[i][1]) for i in top_5_cand_idx
            ]
            top_5_cand_centers = [(float(x), float(y)) for x, y in top_5_cand_centers]
            top_5_cand_scores = [float(candidate_scores[i]) for i in top_5_cand_idx]

            # print(
            #     f"\ncurrent pos {current_pos}, n {n}, radius {min_radius}-{radius}, method {logger_prefix}"
            # )
            # print(
            #     f"top 5 scoring points {top_5_score_centers} with scores {top_5_score_values}"
            # )
            # print(
            #     f"top 5 candidate points {top_5_cand_centers} with scores {top_5_cand_scores}"
            # )
            # print(
            #     f"Selected frontier points: {[(float(x), float(y)) for x, y in results]}\n"
            # )

        self.frontier_candidate_indices = indices
        self.frontier_scores = candidate_scores
        self.frontier_probs = all_probs
        self.frontier_indices = idxs
        return results

    def sample_frontier_points(self, indices, scores, n, temperature):
        """
        Samples points using Softmax (exponential) scaling with temperature normalization.

        Args:
            indices: (N, 2) array of grid coordinates.
            scores: (N,) array of scores.
            n: Number of points to sample.
            temperature: Controls 'sharpness'. Lower = more biased to max.

        Returns:
            idxs: (actual_n, 2) array of sampled grid coordinates.
            probs: (N,) array of probabilities used for sampling.
        """
        if len(indices) == 0:
            return np.array([]), np.array([])

        # 1. Numerical Stability Trick: Subtract the max score
        # This prevents np.exp() from blowing up to infinity.
        shift_scores = scores - np.max(scores)

        # 2. Apply Exponential Scaling
        # Using the temperature to control the spread
        exp_scores = np.exp(shift_scores / temperature)

        # 3. Normalize to get probabilities
        probs = exp_scores / np.sum(exp_scores)

        # 4. Stochastic Sampling
        actual_n = min(n, len(indices))

        # Check if we have enough points with non-zero probability
        num_nonzero = np.count_nonzero(probs > 1e-10)

        if num_nonzero < actual_n:
            # Fallback to uniform if distribution is too sparse
            selected_idx = np.random.choice(len(indices), size=actual_n, replace=False)
        else:
            selected_idx = np.random.choice(
                len(indices), size=actual_n, replace=False, p=probs
            )

        return indices[selected_idx], probs

    def save_frontier_scores(self, filename, current_pos):
        """
        Plot and save frontier scores and probabilities side-by-side.
        """
        if self.frontier_scores is None or self.frontier_candidate_indices is None:
            return

        import matplotlib.pyplot as plt
        import os

        # Create maps
        score_map = np.zeros((self.nx, self.ny))
        prob_map = np.zeros((self.nx, self.ny))

        score_map[
            self.frontier_candidate_indices[:, 0], self.frontier_candidate_indices[:, 1]
        ] = self.frontier_scores
        prob_map[
            self.frontier_candidate_indices[:, 0], self.frontier_candidate_indices[:, 1]
        ] = self.frontier_probs

        # Bounding box
        min_ix = np.min(self.frontier_candidate_indices[:, 0])
        max_ix = np.max(self.frontier_candidate_indices[:, 0])
        min_iy = np.min(self.frontier_candidate_indices[:, 1])
        max_iy = np.max(self.frontier_candidate_indices[:, 1])

        # Add some padding
        pad = 5
        min_ix = max(0, min_ix - pad)
        max_ix = min(self.nx, max_ix + pad)
        min_iy = max(0, min_iy - pad)
        max_iy = min(self.ny, max_iy + pad)

        cropped_scores = score_map[min_ix:max_ix, min_iy:max_iy]
        cropped_probs = prob_map[min_ix:max_ix, min_iy:max_iy]

        # Extent for rulers: [y_min, y_max, x_min, x_max] (East, East, North, North)
        world_min_x = self.x_range[0] + min_ix * self.res
        world_max_x = self.x_range[0] + max_ix * self.res
        world_min_y = self.y_range[0] + min_iy * self.res
        world_max_y = self.y_range[0] + max_iy * self.res

        extent = [world_min_y, world_max_y, world_min_x, world_max_x]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        im1 = ax1.imshow(cropped_scores, extent=extent, origin="lower", cmap="viridis")
        ax1.set_title("Frontier Scores")
        ax1.set_xlabel("East (m)")
        ax1.set_ylabel("North (m)")
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(cropped_probs, extent=extent, origin="lower", cmap="magma")
        ax2.set_title("Frontier Probabilities")
        ax2.set_xlabel("East (m)")
        ax2.set_ylabel("North (m)")
        fig.colorbar(im2, ax=ax2)

        # Highlight chosen indices
        if self.frontier_indices is not None:
            chosen_worlds = [
                self._get_center(idx[0], idx[1]) for idx in self.frontier_indices
            ]
            for cx, cy in chosen_worlds:
                ax1.plot(
                    cy,
                    cx,
                    "ro",
                    markersize=5,
                    label="Chosen" if cx == chosen_worlds[0][0] else "",
                )
                ax2.plot(
                    cy,
                    cx,
                    "ro",
                    markersize=5,
                    label="Chosen" if cx == chosen_worlds[0][0] else "",
                )

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_image(self, filename, cur_pos, yaw_deg=0.0):
        """
        Save the current occupancy grid as a PNG image with colors.
        """
        img_final = self._get_colored_occupancy_grid(cur_pos, yaw_deg=yaw_deg)
        cv2.imwrite(filename, img_final)

    def save_all_occup_grid_images(self, filenames, cur_pos, yaw_deg=0.0):
        zs = [
            LOWER_MID_Z_THRESHOLD + 3,
            LOWER_MID_Z_THRESHOLD - 3,
            MID_UPPER_Z_THRESHOLD - 3,
            UPPER_UPPER_2_Z_THRESHOLD - 3,
            UPPER_2_UPPER_3_Z_THRESHOLD - 3,
        ]
        for filename, z in zip(filenames, zs):
            cur_pos[2] = z
            self.save_image(filename, cur_pos, yaw_deg=yaw_deg)

    def apply_visualization_layout(
        self,
        img: np.ndarray,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
        cur_pos: list[float],
        yaw_deg: float,
    ):
        """
        Scales, adds rulers, and optional overlays to the raw map image.

        Args:
            cur_pos and yaw_deg are current pose
            r0, r1, c0, c1 are the cropping bounds
        """
        return apply_visualization_layout_static(
            img, r0, r1, c0, c1, cur_pos, yaw_deg, self.x_range, self.y_range, self.res
        )

    def _get_colored_occupancy_grid(
        self,
        cur_pos: list[float],
        yaw_deg: float = 0.0,
    ):
        """
        Apply coloring to the occupancy grid and crop to minimum explored area.
        - UNKNOWN: Light Gray (180, 180, 180)
        - FREE: Green (0, 255, 0)
        - OBSTACLE: Black (0, 0, 0)
        - VISITED: Red (0, 0, 255)
        - POTENTIAL_TARGET: Yellow (0, 255, 255)
        """
        grid = self._get_grid_for_z(cur_pos[2])
        return _get_colored_grid_static(
            grid,
            self.visited_grid,
            getattr(self, "target_value", None),
            cur_pos,
            yaw_deg,
            self.x_range,
            self.y_range,
            self.res,
            self.side_in_meters,
        )


def apply_visualization_layout_static(
    img: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    cur_pos: list[float],
    yaw_deg: float,
    x_range,
    y_range,
    res,
):
    """
    Scales, adds rulers, and optional overlays to the raw map image.
    """
    h, w = img.shape[:2]

    # 1. Overlay drone pose and orientation
    # Drone position in grid indices
    cx = int((cur_pos[0] - x_range[0]) // res)
    cy = int((cur_pos[1] - y_range[0]) // res)

    if r0 <= cx < r1 and c0 <= cy < c1:
        # We use (150, 75, 0) Medium Dark Blue for drone position
        # Note: img is flipped so row 0 is North. Row index = (h - 1) - (x_offset)
        row_idx = h - 1 - (cx - r0)
        img[row_idx - 1 : row_idx + 2, cy - c0 - 1 : cy - c0 + 2] = (150, 75, 0)

        # Yaw direction (Yellow line)
        # Draw a line in the direction of yaw
        yaw_rad = np.radians(yaw_deg)
        for dist in [1, 2, 3]:  # units of res
            # the img is already flipped, so x grows up (North), y grows right (East)
            # Correct sign for North (+X) and East (+Y)
            fx = cur_pos[0] + dist * res * np.cos(yaw_rad)
            fy = cur_pos[1] + dist * res * np.sin(yaw_rad)
            fcx = int((fx - x_range[0]) // res)
            fcy = int((fy - y_range[0]) // res)
            if r0 <= fcx < r1 and c0 <= fcy < c1:
                frow_idx = h - 1 - (fcx - r0)
                img[frow_idx - 1 : frow_idx + 2, fcy - c0 - 1 : fcy - c0 + 2] = (
                    0,
                    255,
                    255,
                )  # Yellow

    # 2. Scaling
    min_size = 400
    scale = 1
    if h < min_size or w < min_size:
        scale = max(1, min_size // min(h, w))

    if scale > 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # 3. Add Rulers
    margin = 60
    h_f, w_f = img.shape[:2]
    img_ruler = np.zeros((h_f + margin, w_f + margin, 3), dtype=np.uint8)
    img_ruler[:, :] = (255, 255, 255)  # White background
    img_ruler[0:h_f, margin : margin + w_f] = img

    # World coordinates for ranges
    world_x_start = x_range[0] + r0 * res
    world_x_end = x_range[0] + r1 * res
    world_y_start = y_range[0] + c0 * res
    world_y_end = y_range[0] + c1 * res

    # Determine interval
    world_y_span = world_y_end - world_y_start
    world_x_span = world_x_end - world_x_start
    if world_y_span <= 20:
        y_interval = 5.0
    elif world_y_span <= 50:
        y_interval = 10.0
    elif world_y_span <= 100:
        y_interval = 20.0
    elif world_y_span <= 200:
        y_interval = 50.0
    else:
        y_interval = 100.0

    if world_x_span <= 20:
        x_interval = 5.0
    elif world_x_span <= 50:
        x_interval = 10.0
    elif world_x_span <= 100:
        x_interval = 20.0
    elif world_x_span <= 200:
        x_interval = 50.0
    else:
        x_interval = 100.0

    grid_color = (200, 200, 200)  # light gray

    # Horizontal ticks (y-axis / East) - at bottom margin
    # Note: In exploration.py, x is North, y is East.
    for y_val in np.arange(
        np.ceil(world_y_start / y_interval) * y_interval, world_y_end, y_interval
    ):
        col = margin + int(
            (y_val - world_y_start) / (world_y_end - world_y_start) * w_f
        )
        if margin <= col < margin + w_f:
            cv2.line(img_ruler, (col, 0), (col, h_f), grid_color, 1)
            cv2.line(img_ruler, (col, h_f), (col, h_f + 5), (0, 0, 0), 1)
            cv2.putText(
                img_ruler,
                f"{y_val:.0f}",
                (col - 10, h_f + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    # Vertical ticks (x-axis / North) - at left margin
    for x_val in np.arange(
        np.ceil(world_x_start / x_interval) * x_interval, world_x_end, x_interval
    ):
        # x grows towards bottom in numpy after flip, row 0 is North?
        # Actually, img_final was flipped, so r0 is bottom.
        # world_x_end is at top of img.
        row = int((world_x_end - x_val) / (world_x_end - world_x_start) * h_f)

        if 0 <= row < h_f:
            cv2.line(img_ruler, (margin, row), (margin + w_f, row), grid_color, 1)
            cv2.line(img_ruler, (margin - 5, row), (margin, row), (0, 0, 0), 1)
            cv2.putText(
                img_ruler,
                f"{x_val:.0f}",
                (margin - 50, row + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    # Labels
    cv2.putText(
        img_ruler,
        "East (m)",
        (margin + w_f - 80, h_f + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        img_ruler,
        "North (m)",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    return img_ruler


def _get_colored_grid_static(
    grid,
    visited_grid,
    target_value_grid,
    cur_pos,
    yaw_deg,
    grid_x_range,
    grid_y_range,
    res,
    side_in_meters: float,
    x_range=None,
    y_range=None,
    planned_path=None,
):
    """Get a colored grid image from a 2D occupancy grid."""
    if x_range is None or y_range is None:
        nx, ny = grid.shape
        cx = int((cur_pos[0] - grid_x_range[0]) // res)
        cy = int((cur_pos[1] - grid_y_range[0]) // res)

        fixed_width = True
        if fixed_width:
            half_width = int((side_in_meters / 2) // res)  #  meters on each side
        else:
            # 1. Find minimum square that is bounded by unknown cells
            start_hw = 10
            step = 10
            max_half_width = 60
            half_width = start_hw
            for hw in range(start_hw, max_half_width + 1, step):
                half_width = hw
                r0, r1 = cx - hw, cx + hw
                c0, c1 = cy - hw, cy + hw

                if r0 < 0 or r1 >= nx or c0 < 0 or c1 >= ny:
                    half_width = max(start_hw, hw - step)
                    break

                border_slices = [
                    np.s_[r0, c0 : c1 + 1],
                    np.s_[r1, c0 : c1 + 1],
                    np.s_[r0 : r1 + 1, c0],
                    np.s_[r0 : r1 + 1, c1],
                ]
                okay_to_crop = True
                for border_slice in border_slices:
                    if np.any(grid[border_slice] != GridMark.UNKNOWN.value) or np.any(
                        visited_grid[border_slice]
                    ):
                        okay_to_crop = False
                        break

                if okay_to_crop:
                    break

        r0, r1 = cx - half_width, cx + half_width
        c0, c1 = cy - half_width, cy + half_width

        r0 = max(0, r0)
        r1 = min(nx, r1)
        c0 = max(0, c0)
        c1 = min(ny, c1)

        grid_cropped = grid[r0:r1, c0:c1]
        visited_cropped = visited_grid[r0:r1, c0:c1]
    else:
        r0 = int((x_range[0] - grid_x_range[0]) // res)
        r1 = int((x_range[1] - grid_x_range[0]) // res)
        c0 = int((y_range[0] - grid_y_range[0]) // res)
        c1 = int((y_range[1] - grid_y_range[0]) // res)
        grid_cropped = grid[r0:r1, c0:c1]
        visited_cropped = visited_grid[r0:r1, c0:c1]

    h, w = grid_cropped.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Define colors (BGR)
    colors = {
        GridMark.UNKNOWN.value: (180, 180, 180),  # Light Gray
        GridMark.FREE.value: (0, 255, 0),  # Green
        GridMark.OBSTACLE.value: (0, 0, 0),  # Black
        "VISITED": (0, 0, 255),  # Red
        "POTENTIAL_TARGET": (255, 150, 255),  # Light pink
    }

    # Priority: Unknown, Free, Obstacle, Visited
    img[grid_cropped == GridMark.UNKNOWN.value] = colors[GridMark.UNKNOWN.value]
    img[grid_cropped == GridMark.FREE.value] = colors[GridMark.FREE.value]
    img[grid_cropped == GridMark.OBSTACLE.value] = colors[GridMark.OBSTACLE.value]
    img[visited_cropped] = colors["VISITED"]

    if target_value_grid is not None:
        target_value_cropped = target_value_grid[r0:r1, c0:c1]
        img[target_value_cropped > 0.99] = colors["POTENTIAL_TARGET"]

    # Draw planned path if provided
    if planned_path is not None and len(planned_path) > 1:
        # Convert path points to float indices within the cropped grid
        # Points are (x, y, z)
        path_pts = np.array(
            [p.to_list() if hasattr(p, "to_list") else p for p in planned_path]
        )
        path_x = path_pts[:, 0]
        path_y = path_pts[:, 1]

        # Convert to grid indices (ix, iy)
        ix = (path_x - grid_x_range[0]) / res
        iy = (path_y - grid_y_range[0]) / res

        # Convert to indices relative to r0, c0
        ix_rel = ix - r0
        iy_rel = iy - c0

        # Prepare points for cv2.polylines: (column, row) which is (iy_rel, ix_rel)
        # Note: In _get_colored_grid_static, r0/r1 corresponds to X, c0/c1 corresponds to Y
        # img is (h, w, 3) where h is dx, w is dy.
        # cv2.polylines takes points as (x, y) where x is horizontal (column/Y), y is vertical (row/X)
        pts = np.vstack((iy_rel, ix_rel)).T.astype(np.int32)

        # Filter points within cropped region
        # h, w = grid_cropped.shape (nx_cropped, ny_cropped)
        # However, cv2.polylines draws on the image.
        # Points outside the image bounds are handled by cv2.polylines (clipped).
        cv2.polylines(
            img, [pts], isClosed=False, color=(255, 255, 0), thickness=2
        )  # Yellow in BGR? Wait, colors dict above: (180, 180, 180).
        # Actually (255, 255, 0) is Cyan in BGR. Cyan is (255, 255, 0).
        # Let's use Yellow: (0, 255, 255) in BGR.
        cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 255), thickness=1)

    # Flip to make North (+X) UP
    img = cv2.flip(img, 0)

    return apply_visualization_layout_static(
        img, r0, r1, c0, c1, cur_pos, yaw_deg, grid_x_range, grid_y_range, res
    )


class ThreeDimMap:
    _ray_template_cache: dict[
        tuple[str, int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    _Z_SHIFT_TRIGGER_RATIO = 0.08
    _Z_SHIFT_SIZE_RATIO = 0.25

    def __init__(
        self,
        x_range,
        y_range,
        z_range,
        res_xy,
        res_z,
        cam_prop: CamProp,
        config: MapConfig,
    ):
        self.res = res_xy
        self.res_z = res_z
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.cam_prop = cam_prop
        self.config = config

        # Calculate grid dimensions
        self.nx = int((self.x_range[1] - self.x_range[0]) // self.res)
        self.ny = int((self.y_range[1] - self.y_range[0]) // self.res)
        self.nz = int((self.z_range[1] - self.z_range[0]) // self.res_z)

        # 3D occupancy grid
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)
        self.grid.fill(GridMark.UNKNOWN.value)

        # 2D visited grid
        self.visited_grid = np.zeros((self.nx, self.ny), dtype=bool)

        # 2D target direction
        self.target_points = []
        self.target_points_marks = []
        self.last_target_point_idx = -1
        self.target_value = None
        self.target_yaw_deg = None
        self.target_vec = None
        self.best_potential_target_location = None

        # Path planning
        self.repeat_target_point_if_far = True
        self.planned_path = None
        self.all_path_points = set()
        self.f_scores_flat = None

        self._init_circle_mask()
        self._free_sample_d = np.arange(
            1.0,
            self.config.free_max_distance + 1.0,
            self.res,
            dtype=np.float32,
        )

    def _init_circle_mask(self):
        # Mark cells outside map range circle as obstacles (vectorized)
        x = np.linspace(self.x_range[0], self.x_range[1], self.nx, endpoint=False)
        y = np.linspace(self.y_range[0], self.y_range[1], self.ny, endpoint=False)
        xv, yv = np.meshgrid(x, y, indexing="ij")
        dist = np.sqrt(xv**2 + yv**2)
        self._circle_outside_mask = dist > MAP_MAX_RANGE
        self.grid[self._circle_outside_mask, :] = GridMark.OBSTACLE.value

    def _make_unknown_z_block(self, n_slices: int) -> np.ndarray:
        """Create an UNKNOWN z block while preserving outside-circle obstacle mask."""
        block = np.full(
            (self.nx, self.ny, n_slices), GridMark.UNKNOWN.value, dtype=self.grid.dtype
        )
        if n_slices > 0:
            block[self._circle_outside_mask, :] = GridMark.OBSTACLE.value
        return block

    def _shift_grid_z_window_if_needed(self, current_z: float):
        """Slide the fixed-size z-window when UAV gets too close to z-range boundaries."""
        z_min, z_max = self.z_range
        z_span = z_max - z_min
        if z_span <= 0 or self.nz <= 1:
            return

        lower_trigger = z_min + self._Z_SHIFT_TRIGGER_RATIO * z_span
        upper_trigger = z_max - self._Z_SHIFT_TRIGGER_RATIO * z_span

        shift_cells = max(1, int(self.nz * self._Z_SHIFT_SIZE_RATIO))
        shift_cells = min(shift_cells, self.nz - 1)
        if shift_cells <= 0:
            return

        shift_dz = shift_cells * self.res_z

        if current_z <= lower_trigger:
            pad = self._make_unknown_z_block(shift_cells)
            self.grid = np.concatenate((pad, self.grid[:, :, :-shift_cells]), axis=2)
            self.z_range = (z_min - shift_dz, z_max - shift_dz)
        elif current_z >= upper_trigger:
            pad = self._make_unknown_z_block(shift_cells)
            self.grid = np.concatenate((self.grid[:, :, shift_cells:], pad), axis=2)
            self.z_range = (z_min + shift_dz, z_max + shift_dz)

    def pos_to_idx(self, x, y, z) -> tuple[int, int, int]:
        ix = int((x - self.x_range[0]) // self.res)
        iy = int((y - self.y_range[0]) // self.res)
        iz = self.z_to_iz(z)
        return self._clamp_idx(ix, iy, iz)

    def x_to_ix(self, x) -> int:
        return int((x - self.x_range[0]) // self.res)
    
    def y_to_iy(self, y) -> int:
        return int((y - self.y_range[0]) // self.res)

    def z_to_iz(self, z) -> int:
        return int((z - self.z_range[0]) // self.res_z)

    def idx_to_pos(self, ix, iy, iz) -> Point3D:
        cx = round(self.x_range[0] + (ix + 0.5) * self.res, 2)
        cy = round(self.y_range[0] + (iy + 0.5) * self.res, 2)
        cz = round(self.z_range[0] + (iz + 0.5) * self.res_z, 2)
        return Point3D(cx, cy, cz)

    def _is_in_bounds(self, ix, iy, iz):
        return 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz

    def _clamp_idx(self, ix: int, iy: int, iz: int) -> tuple[int, int, int]:
        """Clamp a potentially out-of-bounds grid index to valid map bounds."""
        return (
            min(max(ix, 0), self.nx - 1),
            min(max(iy, 0), self.ny - 1),
            min(max(iz, 0), self.nz - 1),
        )

    def set_target_dir(self, yaw_degree: float):
        # list of 2D points in the direction of yaw_degree
        self.target_yaw_deg = yaw_degree
        self.target_vec = yaw_deg_to_vector(yaw_degree)
        self.target_points = [
            self.target_vec * i
            for i in np.arange(
                self.config.target_dir_dest_interval,
                MAP_MAX_RANGE + 1,
                self.config.target_dir_dest_interval,
            )
        ]
        self.target_points_marks = [TargetDirMark.UNVISITED] * len(self.target_points)

    def set_best_potential_target_location(
        self,
        point: Point3D | list[float] | tuple[float, float, float] | np.ndarray | None,
    ):
        if point is None:
            self.best_potential_target_location = None
            return

        if isinstance(point, Point3D):
            self.best_potential_target_location = point
            return

        self.best_potential_target_location = Point3D(*point)

    def is_nearby_area_visited(self, point, radius) -> bool:
        """Returns True if any x,y cell within the circular radius of the point is visited."""
        px, py = point[:2]
        ix = int((px - self.x_range[0]) // self.res)
        iy = int((py - self.y_range[0]) // self.res)
        radius_cells = int(np.ceil(radius / self.res))

        x_start = max(0, ix - radius_cells)
        x_end = min(self.nx, ix + radius_cells + 1)
        y_start = max(0, iy - radius_cells)
        y_end = min(self.ny, iy + radius_cells + 1)

        ix_range = np.arange(x_start, x_end)
        iy_range = np.arange(y_start, y_end)
        ixv, iyv = np.meshgrid(ix_range, iy_range, indexing="ij")

        cx = self.x_range[0] + (ixv + 0.5) * self.res
        cy = self.y_range[0] + (iyv + 0.5) * self.res

        dist_sq = (cx - px) ** 2 + (cy - py) ** 2
        mask = dist_sq <= radius**2

        sub_visited = self.visited_grid[x_start:x_end, y_start:y_end]
        return np.any(sub_visited[mask])

    def has_nearby_sphere_mark(
        self,
        center: Point3D | tuple[float, float, float] | list[float] | np.ndarray,
        radius: float,
        mark: int,
    ) -> bool:
        """Returns True if any x,y,z cell within the spherical radius of the point is marked with the given mark."""
        sub_grid, mask = self._get_sphere(center, radius)
        return np.any(sub_grid[mask] == mark)

    def frac_nearby_sphere_mark(self, center, radius, mark) -> float:
        """Return a value in [0, 1] representing the fraction of cells within a sphere that has given mark."""
        sub_grid, mask = self._get_sphere(center, radius)
        num_cells = np.sum(mask)
        num_mark = np.sum(sub_grid[mask] == mark)
        return num_mark / num_cells if num_cells > 0 else 0.0

    def _get_sphere(self, center, radius):
        """Returns a sub-grid and a spherical mask for the given center and radius."""
        if isinstance(center, Point3D):
            px, py, pz = center.x, center.y, center.z
        else:
            px, py, pz = float(center[0]), float(center[1]), float(center[2])

        ix, iy, iz = self.pos_to_idx(px, py, pz)
        radius_cells_xy = int(np.ceil(radius / self.res))
        radius_cells_z = int(np.ceil(radius / self.res_z))

        x_start = max(0, ix - radius_cells_xy)
        x_end = min(self.nx, ix + radius_cells_xy + 1)
        y_start = max(0, iy - radius_cells_xy)
        y_end = min(self.ny, iy + radius_cells_xy + 1)
        z_start = max(0, iz - radius_cells_z)
        z_end = min(self.nz, iz + radius_cells_z + 1)

        ix_range = np.arange(x_start, x_end)
        iy_range = np.arange(y_start, y_end)
        iz_range = np.arange(z_start, z_end)
        ixv, iyv, izv = np.meshgrid(ix_range, iy_range, iz_range, indexing="ij")

        cx = self.x_range[0] + (ixv + 0.5) * self.res
        cy = self.y_range[0] + (iyv + 0.5) * self.res
        cz = self.z_range[0] + (izv + 0.5) * self.res_z
        dist_sq = (cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2
        mask = dist_sq <= radius**2

        sub_grid = self.grid[x_start:x_end, y_start:y_end, z_start:z_end]
        return sub_grid, mask

    def is_free(self, point: tuple[float, float, float]) -> bool:
        """Returns True if the cell at the given point is free (not obstacle or unknown)."""
        ix, iy, iz = self.pos_to_idx(*point)
        return (
            self._is_in_bounds(ix, iy, iz)
            and self.grid[ix, iy, iz] == GridMark.FREE.value
        )

    def mark_path_visited(self, path: list[Point3D]):
        if path is None or len(path) == 0:
            return

        cur_pt = path[-1]
        if isinstance(cur_pt, Point3D):
            current_z = cur_pt.z
        else:
            current_z = float(cur_pt[2])
        self._shift_grid_z_window_if_needed(current_z)

        if len(path) == 1:
            ix, iy, iz = self.pos_to_idx(*path[0])
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                self.visited_grid[ix, iy] = True
            return

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            dist = p1.dist_to(p2)
            num_points = int(dist / (self.res / 2)) + 2
            pts = np.linspace(p1[:2], p2[:2], num_points)
            ix = ((pts[:, 0] - self.x_range[0]) // self.res).astype(int)
            iy = ((pts[:, 1] - self.y_range[0]) // self.res).astype(int)
            valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            self.visited_grid[ix[valid], iy[valid]] = True

        self.f_scores_flat = None  # erase f-scores as this is a new step

    def mark_obstacle_block(self, x_range, y_range, z_range):
        """Mark a block of cells as obstacles."""
        ix_min, iy_min, iz_min = self.pos_to_idx(x_range[0], y_range[0], z_range[0])
        ix_max, iy_max, iz_max = self.pos_to_idx(x_range[1], y_range[1], z_range[1])

        ix_min, iy_min, iz_min = self._clamp_idx(ix_min, iy_min, iz_min)
        ix_max, iy_max, iz_max = self._clamp_idx(ix_max, iy_max, iz_max)

        self.grid[ix_min : ix_max + 1, iy_min : iy_max + 1, iz_min : iz_max + 1] = (
            GridMark.OBSTACLE.value
        )

    def update_from_depth_images(
        self,
        current_pos,
        rot_mat,
        depth_images: dict[str, np.ndarray],
        sample_step: int | None = None,
    ):
        """
        Supports 6 cameras: front, back, left, right, up, down.
        """
        sample_step = (
            DEPTH_IMG_SAMPLE_STEP if sample_step is None else max(1, int(sample_step))
        )
        max_depth_meters = self.cam_prop.max_depth_meters
        rot_mat = np.asarray(rot_mat, dtype=np.float32)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        inv_res = 1.0 / self.res
        inv_res_z = 1.0 / self.res_z
        all_local_ray_dirs = []
        all_depths = []

        for cam_name, depth_img in depth_images.items():
            u, v, local_ray_dirs = self._get_ray_template(
                cam_name, sample_step, self.cam_prop
            )
            pixel_vals = depth_img[v, u]
            depths = (pixel_vals.astype(np.float32) / 255.0) * max_depth_meters

            all_local_ray_dirs.append(local_ray_dirs)
            all_depths.append(depths)

        if not all_local_ray_dirs:
            return

        local_ray_dirs = np.concatenate(all_local_ray_dirs, axis=0)
        depths = np.concatenate(all_depths, axis=0)
        world_ray_dirs = local_to_world(local_ray_dirs, rot_mat)
        # world_ray_dirs = (rot_mat @ local_ray_dirs.T).T

        # 1. FREE points along rays
        free_dist_eligible = depths >= self.config.free_min_distance
        if np.any(free_dist_eligible):
            d_max = np.minimum(
                depths[free_dist_eligible], self.config.free_max_distance
            )
            mask = self._free_sample_d[None, :] < d_max[:, None]
            ray_indices, sample_indices = np.where(mask)

            if len(ray_indices) > 0:
                free_world_dirs = world_ray_dirs[free_dist_eligible][ray_indices]
                free_depths = self._free_sample_d[sample_indices]
                free_world_pts = current_pos + free_world_dirs * free_depths[:, None]

                ix = ((free_world_pts[:, 0] - self.x_range[0]) * inv_res).astype(
                    np.int32
                )
                iy = ((free_world_pts[:, 1] - self.y_range[0]) * inv_res).astype(
                    np.int32
                )
                iz = ((free_world_pts[:, 2] - self.z_range[0]) * inv_res_z).astype(
                    np.int32
                )

                valid = (
                    (ix >= 0)
                    & (ix < self.nx)
                    & (iy >= 0)
                    & (iy < self.ny)
                    & (iz >= 0)
                    & (iz < self.nz)
                )
                # Only update if UNKNOWN
                target_indices = (ix[valid], iy[valid], iz[valid])
                if self.config.free_on_unknown_only:
                    mask_to_free = self.grid[target_indices] == GridMark.UNKNOWN.value
                else:
                    mask_to_free = np.ones(len(target_indices[0]), dtype=bool)

                self.grid[
                    target_indices[0][mask_to_free],
                    target_indices[1][mask_to_free],
                    target_indices[2][mask_to_free],
                ] = GridMark.FREE.value

        # 2. OBSTACLE points
        obs_mask = (depths > self.config.obs_min_distance) & (
            depths < self.config.obs_max_distance
        )
        if np.any(obs_mask):
            obs_world_pts = (
                current_pos + world_ray_dirs[obs_mask] * depths[obs_mask][:, None]
            )
            ix = ((obs_world_pts[:, 0] - self.x_range[0]) * inv_res).astype(np.int32)
            iy = ((obs_world_pts[:, 1] - self.y_range[0]) * inv_res).astype(np.int32)
            iz = ((obs_world_pts[:, 2] - self.z_range[0]) * inv_res_z).astype(np.int32)

            valid = (
                (ix >= 0)
                & (ix < self.nx)
                & (iy >= 0)
                & (iy < self.ny)
                & (iz >= 0)
                & (iz < self.nz)
            )
            self.grid[ix[valid], iy[valid], iz[valid]] = GridMark.OBSTACLE.value

    @classmethod
    def _get_ray_template(
        cls, cam_name: str, sample_step: int, cam_prop: CamProp
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return precomputed ray template for the given camera and image dimensions. Ray template consists of (u, v) pixel coordinates and corresponding local ray directions in camera frame.

        Assumes that FOV degree remains same everytime
        """
        h, w = cam_prop.img_height, cam_prop.img_width
        key = (cam_name, h, w, sample_step)
        cached = cls._ray_template_cache.get(key)
        if cached is not None:
            return cached

        cameras = get_cam_axes()

        u_grid, v_grid = np.meshgrid(
            np.arange(0, w, sample_step, dtype=np.int32),
            np.arange(0, h, sample_step, dtype=np.int32),
        )
        u = u_grid.reshape(-1)
        v = v_grid.reshape(-1)

        u_f = u.astype(np.float32)
        v_f = v.astype(np.float32)

        if cam_prop.fov_deg is not None:
            f, cx, cy = get_cam_intrinsic_params(cam_prop)
            z_dir_unit = np.ones_like(u_f, dtype=np.float32)
            x_dir_unit = (u_f - float(cx)) / float(f)
            y_dir_unit = (v_f - float(cy)) / float(f)
        else:
            fx, fy, cx, cy = get_cam_intrinsic_params_diff_aspect(cam_prop)
            z_dir_unit = np.ones_like(u_f, dtype=np.float32)
            x_dir_unit = (u_f - float(cx)) / float(fx)
            y_dir_unit = (v_f - float(cy)) / float(fy)

        norms = np.sqrt(x_dir_unit**2 + y_dir_unit**2 + z_dir_unit**2)
        x_dir_unit = x_dir_unit / norms
        y_dir_unit = y_dir_unit / norms
        z_dir_unit = z_dir_unit / norms

        axis_f, axis_r, axis_d = cameras[cam_name]
        axis_f = np.asarray(axis_f, dtype=np.float32)
        axis_r = np.asarray(axis_r, dtype=np.float32)
        axis_d = np.asarray(axis_d, dtype=np.float32)

        local_ray_dirs = (
            z_dir_unit[:, None] * axis_f
            + x_dir_unit[:, None] * axis_r
            + y_dir_unit[:, None] * axis_d
        ).astype(np.float32)

        cached = (u, v, local_ray_dirs)
        cls._ray_template_cache[key] = cached
        return cached

    def save_image(
        self,
        filename,
        cur_pos,
        side_in_meters,
        yaw_deg=0.0,
        z=None,
        x_range=None,
        y_range=None,
    ):
        if z is None:
            z = cur_pos[2]
        iz = self.z_to_iz(z)
        if 0 <= iz < self.nz:
            grid_slice = self.grid[:, :, iz]
            img_final = _get_colored_grid_static(
                grid_slice,
                self.visited_grid,
                self.target_value,
                cur_pos,
                yaw_deg,
                self.x_range,
                self.y_range,
                self.res,
                side_in_meters,
                x_range,
                y_range,
                planned_path=self.planned_path,
            )
            cv2.imwrite(filename, img_final)
        else:
            logger.error(
                f"Z coordinate {z} is out of range for ThreeDimMap. Image not saved."
            )

    def save_large_image(self, filename, cur_pos, side_in_meters, yaw_deg):
        """Saves a large image covering the entire visit area for the given z."""
        x, y, z = cur_pos[0], cur_pos[1], cur_pos[2]
        iz = self.z_to_iz(z)
        x_range = (-20, x + 20) if x >= 0 else (x - 20, 20)
        y_range = (-20, y + 20) if y >= 0 else (y - 20, 20)
        if 0 <= iz < self.nz:
            grid_slice = self.grid[:, :, iz]
            img_final = _get_colored_grid_static(
                grid_slice,
                self.visited_grid,
                self.target_value,
                cur_pos,
                yaw_deg,
                self.x_range,
                self.y_range,
                self.res,
                side_in_meters,
                x_range,
                y_range,
                planned_path=self.planned_path,
            )
            cv2.imwrite(filename, img_final)
        else:
            logger.error(
                f"Z coordinate {z} is out of range for ThreeDimMap. Image not saved."
            )

    def get_next_target_dir_dest(
        self, cur_pos_offset, z, retreat_mode: bool = False
    ) -> np.ndarray | None:
        """
        Returns first unvisited target dir point.

        Assumes that the UAV tried its best to reach the previous target point. If distance to previous target point is greater than 20 meters, returns it again for once (so 2 times in total) before moving to next target point.

        When iterating through target points, marks a target point as obstacle, unreached or visited.

        Supports retreat mode where it ignores target points and directly goes to the best potential target location. This is used when UAV reaches near end of map or very close to running out of timesteps.
        """
        if retreat_mode:
            if self.best_potential_target_location is None:
                return None

            return Point3D(
                self.best_potential_target_location.x,
                self.best_potential_target_location.y,
                z,
            )

        if not self.target_points:
            return None

        target_pt = self.target_points[self.last_target_point_idx]
        dist = np.linalg.norm(np.array(cur_pos_offset[:2]) - np.array(target_pt))
        if dist > 20 and self.repeat_target_point_if_far:
            start_idx = self.last_target_point_idx
            self.repeat_target_point_if_far = False
        else:
            start_idx = self.last_target_point_idx + 1
            self.repeat_target_point_if_far = True

        for i in range(start_idx, len(self.target_points)):
            if self.target_points_marks[i] == TargetDirMark.UNVISITED:
                pos_np = np.concatenate((self.target_points[i], [z]))
                pos = Point3D(*pos_np)
                if self.has_nearby_sphere_mark(pos, 0.4, GridMark.OBSTACLE.value):
                    self.target_points_marks[i] = TargetDirMark.OBSTACLE
                    continue

                if self.is_nearby_area_visited(pos, 2.0):
                    self.target_points_marks[i] = TargetDirMark.VISITED
                    continue

                if (
                    self.target_points_marks[self.last_target_point_idx]
                    == TargetDirMark.UNVISITED
                ):
                    self.target_points_marks[self.last_target_point_idx] = (
                        TargetDirMark.UNREACHED
                    )

                self.last_target_point_idx = i
                return pos

        # If all target points are visited, return None
        return None

    def get_next_target_dir_dest_for_physical_uav(
        self, cur_pos_offset: Point3D, z, skip_obstacle_dests: bool
    ):
        """
        Does not assume that this is only called after UAV tried its best to reach the previous target point. Instead, it allows returning the same target point multiple times upto k times until it is either marked as obstacle or visited.

        Does not return target dir point that is more than 10 meters away from current position.
        """
        start_idx = 0 if self.last_target_point_idx < 0 else self.last_target_point_idx
        for i in range(start_idx, len(self.target_points)):
            if self.target_points_marks[i] == TargetDirMark.UNVISITED:
                pos_np = np.concatenate((self.target_points[i], [z]))
                pos = Point3D(*pos_np)
                next_target_dir_pos = Point3D(
                    *(
                        self.target_points[i + 1]
                        if i + 1 < len(self.target_points)
                        else self.target_points[i]
                    ),
                    z,
                )
                if skip_obstacle_dests and self.has_nearby_sphere_mark(
                    pos, 0.3, GridMark.OBSTACLE.value
                ):
                    self.target_points_marks[i] = TargetDirMark.OBSTACLE
                    continue

                if self.is_nearby_area_visited(pos, 0.5):
                    self.target_points_marks[i] = TargetDirMark.VISITED
                    continue

                # Assumes that UAV will likely stay closer to the target dir points
                if (
                    cur_pos_offset.dist_to(pos) > 10
                    or cur_pos_offset.dist_to(next_target_dir_pos) > 10
                ):
                    self.last_target_point_idx = i
                    self.target_point_idx_repeated = 0
                    return pos

                if i == self.last_target_point_idx:
                    if self.target_point_idx_repeated >= 4:
                        self.target_points_marks[i] = TargetDirMark.UNREACHED
                        self.target_point_idx_repeated = 0
                        continue
                    else:
                        self.target_point_idx_repeated += 1
                else:
                    self.target_point_idx_repeated = 0

                self.last_target_point_idx = i
                return pos

    def save_frontier_scores(self, filename, cur_pos):
        """Dummy method for compatibility with closeloop_util.py"""
        pass

    def is_near_target_dir_line(
        self,
        point: Point3D | list[float] | np.ndarray,
        obj_type: Literal["target", "nearby"],
    ) -> tuple[bool, float]:
        """
        Check if a given point is within the threshold distance of any target direction points.
        """
        if obj_type == "nearby":
            threshold = self.config.target_dir_line_max_nearby_obj
        if obj_type == "target":
            if np.linalg.norm(np.array(point)) < 200:
                threshold = self.config.target_dir_line_max_near
            else:
                threshold = self.config.target_dir_line_max_far

        if not isinstance(point, Point3D):
            point = Point3D(*point)

        distance = perpendicular_distance(point, self.target_vec)
        if distance <= threshold:
            return True, distance

        return False, distance

    def plan_path(
        self, source: Point3D, destination: Point3D, budget: int
    ) -> tuple[list[Point3D], dict]:
        """Plan a path through 3D occupancy grid using A* search algorithm."""
        start_idx = self.pos_to_idx(*source.to_list())
        end_idx = self.pos_to_idx(*destination.to_list())
        path_log = {}

        if not self._is_in_bounds(*start_idx):
            return [source], path_log

        # Priority queue: (f_score, current_idx, current_direction)
        # direction is (dx, dy, dz)
        open_set = []
        heapq.heappush(open_set, (0, start_idx, (0, 0, 0)))

        # For visualization
        f_scores = np.zeros_like(self.grid, dtype=float)
        reg_ix = (
            min(start_idx[0], end_idx[0]) - 75,
            max(start_idx[0], end_idx[0]) + 75,
        )
        reg_iy = (
            min(start_idx[1], end_idx[1]) - 75,
            max(start_idx[1], end_idx[1]) + 75,
        )
        reg_iz = (min(start_idx[2], end_idx[2]) - 4, max(start_idx[2], end_idx[2]) + 4)

        f_scores_flat = []

        came_from = {}
        g_score = {start_idx: 0}

        # To handle the case where destination is unreachable
        best_idx = start_idx
        min_h = self._heuristic(start_idx, end_idx)

        # 10 Neighbors: 8 in-plane + 1 up + 1 down
        # neighbors_offsets = [
        #     (1, 0, 0),
        #     (-1, 0, 0),
        #     (0, 1, 0),
        #     (0, -1, 0),
        #     (1, 1, 0),
        #     (1, -1, 0),
        #     (-1, 1, 0),
        #     (-1, -1, 0),
        #     (0, 0, 1),
        #     (0, 0, -1),
        # ]
        neighbors_offsets = [
            (2, 0, 0),
            (-2, 0, 0),
            (0, 2, 0),
            (0, -2, 0),
            (2, 2, 0),
            (2, -2, 0),
            (-2, 2, 0),
            (-2, -2, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        start_idx = (start_idx[0] // 2 * 2, start_idx[1] // 2 * 2, start_idx[2])
        end_idx = (end_idx[0] // 2 * 2, end_idx[1] // 2 * 2, end_idx[2])

        # Limit search to avoid excessive computation on large maps
        max_iterations = 20000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            # print("it:", iterations)
            f, current, last_dir = heapq.heappop(open_set)
            # print("f", f, "current", current, "last_dir", last_dir)

            if current == end_idx:
                break

            for dx, dy, dz in neighbors_offsets:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                if not self._is_in_bounds(*neighbor):
                    continue

                if self.grid[neighbor] == GridMark.OBSTACLE.value:
                    continue

                # Distance cost (using meters for scale consistency)
                dist = np.sqrt(
                    (dx * self.res) ** 2 + (dy * self.res) ** 2 + (dz * self.res_z) ** 2
                )

                # Pentalty for going through already generated path points
                path_penalty = 0
                if neighbor in self.all_path_points:
                    path_penalty = 1.0 * self.res

                # Turn penalty to encourage smoothness
                turn_penalty = 0
                if last_dir != (0, 0, 0) and (dx, dy, dz) != last_dir:
                    # Heuristic penalty for changing direction
                    turn_penalty = 0.5 * self.res

                # Visited and obstacle penalty to discourage going near them
                visited_penalty = 0
                if self.is_nearby_area_visited(neighbor[:2], 0.4):
                    visited_penalty = 0.2 * self.res

                # closeby obstacle
                obstacle_penalty = 0
                if self.has_nearby_sphere_mark(neighbor, 0.4, GridMark.OBSTACLE.value):
                    obstacle_penalty = 0.2 * self.res

                # even discourage narrow aisle between obstacles
                frac_obstacle = self.frac_nearby_sphere_mark(
                    neighbor, 1.5, GridMark.OBSTACLE.value
                )
                obstacle_penalty += frac_obstacle * 0.8 * self.res

                # free bonus to encourage going through more open areas
                free_bonus = 0
                frac_free = self.frac_nearby_sphere_mark(
                    neighbor, 3, GridMark.FREE.value
                )
                free_bonus = frac_free * 0.2 * self.res

                tentative_g_score = (
                    g_score[current]
                    + dist
                    + path_penalty
                    + turn_penalty
                    + visited_penalty
                    + obstacle_penalty
                    - free_bonus
                )

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = (current, (dx, dy, dz))
                    g_score[neighbor] = tentative_g_score
                    h = self._heuristic(neighbor, end_idx)
                    f_score = tentative_g_score + h
                    # print("Pushing to open set:", f_score, neighbor, (dx, dy, dz))
                    heapq.heappush(open_set, (f_score, neighbor, (dx, dy, dz)))

                    f_scores[neighbor] = f_score

                    if h < min_h:
                        min_h = h
                        best_idx = neighbor

            if iterations % 2000 == 0:
                f_scores_flat.append(
                    (
                        iterations,
                        np.mean(
                            f_scores[
                                reg_ix[0] : reg_ix[1],
                                reg_iy[0] : reg_iy[1],
                                reg_iz[0] : reg_iz[1],
                            ],
                            2,
                        ),
                    )
                )

        # Reconstruct path
        path_indices = []
        curr = end_idx if end_idx in came_from else best_idx

        while curr != start_idx:
            path_indices.append(curr)
            if curr not in came_from:
                break
            curr = came_from[curr][0]

        path_indices.reverse()

        # Convert indices to positions
        path = [source]
        for idx in path_indices:
            path.append(self.idx_to_pos(*idx))

        # Truncate to budget
        if budget is not None and len(path) > budget:
            path = path[:budget]

        self.planned_path = path
        self.all_path_points.update((p.x, p.y, p.z) for p in path)

        path_log.update(
            {
                "length": len(path),
                "final_pos": self.idx_to_pos(*curr).to_list(),
                "path_end": path[-1].to_list(),
                "iterations": iterations,
                "best_h": min_h,
                "best_idx": self.idx_to_pos(*best_idx).to_list(),
            }
        )
        f_scores_flat.append(
            (
                iterations,
                np.mean(
                    f_scores[
                        reg_ix[0] : reg_ix[1],
                        reg_iy[0] : reg_iy[1],
                        reg_iz[0] : reg_iz[1],
                    ],
                    2,
                ),
            )
        )
        self.f_scores_flat = f_scores_flat

        return path, path_log

    def _get_plan_region(
        self,
        start_idx: tuple[int, int, int],
        end_idx: tuple[int, int, int],
        budget: int | None,
    ) -> tuple[int, int, int, int, int, int]:
        """Returns bounded local planning region around source/destination."""
        budget_steps = 60 if budget is None else max(1, int(budget))
        margin_xy = max(40, budget_steps * 2 + 20)
        margin_z = max(6, budget_steps // 10 + 4)

        x0 = max(0, min(start_idx[0], end_idx[0]) - margin_xy)
        x1 = min(self.nx, max(start_idx[0], end_idx[0]) + margin_xy + 1)
        y0 = max(0, min(start_idx[1], end_idx[1]) - margin_xy)
        y1 = min(self.ny, max(start_idx[1], end_idx[1]) + margin_xy + 1)
        z0 = max(0, min(start_idx[2], end_idx[2]) - margin_z)
        z1 = min(self.nz, max(start_idx[2], end_idx[2]) + margin_z + 1)

        return x0, x1, y0, y1, z0, z1

    def _compute_obstacle_edt(self, obstacle_mask: np.ndarray) -> np.ndarray | None:
        """Returns obstacle distance grid in meters for a local 3D region."""
        if distance_transform_edt is None:
            return None

        free_mask = ~obstacle_mask
        edt = distance_transform_edt(
            free_mask, sampling=(self.res, self.res, self.res_z)
        )
        if isinstance(edt, tuple):
            edt = edt[0]
        return np.asarray(edt, dtype=np.float32)

    def _compute_visited_nearby_mask(
        self, visited_mask_2d: np.ndarray
    ) -> np.ndarray | None:
        """Returns a boolean mask for cells within 0.4m of any visited x,y cell."""
        if distance_transform_edt is None:
            return None

        if not np.any(visited_mask_2d):
            return np.zeros_like(visited_mask_2d, dtype=bool)

        dist = distance_transform_edt(
            ~visited_mask_2d,
            sampling=(self.res, self.res),
        )
        if isinstance(dist, tuple):
            dist = dist[0]
        return np.asarray(dist) <= 0.4

    def plan_path_fast(
        self,
        source: Point3D,
        destination: Point3D,
        budget: int,
        include_visited_penalty: bool,
        include_path_penalty: bool,
        include_obstacle_avoidance: bool,
        heuristic_weight: float = 1.25,
    ) -> tuple[list[Point3D], dict]:
        """Faster weighted A* path planning with EDT-based obstacle/open-space cost."""
        ijmp = self.config.path_idx_jump
        path_log = {}
        start_idx = self.pos_to_idx(*source.to_list())
        end_idx = self.pos_to_idx(*destination.to_list())

        if not self._is_in_bounds(*start_idx):
            return [source], path_log

        start_idx = (
            start_idx[0] // ijmp * ijmp,
            start_idx[1] // ijmp * ijmp,
            start_idx[2],
        )
        end_idx = (end_idx[0] // ijmp * ijmp, end_idx[1] // ijmp * ijmp, end_idx[2])

        x0, x1, y0, y1, z0, z1 = self._get_plan_region(start_idx, end_idx, budget)
        local_grid = self.grid[x0:x1, y0:y1, z0:z1]
        local_visited = self.visited_grid[x0:x1, y0:y1]

        local_start = (start_idx[0] - x0, start_idx[1] - y0, start_idx[2] - z0)
        local_end = (end_idx[0] - x0, end_idx[1] - y0, end_idx[2] - z0)

        shape = local_grid.shape
        # print("local grid shape:", shape, "in meters:", [dim * self.res for dim in shape[:2]] + [shape[2] * self.res_z])
        sx, sy, sz = shape
        if sx <= 0 or sy <= 0 or sz <= 0:
            return [source], path_log

        obstacle_mask = local_grid == GridMark.OBSTACLE.value
        obstacle_edt = self._compute_obstacle_edt(obstacle_mask)
        visited_nearby_mask = self._compute_visited_nearby_mask(local_visited)

        reg_ix = (
            max(0, min(local_start[0], local_end[0]) - 75),
            min(sx, max(local_start[0], local_end[0]) + 76),
        )
        reg_iy = (
            max(0, min(local_start[1], local_end[1]) - 75),
            min(sy, max(local_start[1], local_end[1]) + 76),
        )
        reg_iz = (
            max(0, min(local_start[2], local_end[2]) - 4),
            min(sz, max(local_start[2], local_end[2]) + 5),
        )

        f_scores = np.zeros(shape, dtype=np.float32)
        f_scores_flat = []

        g_score = np.full(shape, np.inf, dtype=np.float32)
        closed = np.zeros(shape, dtype=bool)
        came_from = {}

        neighbors = [
            (ijmp, 0, 0),
            (-ijmp, 0, 0),
            (0, ijmp, 0),
            (0, -ijmp, 0),
            (ijmp, ijmp, 0),
            (ijmp, -ijmp, 0),
            (-ijmp, ijmp, 0),
            (-ijmp, -ijmp, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        neighbor_step_cost = [
            np.sqrt(
                (dx * self.res) ** 2 + (dy * self.res) ** 2 + (dz * self.res_z) ** 2
            )
            for dx, dy, dz in neighbors
        ]

        yz_size = sy * sz

        def to_flat(ix, iy, iz):
            return ix * yz_size + iy * sz + iz

        def from_flat(flat_idx):
            ix = flat_idx // yz_size
            rem = flat_idx % yz_size
            iy = rem // sz
            iz = rem % sz
            return ix, iy, iz

        start_flat = to_flat(*local_start)
        end_flat = to_flat(*local_end)

        def heuristic(idx):
            dx = (idx[0] - local_end[0]) * self.res
            dy = (idx[1] - local_end[1]) * self.res
            dz = (idx[2] - local_end[2]) * self.res_z
            return np.sqrt(dx * dx + dy * dy + dz * dz)

        path_idx_global = set()
        for p in self.all_path_points:
            ix, iy, iz = self.pos_to_idx(*p)
            if self._is_in_bounds(ix, iy, iz):
                path_idx_global.add((ix, iy, iz))

        max_iterations = self.config.max_path_plan_iterations
        iterations = 0

        g_score[local_start] = 0.0
        open_set = []
        h0 = heuristic(local_start)
        heapq.heappush(open_set, (h0 * heuristic_weight, start_flat, -1))

        best_flat = start_flat
        min_h = h0

        dist_near_obstacle_thr = self.config.dist_near_obstacle_thr
        path_penalty_val = 1.0 * self.res
        turn_penalty_val = 0.5 * self.res
        visited_penalty_val = 0.2 * self.res
        obstacle_near_penalty_val = 0.2 * self.res

        while open_set and iterations < max_iterations:
            iterations += 1
            _, cur_flat, last_dir_idx = heapq.heappop(open_set)
            cx, cy, cz = from_flat(cur_flat)

            if closed[cx, cy, cz]:
                continue
            closed[cx, cy, cz] = True

            if cur_flat == end_flat:
                break

            cur_g = g_score[cx, cy, cz]

            for ndir, (dx, dy, dz) in enumerate(neighbors):
                nx, ny, nz = cx + dx, cy + dy, cz + dz

                if not (0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz):
                    continue
                if obstacle_mask[nx, ny, nz] or closed[nx, ny, nz]:
                    continue

                global_neighbor = (nx + x0, ny + y0, nz + z0)

                path_penalty = (
                    path_penalty_val if global_neighbor in path_idx_global else 0.0
                )

                if not include_path_penalty:
                    path_penalty = 0.0

                turn_penalty = (
                    turn_penalty_val
                    if last_dir_idx != -1 and ndir != last_dir_idx
                    else 0.0
                )

                if visited_nearby_mask is not None:
                    visited_penalty = (
                        visited_penalty_val if visited_nearby_mask[nx, ny] else 0.0
                    )
                else:
                    visited_penalty = (
                        visited_penalty_val
                        if self.is_nearby_area_visited(
                            self.idx_to_pos(*global_neighbor),
                            self.config.dist_near_visited_thr,
                        )
                        else 0.0
                    )

                if not include_visited_penalty:
                    visited_penalty = 0.0

                if obstacle_edt is not None:
                    dist_to_obs = obstacle_edt[nx, ny, nz]
                    obstacle_penalty = (
                        obstacle_near_penalty_val
                        if dist_to_obs <= dist_near_obstacle_thr
                        else 0.0
                    )
                    obstacle_penalty += (
                        max(
                            0.0,
                            (self.config.obs_penalty_constant_factor - dist_to_obs)
                            / self.config.obs_penalty_constant_factor,
                        )
                        * 0.8
                        * self.res
                    )
                    free_bonus = (
                        min(dist_to_obs / self.config.free_bonus_constant_factor, 1.0)
                        * 0.2
                        * self.res
                    )
                else:
                    obstacle_penalty = 0.0
                    if self.has_nearby_sphere_mark(
                        self.idx_to_pos(*global_neighbor),
                        self.config.dist_near_obstacle_thr,
                        GridMark.OBSTACLE.value,
                    ):
                        obstacle_penalty += obstacle_near_penalty_val
                    obstacle_penalty += (
                        self.frac_nearby_sphere_mark(
                            self.idx_to_pos(*global_neighbor),
                            self.config.frac_obstacle_radius,
                            GridMark.OBSTACLE.value,
                        )
                        * 0.8
                        * self.res
                    )
                    free_bonus = (
                        self.frac_nearby_sphere_mark(
                            self.idx_to_pos(*global_neighbor),
                            self.config.frac_free_radius,
                            GridMark.FREE.value,
                        )
                        * 0.2
                        * self.res
                    )

                if not include_obstacle_avoidance:
                    obstacle_penalty = 0.0
                    free_bonus = 0.0

                tentative_g = (
                    cur_g
                    + neighbor_step_cost[ndir]
                    + path_penalty
                    + turn_penalty
                    + visited_penalty
                    + obstacle_penalty
                    - free_bonus
                )

                if tentative_g < g_score[nx, ny, nz]:
                    g_score[nx, ny, nz] = tentative_g
                    nflat = to_flat(nx, ny, nz)
                    came_from[nflat] = (cur_flat, ndir)

                    h = heuristic((nx, ny, nz))
                    f_score = tentative_g + heuristic_weight * h
                    f_scores[nx, ny, nz] = f_score
                    heapq.heappush(open_set, (f_score, nflat, ndir))

                    if h < min_h:
                        min_h = h
                        best_flat = nflat

            if iterations % (self.config.max_path_plan_iterations // 10) == 0:
                f_scores_flat.append(
                    (
                        iterations,
                        np.mean(
                            f_scores[
                                reg_ix[0] : reg_ix[1],
                                reg_iy[0] : reg_iy[1],
                                reg_iz[0] : reg_iz[1],
                            ],
                            axis=2,
                        ),
                    )
                )

        goal_reached = np.isfinite(g_score[local_end])
        curr_flat = end_flat if goal_reached else best_flat

        path_indices = []
        while curr_flat != start_flat:
            lx, ly, lz = from_flat(curr_flat)
            path_indices.append((lx + x0, ly + y0, lz + z0))
            if curr_flat not in came_from:
                break
            curr_flat = came_from[curr_flat][0]
        path_indices.reverse()

        path = [source]
        for idx in path_indices:
            path.append(self.idx_to_pos(*idx))

        if budget is not None and len(path) > budget:
            path = path[:budget]

        self.planned_path = path
        self.all_path_points.update((p.x, p.y, p.z) for p in path)

        final_flat = end_flat if goal_reached else best_flat
        fx, fy, fz = from_flat(final_flat)
        final_idx = (fx + x0, fy + y0, fz + z0)

        path_log.update(
            {
                "planner": "plan_path_fast",
                "length": len(path),
                "destination": destination.to_list(),
                "final_pos": self.idx_to_pos(*final_idx).to_list(),
                "path_end": path[-1].to_list(),
                "iterations": iterations,
                "best_h": float(min_h),
                "best_idx": self.idx_to_pos(*final_idx).to_list(),
                "goal_reached": bool(goal_reached),
                "heuristic_weight": heuristic_weight,
                "plan_region_shape": [int(sx), int(sy), int(sz)],
            }
        )

        f_scores_flat.append(
            (
                iterations,
                np.mean(
                    f_scores[
                        reg_ix[0] : reg_ix[1],
                        reg_iy[0] : reg_iy[1],
                        reg_iz[0] : reg_iz[1],
                    ],
                    axis=2,
                ),
            )
        )
        self.f_scores_flat = f_scores_flat

        return path, path_log

    def _heuristic(self, idx1, idx2):
        # Euclidean distance in meters
        p1 = self.idx_to_pos(*idx1)
        p2 = self.idx_to_pos(*idx2)
        return p1.dist_to(p2)

    def get_turning_points(self, path) -> list[tuple[int, list[float], bool]]:
        # path is a list of 3D points
        # return the point where the path turns
        turning_points = []
        epsilon = 0.05
        for i in range(1, len(path) - 1):
            # Calculate the angle between the current segment and the next segment
            v1 = np.array(path[i]) - np.array(path[i - 1])
            v2 = np.array(path[i + 1]) - np.array(path[i])
            val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            val = np.clip(val, -1.0, 1.0)  # Clip to handle numerical issues
            angle = np.arccos(val)
            if angle >= (np.pi / 4 - epsilon):
                turning_points.append((i, path[i]))

        # Deliberately add the last point as a turning point
        if not turning_points or turning_points[-1][0] != len(path) - 1:
            turning_points.append((len(path) - 1, path[-1]))

        tp_with_visited = []
        for tp in turning_points:
            if self.is_nearby_area_visited(tp[1], 0.4):
                tp_with_visited.append((tp[0], tp[1], True))
            else:
                tp_with_visited.append((tp[0], tp[1], False))

        return tp_with_visited

    def plot_path(self, path, x_range, y_range, z_range, filename):
        """
        Plots the path in 3D space with obstacles.

        x_range, y_range, z_range sets the bounds of the plot. Should be (min, max)

        """
        # 1. Plot Obstacles
        # Extract indices within range
        ix_min, iy_min, iz_min = self.pos_to_idx(x_range[0], y_range[0], z_range[0])
        ix_max, iy_max, iz_max = self.pos_to_idx(x_range[1], y_range[1], z_range[1])

        # Clamp to grid bounds
        ix_min, ix_max = max(0, ix_min), min(self.nx, ix_max)
        iy_min, iy_max = max(0, iy_min), min(self.ny, iy_max)
        iz_min, iz_max = max(0, iz_min), min(self.nz, iz_max)

        sub_grid = self.grid[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]
        obs_indices = np.argwhere(sub_grid == GridMark.OBSTACLE.value)

        def _configure_axes(ax):
            # default is (elev=30, azim=-60)
            # standard: (elev=30, azim=45)
            # xy plane view: (elev=90, azim=0)
            # side view (xz plane): (elev=0, azim=-90)
            # front view (yz plane): (elev=0, azim=0)
            ax.view_init(elev=90, azim=0)

            ax.set_xlabel("North (m)")
            ax.set_ylabel("East (m)")
            ax.set_zlabel("Down (m)")
            ax.set_title("3D Map Path Visualization")

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            # In NED, Z is positive downwards. So, plotting z +ve to -ve.
            ax.set_zlim(z_range[::-1])
            ax.grid(True)

        def _plot_obstacles(ax):
            if len(obs_indices) > 0:
                # Shift indices back to global
                plot_obs_indices = obs_indices.copy()
                plot_obs_indices[:, 0] += ix_min
                plot_obs_indices[:, 1] += iy_min
                plot_obs_indices[:, 2] += iz_min

                # Convert to world coordinates (centers)
                # bar3d takes (x, y, z, dx, dy, dz)
                # x, y, z should be the *starting* point of the bar, not the center.
                # idx_to_pos gives the center of the cell.
                # So start point = center - res/2
                x_starts = self.x_range[0] + plot_obs_indices[:, 0] * self.res
                y_starts = self.y_range[0] + plot_obs_indices[:, 1] * self.res
                z_starts = self.z_range[0] + plot_obs_indices[:, 2] * self.res_z

                ax.bar3d(
                    x_starts,
                    y_starts,
                    z_starts,
                    self.res,
                    self.res,
                    self.res_z,
                    color="red",
                    alpha=0.1,
                    label="Obstacle",
                )

        # 1) Save obstacles-only plot
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection="3d")
        # _configure_axes(ax)
        # _plot_obstacles(ax)

        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # if by_label:
        #     ax.legend(by_label.values(), by_label.keys())

        # file_stem, file_ext = os.path.splitext(filename)
        # obstacles_only_filename = f"{file_stem}_obstacles{file_ext}"
        # plt.savefig(obstacles_only_filename)
        # plt.close(fig)

        # 2) Save obstacles + path plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        _configure_axes(ax)
        _plot_obstacles(ax)

        # Plot Path
        if path is not None and len(path) > 0:
            path_pts = np.array(path)
            ax.plot(
                path_pts[:, 0],
                path_pts[:, 1],
                path_pts[:, 2],
                color="red",
                linewidth=2,
                label="Path",
            )
            # Mark start and end
            ax.scatter(
                path_pts[0, 0],
                path_pts[0, 1],
                path_pts[0, 2],
                color="green",
                s=100,
                label="Start",
            )
            ax.scatter(
                path_pts[-1, 0],
                path_pts[-1, 1],
                path_pts[-1, 2],
                color="blue",
                s=100,
                label="Target",
            )

        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        plt.savefig(filename)
        plt.close(fig)

    def plot_path_paper(self, path, visited_path, safety_points, x_range, y_range, z_range, filename, elev, azim, plot_obstacle_only=True):
        """
        Plots the path in 3D space with obstacles.

        x_range, y_range, z_range sets the bounds of the plot. Should be (min, max)

        """
        # 1. Plot Obstacles
        # Extract indices within range
        ix_min, iy_min, iz_min = self.pos_to_idx(x_range[0], y_range[0], z_range[0])
        ix_max, iy_max, iz_max = self.pos_to_idx(x_range[1], y_range[1], z_range[1])

        # Clamp to grid bounds
        ix_min, ix_max = max(0, ix_min), min(self.nx, ix_max)
        iy_min, iy_max = max(0, iy_min), min(self.ny, iy_max)
        iz_min, iz_max = max(0, iz_min), min(self.nz, iz_max)

        sub_grid = self.grid[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]
        obs_indices = np.argwhere(sub_grid == GridMark.OBSTACLE.value)

        def _configure_axes(ax):
            ax.view_init(elev=elev, azim=azim)
            
            # ax.set_xlabel("North (m)")
            # ax.set_ylabel("East (m)")
            # ax.set_zlabel("Down (m)")

            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # In NED, Z is positive downwards. So, plotting z +ve to -ve.
            ax.set_zlim(z_range[::-1])
            ax.grid(True)
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.set_zticks([])

        def _plot_obstacles(ax):
            if len(obs_indices) > 0:
                # Shift indices back to global
                plot_obs_indices = obs_indices.copy()
                plot_obs_indices[:, 0] += ix_min
                plot_obs_indices[:, 1] += iy_min
                plot_obs_indices[:, 2] += iz_min

                # Convert to world coordinates (centers)
                # bar3d takes (x, y, z, dx, dy, dz)
                # x, y, z should be the *starting* point of the bar, not the center.
                # idx_to_pos gives the center of the cell.
                # So start point = center - res/2
                x_starts = self.x_range[0] + plot_obs_indices[:, 0] * self.res
                y_starts = self.y_range[0] + plot_obs_indices[:, 1] * self.res
                z_starts = self.z_range[0] + plot_obs_indices[:, 2] * self.res_z

                ax.bar3d(
                    x_starts,
                    y_starts,
                    z_starts,
                    self.res,
                    self.res,
                    self.res_z,
                    color="red",
                    alpha=0.1,
                    label="Obstacle",
                )

        
        # 1) Save obstacles-only plot
        if plot_obstacle_only:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            _configure_axes(ax)
            _plot_obstacles(ax)
            
            # Plot current point and visited path
            if path is not None and len(path) > 0:
                path_pts = np.array(path)
                ax.scatter(
                    path_pts[0, 0],
                    path_pts[0, 1],
                    path_pts[0, 2],
                    color="royalblue",
                    s=300,
                )
    
            if visited_path is not None and len(visited_path) > 0:
                visited_pts = np.array(visited_path)
                ax.plot(
                    visited_pts[:, 0],
                    visited_pts[:, 1],
                    visited_pts[:, 2],
                    color="royalblue",
                    linewidth=8,
                )

            file_stem, file_ext = os.path.splitext(filename)
            obstacles_only_filename = f"{file_stem}_obstacles{file_ext}"
            plt.savefig(obstacles_only_filename)
            plt.close(fig)

        # 2) Save obstacles + path plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        _configure_axes(ax)
        _plot_obstacles(ax)

        # Plot Path
        if path is not None and len(path) > 0:
            path_pts = np.array(path)
            ax.plot(
                path_pts[:, 0],
                path_pts[:, 1],
                path_pts[:, 2],
                color="gold",
                linewidth=8,
            )
            # Mark start and end
            ax.scatter(
                path_pts[0, 0],
                path_pts[0, 1],
                path_pts[0, 2],
                color="royalblue",
                s=300,
            )
            ax.scatter(
                path_pts[-1, 0],
                path_pts[-1, 1],
                path_pts[-1, 2],
                color="gold",
                s=300,
            )
            
        if visited_path is not None and len(visited_path) > 0:
            visited_pts = np.array(visited_path)
            ax.plot(
                visited_pts[:, 0],
                visited_pts[:, 1],
                visited_pts[:, 2],
                color="royalblue",
                linewidth=8,
            )
            
        if safety_points is not None and len(safety_points) > 0:
            safety_pts = np.array(safety_points)
            ax.plot(
                safety_pts[:, 0],
                safety_pts[:, 1],
                safety_pts[:, 2],
                color="limegreen",
                linewidth=8,
            )
            
            ax.scatter(
                safety_pts[-1, 0],
                safety_pts[-1, 1],
                safety_pts[-1, 2],
                color="limegreen",
                s=300,
            )

        # Remove duplicate labels in legend

        plt.savefig(filename)
        plt.close(fig)

    def save_grid_slice(self, cur_pos, filename, z=None):
        if z is None:
            z = cur_pos[2]

        iz = self.z_to_iz(z)

        if 0 <= iz < self.nz:
            grid_slice = self.grid[:, :, iz]
            raw_filename = filename.rsplit(".", 1)[0] + "_raw.npy"
            np.save(raw_filename, grid_slice)

            # Also save visited grid
            visited_filename = filename.rsplit(".", 1)[0] + "_visited.npy"
            np.save(visited_filename, self.visited_grid)

            # Also save planned path points
            if self.planned_path is not None:
                path_filename = filename.rsplit(".", 1)[0] + "_path.npy"
                path_points = np.array([p.to_list() for p in self.planned_path])
                np.save(path_filename, path_points)
        else:
            logger.error(
                f"Z coordinate {z} is out of range for ThreeDimMap. Image not saved."
            )

    def save_grid_3d(self, filename, x_range, y_range, z_range):
        """
        Saves a cropped 3D grid and visited grid within the specified x,y,z ranges, along with the planned path if available.

        Args:
            x_range, y_range, z_range: Tuples specifying the (min, max) bounds
        """
        ix0, iy0, iz0 = self.pos_to_idx(x_range[0], y_range[0], z_range[0])
        ix1, iy1, iz1 = self.pos_to_idx(x_range[1], y_range[1], z_range[1])

        grid_3d_cropped = self.grid[ix0:ix1, iy0:iy1, iz0:iz1]
        visited_cropped = self.visited_grid[ix0:ix1, iy0:iy1]

        base_filename = filename.rsplit(".", 1)[0]
        grid_filename = base_filename + "_raw_3d.npy"
        visited_filename = base_filename + "_visited.npy"

        np.save(grid_filename, grid_3d_cropped)
        np.save(visited_filename, visited_cropped)

        if self.planned_path is not None:
            path_filename = base_filename + "_path.npy"
            path_points = np.array([p.to_list() for p in self.planned_path])
            np.save(path_filename, path_points)


def get_colored_occupancy_grid(
    grid, potential_target, res, side_in_meters, cur_pos, visited_grid=None
):
    """
    Backward compatibility wrapper for tests.
    """
    # Assuming x_range and y_range based on grid shape and res if res is default
    # But usually this is called with specific grids.
    # To keep it simple, we'll use a dummy range that fits.
    nx, ny = grid.shape
    x_range = (-nx * res / 2, nx * res / 2)
    y_range = (-ny * res / 2, ny * res / 2)
    if visited_grid is None:
        visited_grid = np.zeros_like(grid, dtype=bool)

    return _get_colored_grid_static(
        grid,
        visited_grid,
        potential_target,
        cur_pos,
        0.0,
        x_range,
        y_range,
        res,
        side_in_meters,
    )
