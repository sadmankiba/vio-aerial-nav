import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.model_wrapper.base_model import BaseModelWrapper
from src.zero_shot.dir_follow_planner import (
    MAP_MAX_RANGE,
    MAX_THREAD_WORKERS,
    PathPlanner,
    RES_XY,
    RES_Z,
)
from src.vlnce_src.batch_state import EvalBatchState
from src.vlnce_src.closeloop_util import NUM_WAYPOINTS_TO_MOVE
from src.zero_shot.exploration import OPENUAV_MAP_CONFIG, ThreeDimMap
from src.zero_shot.llm_server.common_prompt import get_target_yaw_deg_from_instruction
from src.zero_shot.localize import Localizer
from src.zero_shot.navigation import prepare_pointnav_cubic_hermite_path

from utils.cam_util import OPENUAV_CAM_PROP
from utils.pos_util import Point3D, ori_to_yaw_deg, normalize_yaw_deg, world_to_local, local_to_world
from utils.safe_nav import HeightAdjust, HeightController, SafeNav


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DriftNav(BaseModelWrapper):
    """
    DriftNav assumes random odometry drift. The drift is imposed in batch_state.

    It does not aim to detect objects or reach a target location. Rather, it keeps moving towards the target direction while avoiding obstacles till max steps.
    """

    def __init__(
        self, scene_ids: list[str], instructions: list[str], batch_state: EvalBatchState
    ):
        super().__init__()
        self.batch_size = batch_state.batch_size
        self.scene_ids = scene_ids
        self.instructions = {
            scene_id: instruction
            for scene_id, instruction in zip(scene_ids, instructions)
        }
        self.batch_state = batch_state

        # Object inits for batch
        init_positions = self.batch_state.init_positions
        init_zs = [pos.z for pos in init_positions]

        x_range = (-MAP_MAX_RANGE - RES_XY / 2, MAP_MAX_RANGE + RES_XY / 2)
        y_range = (-MAP_MAX_RANGE - RES_XY / 2, MAP_MAX_RANGE + RES_XY / 2)

        self.occupancy_grids = {
            scene_id: ThreeDimMap(
                x_range=x_range,
                y_range=y_range,
                z_range=(init_z - 60, init_z + 40),
                res_xy=RES_XY,
                res_z=RES_Z,
                cam_prop=OPENUAV_CAM_PROP,
                config=OPENUAV_MAP_CONFIG,
            )
            for scene_id, init_z in zip(scene_ids, init_zs)
        }
        self.localizer = Localizer(self.batch_state, self.scene_ids)
        self.height_controller = HeightController()
        self.path_planner = PathPlanner(
            scene_ids, self.batch_state, self.occupancy_grids, self.height_controller
        )
        self.safe_nav = SafeNav()
        self.target_yaws = {}

        # Initialize height control for each scene
        for i, scene_id in enumerate(scene_ids):
            self.height_controller.add_scene(scene_id)

        # Initialize target directions from instructions
        init_orientations = self.batch_state.init_orientations
        for i, scene_id in enumerate(scene_ids):
            instruction = instructions[i]
            target_yaw_angle = get_target_yaw_deg_from_instruction(instruction)
            cur_yaw = ori_to_yaw_deg(init_orientations[i])
            self.target_yaws[scene_id] = normalize_yaw_deg(cur_yaw + target_yaw_angle)
            self.occupancy_grids[scene_id].set_target_dir(self.target_yaws[scene_id])

        # Runtime data
        self.active_indices = []
        self.active_scene_ids = []
        self.active_episodes = []
        self.active_step_indices = []
        self.active_cur_pos_offsets = []
        self.active_cur_pos_drift_offsets = []
        self.active_cur_rot_mats = []
        self.active_cur_depth_images = []
        self.active_last_path_offsets_drift = []

    def _depth_update_task(
        self,
        scene_id: str,
        cur_pos_offset: Point3D,
        cur_rot_mat,
        cur_depth_image,
    ):
        occup_grid = self.occupancy_grids[scene_id]
        occup_grid.update_from_depth_images(
            cur_pos_offset,
            cur_rot_mat,
            cur_depth_image,
        )

    def _set_active_states(self):
        skips = self.batch_state.skips
        episodes = self.batch_state.episodes
        step_indices = self.batch_state.step_indices
        cur_positions_drift = self.batch_state.cur_positions_drift
        cur_pos_offsets = self.batch_state.cur_pos_offsets
        cur_pos_drift_offsets = self.batch_state.cur_pos_drift_offsets
        cur_rot_mats = self.batch_state.cur_rot_mats
        cur_depth_images = self.batch_state.cur_depth_images
        last_path_offsets_drift = self.batch_state.last_path_offsets_drift
    
        self.active_indices = [i for i, skip in enumerate(skips) if not skip]
        self.active_episodes = [episodes[i] for i in self.active_indices]
        self.active_scene_ids = [self.scene_ids[i] for i in self.active_indices]
        self.active_step_indices = [step_indices[i] for i in self.active_indices]

        self.active_cur_positions_drift = [cur_positions_drift[i] for i in self.active_indices]
        self.active_cur_pos_offsets = [cur_pos_offsets[i] for i in self.active_indices]
        self.active_cur_pos_drift_offsets = [cur_pos_drift_offsets[i] for i in self.active_indices]
        self.active_cur_rot_mats = [cur_rot_mats[i] for i in self.active_indices]
        self.active_cur_depth_images = [
            cur_depth_images[i] for i in self.active_indices
        ]
        self.active_last_path_offsets_drift = [
            last_path_offsets_drift[i] for i in self.active_indices
        ]

    def _get_safe_dirs(self, active_cur_dest_offsets: dict[str, Point3D]):
        safe_dirs = []
        step_sizes = []
        safety_log_data = []
        for i, _ in enumerate(self.active_indices):
            scene_id = self.active_scene_ids[i]
            dest_vec_world = np.array(active_cur_dest_offsets[scene_id]) - np.array(
                self.active_cur_pos_drift_offsets[i]
            )
            dest_vec_local = world_to_local(dest_vec_world, self.active_cur_rot_mats[i])

            safe_dir, step_size, safe_region_log_data = (
                self.safe_nav.find_safe_direction_with_region(
                    dest_vec_local.tolist(),
                    self.active_cur_depth_images[i],
                    vertical_dir_preference="horizontal",
                    slower=False,
                    scene_id=scene_id,
                    height_controller=self.height_controller,
                )
            )
            safe_dirs.append(safe_dir)
            step_sizes.append(step_size)
            safety_log_data.append(safe_region_log_data)

        return safe_dirs, step_sizes, safety_log_data

    def run(self):
        self._set_active_states()
        log_data = [{} for _ in range(self.batch_size)]
        
        # Correct drift 
        correct_drift = False
        if correct_drift:        
            self.localizer.correct_drift(self.active_indices, self.active_scene_ids)
       
        # Print cur pos offsets with drift 
        for ia, i in enumerate(self.active_indices):
            logger.info(
                f"Episode {i} - Cur Pos Offset with Drift: {self.active_cur_pos_drift_offsets[ia]}, Total Drift: {self.active_cur_pos_drift_offsets[ia] - self.active_cur_pos_offsets[ia]}"
            )

        # Mark last path visited and update occupancy grid for active scenes
        for ia, _ in enumerate(self.active_indices):
            active_scene_id = self.active_scene_ids[ia]
            occup_grid = self.occupancy_grids[active_scene_id]
            occup_grid.mark_path_visited(self.active_last_path_offsets_drift[ia])

        # Update occupancy from depth images using thread pooling
        if len(self.active_indices) > 0:
            max_workers = min(len(self.active_indices), MAX_THREAD_WORKERS)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._depth_update_task,
                        self.active_scene_ids[ia],
                        self.active_cur_pos_drift_offsets[ia],
                        self.active_cur_rot_mats[ia],
                        self.active_cur_depth_images[ia],
                    )
                    for ia, _ in enumerate(self.active_indices)
                ]
                for future in as_completed(futures):
                    future.result()

        # Get destination points for active scenes
        height_adjusts = {
            scene_id: HeightAdjust.NO_ADJUST for scene_id in self.active_scene_ids
        }
        active_cur_dest_offsets, new_paths, path_logs = (
            self.path_planner.get_dest_offsets(
                self.active_indices, self.active_scene_ids, self.active_cur_pos_drift_offsets, height_adjusts
            )
        )

        # Get safe unit direction and step size
        safe_dirs, step_sizes, safety_log_data = self._get_safe_dirs(
            active_cur_dest_offsets
        )

        # Generate intermediate points
        all_waypoints: list[np.ndarray] = [
            np.array([self.batch_state.cur_positions_drift[i]] * 7)
            for i in range(self.batch_size)
        ]

        for ia, i in enumerate(self.active_indices):
            scene_id = self.active_scene_ids[ia]
            ep = self.active_episodes[ia]
            cur_pos = self.active_cur_positions_drift[ia]
            rot_mat = self.active_cur_rot_mats[ia]
            step_idx = self.active_step_indices[ia]
            safe_dir = safe_dirs[ia]
            step_size = step_sizes[ia]

            cur_obs = ep[-1]
            prev_obs = (
                ep[-NUM_WAYPOINTS_TO_MOVE - 1]
                if len(ep) > NUM_WAYPOINTS_TO_MOVE
                else cur_obs
            )

            dest_local_offset = np.array(safe_dir) * step_size
            dest_world_offset = local_to_world(dest_local_offset, rot_mat)

            path = prepare_pointnav_cubic_hermite_path(
                dest_world_offset,
                cur_obs,
                prev_obs,
                step_idx,
            )
            wps_world = np.array(cur_pos) + path
            all_waypoints[i] = wps_world

            # Log data similar to DirFollowPlanner (without profiling)
            log_data[i]["nav_short"] = {
                "safety_log_data": safety_log_data[ia],
                "next_waypoint": all_waypoints[i][NUM_WAYPOINTS_TO_MOVE - 1].tolist(),
                "desired_height": self.height_controller.desired_height.get(scene_id),
            }

            target_point_idx = self.occupancy_grids[scene_id].last_target_point_idx
            target_dir_point = None
            if (
                target_point_idx is not None
                and target_point_idx >= 0
                and target_point_idx < len(self.occupancy_grids[scene_id].target_points)
            ):
                target_dir_point = (
                    self.occupancy_grids[scene_id]
                    .target_points[target_point_idx]
                    .tolist()
                )

            log_data[i]["path_plan"] = {
                "destination": active_cur_dest_offsets[scene_id].to_list(),
                "target_point_idx": target_point_idx,
                "target_dir_point": target_dir_point,
            }
            if scene_id in new_paths:
                log_data[i]["path_plan"]["planned_path"] = [
                    p.to_list() for p in new_paths[scene_id]
                ]
            if scene_id in path_logs:
                log_data[i]["path_plan"]["path_log"] = path_logs[scene_id]

        return log_data, all_waypoints, self.occupancy_grids
