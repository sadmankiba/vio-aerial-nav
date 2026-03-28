import json
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vlnce_src.assist import Assist

from airsim_plugin.AirVLNSimulatorClientTool import NUM_WAYPOINTS_TO_MOVE
from src.vlnce_src.closeloop_util import (
    load_object_description,
    save_to_dataset_dagger,
    save_to_dataset_eval,
    target_distance_increasing_consecutively,
    target_distance_increasing_mostly,
    little_progress,
)
from src.zero_shot.llm_server.common_prompt import (
    get_orientation_text,
    get_relative_position,
)
from utils.pos_util import Point3D


### CONFIGURABLE ###
ORACLE_SUCCESS_EARLY_END_DIST_THRESHOLD = 45  # meters
####################

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EvalBatchState:
    def __init__(
        self,
        batch_size: int,
        env_batchs,
        env,
        assist,
        object_name_json_path: str,
        max_waypoints: int,
        eval_save_path: str,
    ):
        # Configurations
        self.batch_size = batch_size
        self.eval_env = env
        self.assist: Assist = assist
        self.object_name_json_path = object_name_json_path
        self.max_waypoints = max_waypoints
        self.eval_save_path = eval_save_path

        # Trajectory and target info
        self.episodes: list[list[dict]] = [
            [] for _ in range(batch_size)
        ]  # trajectory points for each env in batch
        self.scene_names: list[str] = [b["seq_name"] for b in env_batchs]
        self.init_positions: list[Point3D] = [
            Point3D(*b["trajectory"][0]["position"]) for b in env_batchs
        ]
        self.target_positions: list[list[float]] = [
            b["object_position"] for b in env_batchs
        ]
        self.object_infos: list[str] = [
            self._get_object_info(b) for b in env_batchs
        ]  # target object descriptions
        self.trajs: list[list[dict]] = [
            b["trajectory"] for b in env_batchs
        ]  # ground truth trajectories
        self.ori_data_dirs: list[str] = [b["trajectory_dir"] for b in env_batchs]

        # Runtime states
        self.dones = [False] * batch_size  # whether the UAV has reached the target
        self.done_reasons = [""] * batch_size  # reason for done
        self.predict_dones = [False] * batch_size
        self.collisions = [False] * batch_size
        self.has_collided = [False] * batch_size
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [
            False
        ] * batch_size  # whether the env ends early due without reaching target
        self.skips = [False] * batch_size  # whether target is reached and data is saved
        self.distance_to_ends: list[list[float]] = [[] for _ in range(batch_size)]
        self.envs_to_pause: list[int] = []  # indices of envs in batch that are finished

        logger.debug(f"Batch state object_infos: {self.object_infos}")

        self._initialize_batch_data()
    
    @property
    def init_orientations(self) -> list[np.ndarray]:
        """Returns the initial orientations of the UAV in the episodes."""
        init_orientations = [ep[0]["sensors"]["imu"]["orientation"] for ep in self.episodes]
        return init_orientations
    
    @property 
    def step_indices(self) -> list[int]:
        """Returns the current step indices of the UAV in the episodes."""
        return [len(ep) // NUM_WAYPOINTS_TO_MOVE for ep in self.episodes]
    
    @property 
    def cur_positions(self) -> list[Point3D]:
        """Returns the current positions of the UAV in the episodes."""
        cur_positions = [ep[-1]["sensors"]["state"]["position"] for ep in self.episodes]
        cur_positions = [Point3D(*pos) for pos in cur_positions]
        return cur_positions
    
    @property
    def cur_positions_drift(self) -> list[Point3D]:
        """Returns the current positions with drift of the UAV in the episodes."""
        cur_positions_drift = [ep[-1]["sensors"]["state"]["position_drift"] for ep in self.episodes]
        cur_positions_drift = [Point3D(*pos) for pos in cur_positions_drift]
        return cur_positions_drift
    
    @property 
    def cur_pos_error(self) -> list[Point3D]:
        """Returns the current position error of the UAV in the episodes."""
        cur_pos_error = []
        for ep in self.episodes:
            if len(ep) > 1:
                x_pos_error = (
                    ep[-1]["sensors"]["state"]["x_pos_error"] 
                    - ep[-NUM_WAYPOINTS_TO_MOVE - 1]["sensors"]["state"]["x_pos_error"]
                )
                y_pos_error = (
                    ep[-1]["sensors"]["state"]["y_pos_error"]
                    - ep[-NUM_WAYPOINTS_TO_MOVE - 1]["sensors"]["state"]["y_pos_error"]
                )
            else:
                x_pos_error = ep[-1]["sensors"]["state"]["x_pos_error"]
                y_pos_error = ep[-1]["sensors"]["state"]["y_pos_error"]
            cur_pos_error.append(Point3D(x_pos_error, y_pos_error, 0.0))
        return cur_pos_error

    @property
    def cur_pos_offsets(self) -> list[Point3D]:
        """Returns the current position offsets of the UAV from the initial position."""
        cur_pos_offsets = [
            get_relative_position(cur, init)
            for cur, init in zip(self.cur_positions, self.init_positions)
        ]
        cur_pos_offsets = [Point3D(*pos) for pos in cur_pos_offsets]
        return cur_pos_offsets
    
    @property
    def cur_pos_drift_offsets(self) -> list[Point3D]:
        """Returns the current position with drift offsets of the UAV from the initial position."""
        cur_pos_drift_offsets = [
            get_relative_position(cur, init)
            for cur, init in zip(self.cur_positions_drift, self.init_positions)
        ]
        cur_pos_drift_offsets = [Point3D(*pos) for pos in cur_pos_drift_offsets]
        return cur_pos_drift_offsets

    @property
    def cur_depth_images(self) -> list[dict]:
        """Returns depth image dictionaries for each episode."""
        result = []
        for ep in self.episodes:
            obs = ep[-1]
            depth = obs.get("depth")
            if isinstance(depth, dict):
                result.append(depth)
            elif "depth_image" in obs:
                result.append({"front": obs["depth_image"]})
            else:
                # Try sensors dict
                sensors = obs.get("sensors", {})
                depth = sensors.get("depth")
                if isinstance(depth, dict):
                    result.append(depth)
                else:
                    result.append({})
        return result

    @property
    def cur_down_depth_images(self) -> list[np.ndarray]:
        """Returns down camera depth images for each episode.

        Falls back to depth_image if depth["down"] is not present.
        """
        result = []
        for ep in self.episodes:
            obs = ep[-1]
            depth = obs.get("depth")
            if isinstance(depth, dict) and "down" in depth:
                result.append(depth["down"])
            else:
                result.append(None)
        return result

    @property
    def cur_observations(self) -> list[dict]:
        return [ep[-1] for ep in self.episodes]

    @property
    def cur_rot_mats(self) -> list[np.ndarray]:
        return [np.array(ep[-1]["sensors"]["imu"]["rotation"]) for ep in self.episodes]

    @property
    def last_path_offsets(self) -> list[list[Point3D]]:
        last_path_offsets = []
        for i, ep in enumerate(self.episodes):
            last_path = [
                Point3D(*obs["sensors"]["state"]["position"])
                for obs in ep[max(0, len(ep) - NUM_WAYPOINTS_TO_MOVE - 1) :]
            ]
            last_path_offset = [
                Point3D(*get_relative_position(pos, self.init_positions[i])) for pos in last_path
            ]
            last_path_offsets.append(last_path_offset)

        return last_path_offsets
    
    @property 
    def last_path_offsets_drift(self) -> list[list[Point3D]]:
        last_path_offsets_drift = []
        for i, ep in enumerate(self.episodes):
            last_path_drift = [
                Point3D(*obs["sensors"]["state"]["position_drift"])
                for obs in ep[max(0, len(ep) - NUM_WAYPOINTS_TO_MOVE - 1) :]
            ]
            last_path_offset_drift = [
                Point3D(*get_relative_position(pos, self.init_positions[i])) for pos in last_path_drift
            ]
            last_path_offsets_drift.append(last_path_offset_drift)

        return last_path_offsets_drift

    def _get_object_info(self, data_item: dict) -> str:
        """Return the object description for the target object in the data item."""
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(data_item["object"]["asset_name"].replace("AA", ""))

    def _load_object_description(self) -> dict[str, str]:
        with open(self.object_name_json_path, "r") as f:
            return {item["object_name"]: item["object_desc"] for item in json.load(f)}

    def _initialize_batch_data(self):
        """Store initial trajectory point and distance to target for each environment in batch."""
        outputs = self.eval_env.reset()
        observations, dones, self.collisions, self.oracle_success = [
            list(x) for x in zip(*outputs)
        ]
        for i in range(self.batch_size):
            if dones[i] and not self.dones[i]:
                self.dones[i] = True
                self.done_reasons[i] = "max_waypoints"
            
            if self.collisions[i] and not self.dones[i]:
                self.dones[i] = True
                self.done_reasons[i] = "collision"

        self.has_collided = [
            (self.collisions[i] or self.has_collided[i]) for i in range(self.batch_size)
        ]

        # Add last trajectory point to episodes and update distance to target for each environment
        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            self.distance_to_ends[i].append(
                self._calculate_distance(observations[i][-1], self.target_positions[i])
            )

            # Set position_drift to init position
            self.episodes[i][0]["sensors"]["state"]["position_drift"] = self.episodes[i][0]["sensors"]["state"]["position"]
            self.episodes[i][0]["sensors"]["state"]["x_accel_error"] = 0.0
            self.episodes[i][0]["sensors"]["state"]["y_accel_error"] = 0.0
            self.episodes[i][0]["sensors"]["state"]["x_pos_error"] = 0.0
            self.episodes[i][0]["sensors"]["state"]["y_pos_error"] = 0.0


    def _calculate_distance(self, observation, target_position) -> float:
        return np.linalg.norm(
            np.array(observation["sensors"]["state"]["position"])
            - np.array(target_position)
        )

    def update_from_env_output(self, outputs):
        """Save observations and update distance to target for each environment in batch"""
        observations, dones, self.collisions, self.oracle_success = [
            list(x) for x in zip(*outputs)
        ]  # for each env in batch
        
        for i in range(self.batch_size):
            if dones[i] and not self.dones[i]:
                self.dones[i] = True
                self.done_reasons[i] = "max_waypoints"

        if self.assist is not None:
            self.collisions, dones = self.assist.check_collision_by_depth(
                self.episodes, observations, self.collisions, self.dones
            )
            for i in range(self.batch_size):
                if dones[i] and not self.dones[i]:
                    self.dones[i] = True
                    self.done_reasons[i] = "collision"
                    
                if self.collisions[i] and not self.dones[i]:
                    self.dones[i] = True
                    self.done_reasons[i] = "collision"

        self.has_collided = [
            (self.collisions[i] or self.has_collided[i]) for i in range(self.batch_size)
        ]
        
        def get_accel_error(dt, accel_noise_density, accel_bias_stability) -> list[float]:
            steps = 25
            accel_white_noise = np.random.normal(0, accel_noise_density, steps)
            accel_bias_drift = np.cumsum(np.random.normal(0, accel_bias_stability, steps)) * dt
            
            return (accel_white_noise + accel_bias_drift)[4:25:5].tolist()

        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            
            dt = 0.46 
            accel_noise_density = 0.25 
            accel_bias_stability = 0.1
            
            x_accel_error = get_accel_error(dt, accel_noise_density, accel_bias_stability)
            y_accel_error = get_accel_error(dt, accel_noise_density, accel_bias_stability)
            
            x_accel_errors = [obs["sensors"]["state"]["x_accel_error"] for obs in self.episodes[i]] + x_accel_error
            y_accel_errors = [obs["sensors"]["state"]["y_accel_error"] for obs in self.episodes[i]] + y_accel_error
            x_vel_error = np.cumsum(x_accel_errors) * dt
            y_vel_error = np.cumsum(y_accel_errors) * dt
            x_pos_error = np.cumsum(x_vel_error) * dt
            y_pos_error = np.cumsum(y_vel_error) * dt
            x_pos_error_cur = x_pos_error[-5:]
            y_pos_error_cur = y_pos_error[-5:]
                    
            for j, observation in enumerate(observations[
                i
            ]):  # observations contains a list of NUM_WAYPOINTS_TO_MOVE (5) trajectory points
                # Add drift
                observation["sensors"]["state"]["x_accel_error"] = x_accel_error[j]
                observation["sensors"]["state"]["y_accel_error"] = y_accel_error[j]
                observation["sensors"]["state"]["x_pos_error"] = x_pos_error_cur[j]
                observation["sensors"]["state"]["y_pos_error"] = y_pos_error_cur[j]

                position = observation["sensors"]["state"]["position"]
                pos_drift = copy.copy(position)
                pos_drift[0] += x_pos_error_cur[j]
                pos_drift[1] += y_pos_error_cur[j]
                observation["sensors"]["state"]["position_drift"] = pos_drift
                
                self.episodes[i].append(observation)

            self.distance_to_ends[i].append(
                self._calculate_distance(observations[i][-1], self.target_positions[i])
            )
            
            # Original OpenUAV code used increasing_for_10frames. We replaced it with 
            # consecutively, mostly and little progress
            if target_distance_increasing_consecutively(self.distance_to_ends[i]) or target_distance_increasing_mostly(self.distance_to_ends[i]):
                self.dones[i] = True
                self.done_reasons[i] = "moving_away_from_target"
            
            if little_progress(self.distance_to_ends[i]):
                self.dones[i] = True
                self.done_reasons[i] = "little_progress"
            
            

    def get_rotation_to_targets(self) -> list[np.ndarray]:
        """Return rotation matrices to target for each environment in batch"""

        def rotation_matrix_from_vector(x: float, y: float) -> np.ndarray:
            """"""
            v_x = np.array([x, y, 0])
            v_x = v_x / np.linalg.norm(v_x)
            v_y = np.array([-v_x[1], v_x[0], 0])
            v_y = v_y / np.linalg.norm(v_y)
            v_z = np.array([0, 0, 1])
            rotation_matrix = np.column_stack((v_x, v_y, v_z))
            return rotation_matrix

        rotation_to_targets = []
        for observations, target_position in zip(self.episodes, self.target_positions):
            pos_init = np.array(observations[0]["sensors"]["state"]["position"])
            rot_init = np.array(observations[0]["sensors"]["imu"]["rotation"])
            target_position = np.array(rot_init.T @ (target_position - pos_init))

            x, y = target_position[0], target_position[1]
            rotation_to_target = rotation_matrix_from_vector(x, y)
            rotation_to_targets.append(rotation_to_target)

        return rotation_to_targets

    def get_assist_notices(self):
        if self.assist is not None:
            return self.assist.get_assist_notice(
                self.episodes, self.trajs, self.object_infos, self.target_positions
            )
        return [None] * self.batch_size

    def update_metric(self):
        """Update success and done status for each environment in batch"""
        for i in range(self.batch_size):
            if self.dones[i]:
                continue

            # If target is detected in this step
            if self.predict_dones[i] and not self.skips[i]:
                logger.debug(f"Scene {i}: Target detected")
                if (
                    self.distance_to_ends[i][-1] <= 20 and not self.early_end[i]
                ):  # drone reached near target (<20m)
                    self.success[i] = True
                    logger.debug(
                        f"Scene {i}: Drone has successfully reached the target, marking done=True"
                    )
                    self.dones[i] = True
                    self.done_reasons[i] = "success"
                # Matching original OpenUAV
                elif self.distance_to_ends[i][-1] > 20:  # if target is far away
                    # False positive. Ignored for oracle success metrics.
                    # Early end means if drone achieves oracle success, the episode will end as soon as it predicts done after that, otherwise the episode will continue
                    self.early_end[i] = True  
                
                if self.early_end[i] and self.oracle_success[i]:
                    logger.debug(
                        f"Scene {i}: Episode ending with oracle success. Setting done=True"
                    )
                    self.dones[i] = True
                    self.done_reasons[i] = "oracle_with_false_positive"

            if self.skips[i]:
                continue


    def print_runtime_status(self):
        """Print status"""
        status_str = ""
        if any(self.has_collided):
            status_str += f"Has Collided: {self.has_collided}, "
        if any(self.oracle_success):
            status_str += f"Oracle Success: {self.oracle_success}, "
        if any(self.dones):
            status_str += f"Dones: {self.dones}, "
        if any(self.predict_dones):
            status_str += f"Predict Dones: {self.predict_dones}, "
        if any(self.success):
            status_str += f"Success: {self.success}, "
        if any(self.early_end):
            status_str += f"Early End: {self.early_end}, "
        if any(self.skips):
            status_str += f"Skips: {self.skips}, "
        if any(self.envs_to_pause):
            status_str += f"Env to Pause: {self.envs_to_pause}, "
        if status_str:
            logger.debug(status_str)

    def get_status(self) -> pd.DataFrame:
        """Return a pandas DataFrame with the current status"""
        data = {
            "has_collided": self.has_collided,
            "oracle_success": self.oracle_success,
            "done": self.dones,
            "done_reason": self.done_reasons,
            "predict_done": self.predict_dones,
            "success": self.success,
            "early_end": self.early_end,
            "skip": self.skips,
            "paused": [(i in self.envs_to_pause) for i in range(self.batch_size)],
        }
        return pd.DataFrame(data)

    def get_progress(self) -> list[dict]:
        progress = []
        for i in range(self.batch_size):
            init_pos = self.episodes[i][0]["sensors"]["state"]["position"]
            cur_pos = self.episodes[i][-1]["sensors"]["state"]["position"]
            cur_ori = self.episodes[i][-1]["sensors"]["imu"]["orientation"]
            progress.append(
                {
                    "position": cur_pos,
                    "relative_position": get_relative_position(cur_pos, init_pos),
                    "target_position": self.target_positions[i],
                    "relative_target_position": get_relative_position(
                        self.target_positions[i], init_pos
                    ),
                    "distance": self.distance_to_ends[i][-1],
                    "orientation_dir": get_orientation_text(cur_ori),
                }
            )

        return progress

    def check_batch_termination(self, t: int) -> bool:
        """Save data for completed environments and check if all environments are done."""
        for i in range(self.batch_size):
            if t == self.max_waypoints:
                self.dones[i] = True

            # If done, mark successful if target is reached
            # And save trajectory data
            if (
                self.dones[i] or self.collisions[i]
            ) and not self.skips[i]:
                if self.collisions[i]:
                    self.dones[i] = True
                    self.done_reasons[i] = "collision"
                        

                self.envs_to_pause.append(i)
                prex = ""

                if self.success[i]:
                    prex = "success_"
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")

                new_traj_name = prex + self.ori_data_dirs[i].split("/")[-1]
                new_traj_dir = os.path.join(self.eval_save_path, new_traj_name)
                status_df = self.get_status()

                # Already saving after each step
                # save_to_dataset_eval(
                #     self.episodes[i], status_df.iloc[i], new_traj_dir, self.ori_data_dirs[i]
                # )
                self.skips[i] = True
                print(i, " has finished!")

        return np.array(self.skips).all()
