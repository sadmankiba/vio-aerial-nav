import random
import os
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add the project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.zero_shot.llm_server.common_prompt import get_relative_position
from utils.depth_util import get_distance_at_pixel_wrapped, get_ned_direction_from_pixel
from utils.cam_util import (
    CAM_NAMES,
    Camera,
    CamProp,
    closest_to_up_cam_pixel,
    wrap_pixel_coord,
    map_3d_point_to_img_pixel,
    OPENUAV_CAM_PROP,
)
from enum import Enum, auto

logging.basicConfig(level=logging.DEBUG)


### Configurable ###
STEP_SIZE = 4
STEP_SIZE_MIN_FORWARD_UP = 2.5
STEP_SIZE_MIN_DOWNWARD = 1.8  # gravity pulls, should go slowly downward
STEP_SIZE_MAX = 5.5

FIXED_HEIGHT = 10  # meters
TAKE_OFF_TARGET_HEIGHT = 18  # meters
TAKE_OFF_STEPS = 5  # steps
NEAR_GROUND_HEIGHT = 7.0  # meters

FORCE_TYPE = "non-linear"  # "linear" or "non-linear"
FORCE_WINDOW_SIZE_NON_FRONT = 128
FORCE_WINDOW_LENGTH_FRONT = 96
FORCE_WINDOW_STRIDE = 1
REPULSIVE_ALPHA = 1.3  # Exponent for non-linear force calculation
ATTRACTIVE_ALPHA = 1.1  # Exponent for non-linear force calculation
REPULSIVE_DIST_THRESHOLD = 15  # meters
REPULSIVE_DIST_FRONT_THRESHOLD = 10  # meters
ATTRACTIVE_DIST_THRESHOLD = 70  # meters

# scale = 1 / expected_norm_of_force * effect_factor_on_dest
REPULSIVE_FORCE_SCALE = (1 / 20) * 0.2  # Scale of the force to apply
ATTRACTIVE_FORCE_SCALE = (1 / 20) * 0.2  # Scale of the force to apply

CLEAR_POINT_OFFSET = 32
CLEAR_SQUARE_SIZE = 64
CLEAR_DOWN_DIST_MIN = 7
CLEAR_LOWER_DIST_MIN = 30
CLEAR_UPPER_DIST_MIN = 42

SAFE_OFFSET_DOWN_DIST_MIN = 9
SAFE_OFFSET_LOWER_DIST_MIN = 36
SAFE_OFFSET_UPPER_DIST_MIN = 48
LOWER_ALT_SAFE_REGION_OFFSET = 85
HIGHER_ALT_SAFE_REGION_OFFSET = 56

SAFE_DIR_PIXEL_OFFSET = 5
SAFE_DIR_PIXEL_SQ_SIZE = 64

# Retreat Mode 
RETREAT_HEIGHT_HIGH_ALT = 22
RETREAT_DIST_TO_DEST_THRESHOLD_HIGH_ALT = 15
####################


def _scaled_px(cam_prop: CamProp, base_px: int, min_px: int = 1) -> int:
    """Scales pixel-based constant values based on the camera resolution"""
    scale = max(cam_prop.img_width, cam_prop.img_height) / 256.0
    return max(min_px, int(round(base_px * scale)))

### Height Stuffs ###


class HeightAdjust(Enum):
    NO_ADJUST = auto()
    GO_LOWER = auto()
    GO_NEAR_GROUND = auto()


def get_height_from_ground(down_camera_depth_image: np.ndarray) -> float:
    """Returns the height from the ground in meters."""
    return np.median(down_camera_depth_image) / 255 * 100


def set_target_z_to_given_height(
    target_pos: list[float],
    cur_pos: list[float],
    down_cam_depth: np.ndarray,
    target_height: float,
):
    """Adjusts the target position's Z component to be at a given height from the ground."""
    cur_height = get_height_from_ground(down_cam_depth)
    diff = target_height - cur_height
    res_pos = target_pos.copy()
    res_pos[2] = (
        cur_pos[2] - diff
    )  # In NED, Z +ve downwards. To increase height, decrease Z.
    return res_pos


class HeightController:
    def __init__(self):
        self.desired_height = {}

    def add_scene(self, scene_id: str):
        """
        Sets the initial position and determines if the scene is high above ground.
        
        Should be called for each episode in the batch.
        """        
        self.desired_height[scene_id] = FIXED_HEIGHT
    
    def change_desired_height(self, scene_id: str, change: float):
        """Allows dynamically changing the desired height during the episode, with limits."""
        desired_height = self.desired_height.get(scene_id, FIXED_HEIGHT) + change 
        desired_height = np.clip(desired_height, FIXED_HEIGHT, FIXED_HEIGHT * 2)
        self.desired_height[scene_id] = float(desired_height)
        
    def get_desired_z(
        self,
        down_cam_depth: np.ndarray,
        cur_z: float,
        step_idx: int,
        scene_id: str,
        cur_pos: Optional[list[float]] = None,
        destination: Optional[list[float]] = None,
        retreat_mode: bool = False,
        height_adjust: HeightAdjust = HeightAdjust.NO_ADJUST,
    ) -> float:
        """
        Get desired z for fixed height based on current height
        
        Considers take off and high above ground. 
        """
        desired_height = self.desired_height.get(scene_id, FIXED_HEIGHT)
        
        # Override if necessary
        if height_adjust == HeightAdjust.GO_NEAR_GROUND:
            desired_height = NEAR_GROUND_HEIGHT
        elif height_adjust == HeightAdjust.GO_LOWER:
            desired_height = FIXED_HEIGHT - 2

        if retreat_mode and cur_pos is not None and destination is not None:
            cur_xy = np.array(cur_pos[:2], dtype=np.float32)
            dest_xy = np.array(destination[:2], dtype=np.float32)
            dist_xy = float(np.linalg.norm(cur_xy - dest_xy))
            if dist_xy > RETREAT_DIST_TO_DEST_THRESHOLD_HIGH_ALT:
                desired_height = RETREAT_HEIGHT_HIGH_ALT
        
        # In retreat mode, be at high altitude unless the UAV is close to destination.
        if step_idx is not None and not (retreat_mode and desired_height == RETREAT_HEIGHT_HIGH_ALT):
            # For take off, reach a higher altitude first, then come down.
            if step_idx < (2 / 3 * TAKE_OFF_STEPS):
                desired_height = 2 / 3 * TAKE_OFF_TARGET_HEIGHT
            elif step_idx < TAKE_OFF_STEPS:
                desired_height = TAKE_OFF_TARGET_HEIGHT
            elif step_idx < (4 / 3 * TAKE_OFF_STEPS):
                desired_height = 2 / 3 * TAKE_OFF_TARGET_HEIGHT
        
        height_from_ground = get_height_from_ground(down_cam_depth)

        diff = desired_height - height_from_ground
        desired_z = cur_z - diff
        
        return desired_z

    def control_height_local_target(
        self,
        down_camera_depth_image: np.ndarray,
        local_target_vec: np.ndarray,
        step_idx: Optional[int],
        height_adjust: HeightAdjust,
        given_height: Optional[float] = None,
        scene_id: Optional[str] = None,
        cur_z: Optional[float] = None,
    ) -> np.ndarray:
        """
        Adjusts the local_target_vec's Z component to control height from the ground.
        """
        if scene_id is not None and self.high_above_ground.get(scene_id, False):
            if cur_z is not None:
                # If high above ground, we want the desired global z to be equal to initial_z.
                # So the relative z should be initial_z - cur_z.
                desired_z = self.initial_z_map[scene_id] - cur_z
                local_target_vec[2] = desired_z
                return local_target_vec

        desired_height = FIXED_HEIGHT

        if height_adjust == HeightAdjust.GO_NEAR_GROUND:
            desired_height = NEAR_GROUND_HEIGHT
        elif height_adjust == HeightAdjust.GO_LOWER:
            desired_height = FIXED_HEIGHT - 2

        # For take off, reach a higher altitude first, then come down
        if step_idx is not None:
            if step_idx < (2 / 3 * TAKE_OFF_STEPS):
                desired_height = 2 / 3 * TAKE_OFF_TARGET_HEIGHT
            elif step_idx < TAKE_OFF_STEPS:
                desired_height = TAKE_OFF_TARGET_HEIGHT
            elif step_idx < (4 / 3 * TAKE_OFF_STEPS):
                desired_height = 2 / 3 * TAKE_OFF_TARGET_HEIGHT

        if given_height is not None:
            desired_height = given_height

        height_from_ground = get_height_from_ground(down_camera_depth_image)
        print("Height from ground:", height_from_ground)
        diff = desired_height - height_from_ground

        if height_adjust == HeightAdjust.GO_NEAR_GROUND:
            desired_z = -diff
        else:
            # Set z to target z or z for fixed height, whichever is lower
            desired_z = min(local_target_vec[2], -diff)

        # Do not change z too abruptly if local_target xy is close
        if np.linalg.norm(local_target_vec[:2]) < 4:
            desired_z = np.clip(desired_z, -8, 8)
        elif np.linalg.norm(local_target_vec[:2]) < 8:
            desired_z = np.clip(desired_z, -15, 15)

        local_target_vec[2] = desired_z
        return local_target_vec

    def control_height_global_target(
        self,
        down_camera_depth_image: np.ndarray,
        target_pos: list[float],
        cur_pos: list[float],
        height_adjust: HeightAdjust,
        given_height: Optional[float] = None,
        scene_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Adjusts the target position's Z component to be at a given height from the ground.

        The target coordinates are in global frame.
        """
        if scene_id is not None and self.high_above_ground.get(scene_id, False):
            res_pos = np.array(target_pos)
            res_pos[2] = self.initial_z_map[scene_id]
            return res_pos

        local_target_vec = np.array(target_pos) - np.array(cur_pos)
        local_target_vec_adjusted = self.control_height(
            down_camera_depth_image,
            local_target_vec,
            None,
            height_adjust=height_adjust,
            given_height=given_height,
            scene_id=scene_id,
        )
        return np.array(cur_pos) + local_target_vec_adjusted


######


def find_safe_direction_with_offset(
    point: list[float],
    depth_images: dict[str, np.ndarray],
    cam_prop: CamProp,
    threshold: float = 5.0,
) -> list[float]:
    """
    Find a safe direction to move an ego object given an initial direction to move to.

    Iterates over the pixels around the initial point in the image and checks if the distance to the object is greater than the threshold. Gradually increases pixel offset in all directions until a safe direction is found.

    Args:
        point: target point in UAV body frame (NED).
        depth_images: Depth images with keys [front, left, right, rear, down].
        img_width, img_height: Image dimensions.
        fov_deg: Field of view.
        threshold: Safety distance threshold in meters.

    Returns:
        list[float]: Safe unit 3D vector in UAV body frame (NED).
    """
    # Map initial point to all visible cameras
    pixel_coords = map_3d_point_to_img_pixel(point, cam_prop)

    # If not safe, start exploring from the first camera it's visible in
    if not pixel_coords:
        # If not visible in any, return original normalized vector and maximum distance
        p_vec = np.array(point)
        norm = np.linalg.norm(p_vec)
        return (p_vec / norm).tolist(), 100 if norm > 0 else [0.0, 0.0, 0.0], 0.0

    # Check if initial point is safe in any camera it's visible in
    initial_avg_dist = None
    safe_dir_pixel_sq_size = _scaled_px(cam_prop, SAFE_DIR_PIXEL_SQ_SIZE)

    for cam_name, (u, v) in pixel_coords.items():
        cam_check, u_check, v_check = cam_name, u, v
        if cam_name == "up":
            cam_check, (u_check, v_check) = closest_to_up_cam_pixel(u, v, cam_prop)

        avg_dist = get_distance_at_pixel_wrapped(
            depth_images,
            cam_check,
            u_check,
            v_check,
            cam_prop,
            max_square_size=safe_dir_pixel_sq_size,
        )
        initial_avg_dist = avg_dist
        if avg_dist >= threshold:
            p_vec = np.array(point)
            norm = np.linalg.norm(p_vec)
            if norm > 0:
                return (p_vec / norm).tolist(), avg_dist

            return [0.0, 0.0, 0.0], avg_dist

    start_cam = list(pixel_coords.keys())[0]
    start_u, start_v = pixel_coords[start_cam]
    if start_cam == "up":
        start_cam, (start_u, start_v) = closest_to_up_cam_pixel(
            start_u, start_v, cam_prop
        )

    max_offset = max(cam_prop.img_width, cam_prop.img_height)
    safe_dir_pixel_offset = _scaled_px(cam_prop, SAFE_DIR_PIXEL_OFFSET)
    for offset in range(safe_dir_pixel_offset, max_offset, safe_dir_pixel_offset):
        candidates = [
            (-offset, 0),
            (offset, 0),
            (0, -offset),
            (0, offset),
            (-offset, -offset),
            (offset, -offset),
            (-offset, offset),
            (offset, offset),
        ]
        random.shuffle(candidates)

        for du, dv in candidates:
            curr_cam, curr_u, curr_v = wrap_pixel_coord(
                start_cam, start_u + du, start_v + dv, cam_prop
            )
            if curr_cam == "up":
                continue
            avg_dist = get_distance_at_pixel_wrapped(
                depth_images,
                curr_cam,
                curr_u,
                curr_v,
                cam_prop,
                max_square_size=safe_dir_pixel_sq_size,
            )
            if avg_dist >= threshold:
                safe_vec = get_ned_direction_from_pixel(
                    curr_cam, curr_u, curr_v, cam_prop
                )
                return safe_vec.tolist(), avg_dist

    # If no safe direction found, return original (or zero vector as fallback)
    p_vec = np.array(point)
    norm = np.linalg.norm(p_vec)
    if norm > 0:
        return (p_vec / norm).tolist(), initial_avg_dist

    return [0.0, 0.0, 0.0], initial_avg_dist


def _get_min_max_avg_dist(img: np.ndarray, kernel_size: tuple[int, int]):
    stride = FORCE_WINDOW_STRIDE

    # Normalize depth to meters first to avoid precision issues with boxFilter
    dist_img = img.astype(np.float32) / 255.0 * 100.0
    # Use boxFilter to get averages for all possible windows
    # anchor=(0,0) means the window starts at the pixel coordinate
    anchor = (0, 0)
    avg_img = cv2.boxFilter(
        dist_img, -1, kernel_size, normalize=True, anchor=anchor
    )  # kernel = (width, height)

    # Valid averages are only where window fits in image
    valid_h = img.shape[0] - kernel_size[1] + 1
    valid_w = img.shape[1] - kernel_size[0] + 1
    if valid_h <= 0 or valid_w <= 0:
        avg = np.mean(dist_img)
        return avg, avg

    avg_img = avg_img[:valid_h, :valid_w]

    # Apply stride
    avg_img = avg_img[::stride, ::stride]

    return np.min(avg_img), np.max(avg_img)


def apply_forces(
    dest_vec: list[float],
    depth_images: dict[str, np.ndarray],
    cam_prop: CamProp,
) -> list[float]:
    """
    Applies repulsive and attractive forces to a point based on depth values in left/right/down direction.

    Args:
        dest_vec: destination vector in UAV body frame (NED)
        depth_images: dictionary of depth images
        img_width: width of images
        img_height: height of images
    """
    force_type = FORCE_TYPE
    window_nf = _scaled_px(cam_prop, FORCE_WINDOW_SIZE_NON_FRONT)
    window_f = _scaled_px(cam_prop, FORCE_WINDOW_LENGTH_FRONT)
    repulse_thres = REPULSIVE_DIST_THRESHOLD
    repulse_thres_front = REPULSIVE_DIST_FRONT_THRESHOLD
    attract_thres = ATTRACTIVE_DIST_THRESHOLD
    repulse_alpha = REPULSIVE_ALPHA
    attract_alpha = ATTRACTIVE_ALPHA
    repulse_scale = REPULSIVE_FORCE_SCALE
    attract_scale = ATTRACTIVE_FORCE_SCALE
    repulsive_force = np.array([0.0, 0.0, 0.0])
    attractive_force = np.array([0.0, 0.0, 0.0])
    log_data = {}
    log_data["dist"] = {}

    # Unit vectors for camera directions (forward axis in NED)
    cam_configs = {
        "left": np.array([0.0, -1.0, 0.0]),
        "right": np.array([0.0, 1.0, 0.0]),
        "down": np.array([0.0, 0.0, 1.0]),
    }

    # 1. Calculate forces from left, right, and down cameras
    for cam_name, unit_vec in cam_configs.items():
        if cam_name in depth_images:
            min_dist, max_dist = _get_min_max_avg_dist(
                depth_images[cam_name], kernel_size=(window_nf, window_nf)
            )
            log_data["dist"][cam_name] = {
                "min": float(min_dist),
                "max": float(max_dist),
            }

            # Repulsive
            delta_r = repulse_thres - min_dist
            if delta_r > 0:
                f_mag = delta_r if force_type == "linear" else (delta_r**repulse_alpha)
                repulsive_force += f_mag * (-unit_vec)

            # Attractive
            delta_a = max_dist - attract_thres
            if delta_a > 0:
                f_mag = delta_a if force_type == "linear" else (delta_a**attract_alpha)
                attractive_force += f_mag * unit_vec

    # 2. Calculate forces from front camera's regions
    if "front" in depth_images:
        img_front = depth_images["front"]
        edge_band = _scaled_px(cam_prop, 64)
        # left region (Hxedge_band), right region (Hxedge_band), bottom region (edge_bandxW)
        regions = {
            "left": img_front[:, :edge_band],
            "right": img_front[:, -edge_band:],
            "down": img_front[-edge_band:, :],
        }
        for region_name, region_img in regions.items():
            if region_name == "down":
                kernel_size = (window_f, edge_band)
            else:
                kernel_size = (edge_band, window_f)

            min_dist, max_dist = _get_min_max_avg_dist(
                region_img, kernel_size=kernel_size
            )
            log_data["dist"][f"front_{region_name}"] = {
                "min": float(min_dist),
                "max": float(max_dist),
            }
            unit_vec = cam_configs[region_name]

            # Repulsive
            delta_r = repulse_thres - min_dist
            if delta_r > 0:
                f_mag = delta_r if force_type == "linear" else (delta_r**repulse_alpha)
                repulsive_force += f_mag * (-unit_vec)

            # Attractive
            delta_a = max_dist - attract_thres
            if delta_a > 0:
                f_mag = delta_a if force_type == "linear" else (delta_a**attract_alpha)
                attractive_force += f_mag * unit_vec

    # 3. Calculate forces from front camera's center region to move sideways
    if "front" in depth_images:
        img_front = depth_images["front"]
        # Center region 128x128
        h, w = img_front.shape
        center_region = img_front[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_kernel = _scaled_px(cam_prop, 128)
        min_dist, max_dist = _get_min_max_avg_dist(
            center_region, kernel_size=(center_kernel, center_kernel)
        )
        log_data["dist"]["front_center"] = {
            "min": float(min_dist),
            "max": float(max_dist),
        }

        # Repulsive from front center
        delta_r = repulse_thres_front - min_dist
        if delta_r > 0:
            f_mag = delta_r if force_type == "linear" else (delta_r**repulse_alpha)

            # Weighted sideways force
            # Get average distance in left and right camera from section 1
            left_avg = (
                log_data["dist"]["left"]["min"] + log_data["dist"]["left"]["max"]
            ) / 2
            right_avg = (
                log_data["dist"]["right"]["min"] + log_data["dist"]["right"]["max"]
            ) / 2

            # More average distance means more free space in that direction
            # Left unit vec: [0, -1, 0], Right unit vec: [0, 1, 0]
            if left_avg > right_avg:
                sideways_unit_vec = np.array([0.0, -1.0, 0.0])
            else:
                sideways_unit_vec = np.array([0.0, 1.0, 0.0])

            # Add small upward force (-Z in NED)
            upward_force = np.array([0.0, 0.0, -0.1])
            sideways_up_vec = sideways_unit_vec + upward_force

            repulsive_force += f_mag * sideways_up_vec

    dest_norm = np.linalg.norm(dest_vec)
    res_vec = np.array(dest_vec) + dest_norm * (
        repulse_scale * repulsive_force + attract_scale * attractive_force
    )
    log_data["dest_vec"] = dest_vec
    log_data["repulsive_force"] = repulsive_force.tolist()
    log_data["attractive_force"] = attractive_force.tolist()
    log_data["res_vec"] = res_vec.tolist()
    return res_vec.tolist(), log_data


def is_direction_clear(
    pixel_coord: tuple[str, float, float],
    depth_images: dict[str, np.ndarray],
    cam_prop: CamProp,
) -> tuple[bool, float]:
    """
    Checks if a given direction (pixel) is clear based on surrounding depth values.
    """
    cam_name, u, v = pixel_coord
    if cam_name == "up":
        return False, 0.0

    method = "8-point"  # "concentric-two" or "8-point"

    offset = _scaled_px(cam_prop, CLEAR_POINT_OFFSET)
    clear_square_size = _scaled_px(cam_prop, CLEAR_SQUARE_SIZE)
    if method == "8-point":
        offsets = [
            (offset, 0),
            (0, offset),
            (-offset, 0),
            (0, -offset),
            (offset, offset),
            (-offset, offset),
            (offset, -offset),
            (-offset, -offset),
        ]
        distances = []
        for du, dv in offsets:
            curr_cam, curr_u, curr_v = wrap_pixel_coord(
                cam_name, u + du, v + dv, cam_prop
            )
            if curr_cam == "up":
                continue

            dist = get_distance_at_pixel_wrapped(
                depth_images,
                curr_cam,
                curr_u,
                curr_v,
                cam_prop,
                max_square_size=clear_square_size,
            )
            distances.append(dist)

        if not distances:
            return False, 0.0

        avg_dist = sum(distances) / len(distances)
    elif method == "concentric-two":
        dist_128_sq = get_distance_at_pixel_wrapped(
            depth_images,
            cam_name,
            u,
            v,
            cam_prop,
            max_square_size=clear_square_size * 2,
        )
        dist_64_sq = get_distance_at_pixel_wrapped(
            depth_images, cam_name, u, v, cam_prop, max_square_size=clear_square_size
        )
        avg_dist = (dist_128_sq + dist_64_sq) / 2

    # logging.debug(f"Average distance in camera {cam_name} (u={u}, v={v}): {avg_dist}")

    if cam_name == Camera.DOWN.value:
        return bool(avg_dist > CLEAR_DOWN_DIST_MIN), avg_dist

    if cam_name in [
        Camera.FRONT.value,
        Camera.LEFT.value,
        Camera.RIGHT.value,
        Camera.REAR.value,
    ]:
        if v > (2 / 3) * cam_prop.img_height:
            return bool(avg_dist > CLEAR_LOWER_DIST_MIN), avg_dist

    return bool(avg_dist > CLEAR_UPPER_DIST_MIN), avg_dist


class SafeNav:
    def __init__(self, cam_prop: CamProp = OPENUAV_CAM_PROP):
        self.cam_prop = cam_prop

    def find_safe_direction_with_region(
        self,
        dest_point: list[float],
        depth_images: dict[str, np.ndarray],
        vertical_dir_preference: str,
        slower: bool,
        scene_id: str,
        height_controller: HeightController
    ) -> tuple[list[float], int, dict[str, list[dict]]]:
        """
        Tries to find a clear region first and then a safe direction to move to.

        Supports an initial up camera directed point and returning an up camera directed point if needed.

        Args:
            dest_point: destination point in UAV body frame (NED)
            vertical_dir_preference: "up", "down", "horizontal"
            slower: whether to move slower

        Return:
            unit vector for safe direction in UAV body frame (NED)
        """
        # Apply repulsive and attractive forces to destination point
        include_forces = True
        include_safe_offset_check = True
        
        force_log_data = {}
        if include_forces:
            force_log_data = {}
            dest_point, force_log_data = apply_forces(
                dest_point, depth_images, self.cam_prop
            )

        # 1. Map initial point to all visible cameras
        pixel_coords = map_3d_point_to_img_pixel(dest_point, self.cam_prop)
        log_data = {
            "regions": [],
            "force": force_log_data,
        }
        if not pixel_coords:
            p_vec = np.array(dest_point)
            norm = np.linalg.norm(p_vec)
            return (
                (p_vec / norm).tolist() if norm > 0 else [0.0, 0.0, 0.0],
                STEP_SIZE,
                log_data,
            )

        # Use the first camera it's visible in
        start_cam = list(pixel_coords.keys())[0]
        start_u, start_v = pixel_coords[start_cam]

        # 2. Update pixel based on preference
        v_shift_up = self.cam_prop.img_height / 3.0
        v_shift_down = self.cam_prop.img_height / 4.0
        u, v = start_u, start_v
        if start_cam == Camera.DOWN.value:
            if vertical_dir_preference == "horizontal":
                v -= v_shift_up
            elif vertical_dir_preference == "up":
                v -= 2 * v_shift_up
        elif start_cam in [
            Camera.FRONT.value,
            Camera.LEFT.value,
            Camera.RIGHT.value,
            Camera.REAR.value,
        ]:
            if vertical_dir_preference == "down":
                if v < (2 / 3) * self.cam_prop.img_height:
                    v += v_shift_down
            elif vertical_dir_preference == "up":
                if v > (1 / 3) * self.cam_prop.img_height:
                    v -= v_shift_up

        if start_cam == "up":
            curr_cam, u, v = start_cam, u, v
            cam_check, (u_check, v_check) = closest_to_up_cam_pixel(u, v, self.cam_prop)
        else:
            curr_cam, u, v = wrap_pixel_coord(start_cam, u, v, self.cam_prop)
            cam_check, u_check, v_check = curr_cam, u, v
        # logging.debug(f"Initial pixel coordinates: {start_cam}, {start_u}, {start_v}")
        # logging.debug(f"Updated pixel coordinates: {curr_cam}, {u}, {v}")

        # 3. Check if base direction is clear
        dest_is_clear, avg_dist = is_direction_clear(
            (cam_check, u_check, v_check), depth_images, self.cam_prop
        )
        log_data["regions"].append(
            {"pixel": (curr_cam, u, v), "is_clear": dest_is_clear, "distance": avg_dist}
        )
        
        if not include_safe_offset_check:
            return (
                get_ned_direction_from_pixel(curr_cam, u, v, self.cam_prop).tolist(),
                STEP_SIZE,
                log_data,
            )

        if dest_is_clear:
            # Get a 3D point towards the pixel
            direction = get_ned_direction_from_pixel(curr_cam, u, v, self.cam_prop) * 20
            safe_dir, avg_dist = find_safe_direction_with_offset(
                direction.tolist(),
                depth_images,
                self.cam_prop,
                threshold=self._derive_safety_threshold(curr_cam, v),
            )
            if not np.allclose(safe_dir, [0, 0, 0]):
                step_size = self._get_step_size(safe_dir, avg_dist, slower, depth_images)
                log_data.update(
                    {
                        "unit_safe_dir": safe_dir,
                        "offset_avg_dist": avg_dist,
                        "step_size": step_size,
                    }
                )
                return (
                    safe_dir,
                    step_size,
                    log_data,
                )

        # 4. Search candidates
        # Change pixel to closest for up camera (others unchanged)
        curr_cam, u, v = cam_check, u_check, v_check

        height_from_ground = get_height_from_ground(depth_images["down"])
        if height_from_ground < 10:
            u_offset = _scaled_px(self.cam_prop, LOWER_ALT_SAFE_REGION_OFFSET)
            v_offset = _scaled_px(self.cam_prop, LOWER_ALT_SAFE_REGION_OFFSET)
        else:
            u_offset = _scaled_px(self.cam_prop, HIGHER_ALT_SAFE_REGION_OFFSET)
            v_offset = _scaled_px(self.cam_prop, HIGHER_ALT_SAFE_REGION_OFFSET)

        u_multipliers = (
            ([-1, 1] if random.random() < 0.5 else [1, -1])
            + ([-2, 2] if random.random() < 0.5 else [2, -2])
            + ([-3, 3] if random.random() < 0.5 else [3, -3])
            + ([-4, 4] if random.random() < 0.5 else [4, -4])
        )
        v_multipliers = [0, -1, 1, -2, 2, -3, 3]
        u_multi_set1 = u_multipliers[: len(u_multipliers) // 2]
        u_multi_set2 = u_multipliers[len(u_multipliers) // 2 :]
        v_multi_set1 = v_multipliers[: len(v_multipliers) // 2]
        v_multi_set2 = v_multipliers[len(v_multipliers) // 2 :]
        
        for v_set, u_set in [(v_multi_set1, u_multi_set1), (v_multi_set1, u_multi_set2), (v_multi_set2, u_multi_set1), (v_multi_set2, u_multi_set2)]:
            # We try to avoid too much yaw turn and prefer vertical movement first
            for vm in v_set:
                target_v = v + vm * v_offset
                for um in u_set:
                    target_u = u + um * u_offset
                    c_cam, c_u, c_v = wrap_pixel_coord(
                        curr_cam, target_u, target_v, self.cam_prop
                    )
                    if c_cam == "up":
                        cam_check, (u_check, v_check) = closest_to_up_cam_pixel(
                            u, v, self.cam_prop
                        )
                    else:
                        cam_check, u_check, v_check = c_cam, c_u, c_v

                    is_clear, avg_dist = is_direction_clear(
                        (cam_check, u_check, v_check), depth_images, self.cam_prop
                    )
                    log_data["regions"].append(
                        {
                            "pixel": (c_cam, c_u, c_v),
                            "is_clear": is_clear,
                            "distance": avg_dist,
                        }
                    )

                    if is_clear:
                        if um in u_multi_set1 and vm in v_multi_set1:
                            # lower height  
                            height_controller.change_desired_height(scene_id, -0.5) 
                        else: 
                            # we needed significant turn, increase height 
                            height_controller.change_desired_height(scene_id, 1)

                        direction = (
                            get_ned_direction_from_pixel(c_cam, c_u, c_v, self.cam_prop)
                            * 20
                        )
                        safe_dir, avg_dist = find_safe_direction_with_offset(
                            direction.tolist(),
                            depth_images,
                            self.cam_prop,
                            threshold=self._derive_safety_threshold(c_cam, c_v),
                        )
                        if np.allclose(safe_dir, [0, 0, 0]):
                            continue

                        step_size = self._get_step_size(
                            safe_dir, avg_dist, slower, depth_images
                        )
                        log_data.update(
                            {
                                "unit_safe_dir": safe_dir,
                                "offset_avg_dist": avg_dist,
                                "step_size": step_size,
                            }
                        )
                        return (
                            safe_dir,
                            step_size,
                            log_data,
                        )

        # Fallback
        p_vec = np.array(dest_point)
        norm = np.linalg.norm(p_vec)
        return (
            (p_vec / norm).tolist() if norm > 0 else [1.0, 0.0, 0.0],
            STEP_SIZE,
            log_data,
        )

    def _derive_safety_threshold(self, cam_name, v):
        # Calculate safety threshold based on camera
        safe_threshold = SAFE_OFFSET_UPPER_DIST_MIN
        if cam_name == Camera.DOWN.value:
            safe_threshold = SAFE_OFFSET_DOWN_DIST_MIN

        if cam_name in [
            Camera.FRONT.value,
            Camera.LEFT.value,
            Camera.RIGHT.value,
            Camera.REAR.value,
        ]:
            if v > (2 / 3) * self.cam_prop.img_height:
                safe_threshold = SAFE_OFFSET_LOWER_DIST_MIN

        return safe_threshold

    def _get_step_size(
        self,
        direction: list[float],
        avg_dist: float,
        slower: bool,
        depth_images: dict[str, np.ndarray],
    ) -> float:
        """
        Calculate an optimal step size considering both efficiency and safety.

        Step size is a function of (avg distance, action, height)

        Args:
            direction (list[float]): The direction that the robot will move.
            avg_dist (float): The average distance in meters towards direction vector.
            slower (bool): Whether to move slower.
        """
        step_size = STEP_SIZE
        if avg_dist > 90:
            step_size = STEP_SIZE + 1.5
        elif avg_dist > 85:
            step_size = STEP_SIZE + 1.25
        elif avg_dist > 80:
            step_size = STEP_SIZE + 0.75
        elif avg_dist > 70:
            step_size = STEP_SIZE + 0.25

        if avg_dist < 25:
            step_size = STEP_SIZE - 1
        elif avg_dist < 40:
            step_size = STEP_SIZE - 0.5

        if slower:
            if step_size > STEP_SIZE:
                step_size = STEP_SIZE + (step_size - STEP_SIZE) * 0.5
            else:
                step_size -= 0.5

        height_from_ground = get_height_from_ground(depth_images[Camera.DOWN.value])
        if height_from_ground < NEAR_GROUND_HEIGHT + 2:
            step_size -= 0.5

        step_size = min(step_size, STEP_SIZE_MAX)

        # Go slower when going downward
        downward = max(0, direction[2])
        if downward > 0:
            step_size -= 2**downward - 1

        if downward > 0.15:
            step_size = max(step_size, STEP_SIZE_MIN_DOWNWARD)
        else:
            step_size = max(step_size, STEP_SIZE_MIN_FORWARD_UP)

        return step_size
