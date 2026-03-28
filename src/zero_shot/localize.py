import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))


from airsim_plugin.AirVLNSimulatorClientTool import NUM_WAYPOINTS_TO_MOVE
from src.vlnce_src.batch_state import EvalBatchState
from utils.cam_util import (
    OPENUAV_CAM_PROP,
    Camera,
    CamProp,
    get_cam_intrinsic_params,
    get_cam_axes,
)
from utils.depth_util import _get_3d_coord_from_pixel, measure_3d_coordinate
from utils.log_util import SceneLogReader
from utils.pos_util import local_to_world

logging.getLogger("matplotlib").setLevel(logging.WARNING)

### CONFIGURABLE #####
# ORB match and filter
HUE_MATCH_THRESHOLD = 30  # Maximum allowed Hue difference for a match
LOWES_TEST_RATIO = 0.85  # Lowe's ratio test threshold
MIN_GOOD_MATCHES = 50  # Minimum number of good matches required to attempt motion estimation
MATCHED_POINTS_MAX_DISTANCE = 8.0 # Maximum distance in meters
#######################

def match_and_filter_orb_with_color(img1, img2):
    """
    Match ORB features between two images and filter matches based on color similarity.

    Args:
        img1 and img2: Input images (BGR format)
    """
    # 1. Convert to grayscale for ORB, but keep HSV for color checking
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return [], kp1, kp2

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        # 2. Lowe's Ratio Test
        if (
            m.distance < LOWES_TEST_RATIO * n.distance
        ):  # Expecting some one-to-many matches within an image

            # 3. Get the pixel coordinates of the matched keypoints
            # (Convert float coordinates to integers)
            pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
            pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))

            # Ensure coordinates are within image bounds
            if (
                0 <= pt1[1] < hsv1.shape[0]
                and 0 <= pt1[0] < hsv1.shape[1]
                and 0 <= pt2[1] < hsv2.shape[0]
                and 0 <= pt2[0] < hsv2.shape[1]
            ):

                # Extract the Hue value (0-179 in OpenCV)
                hue1 = int(hsv1[pt1[1], pt1[0], 0])
                hue2 = int(hsv2[pt2[1], pt2[0], 0])

                # Calculate circular distance for Hue (since 179 is close to 0)
                hue_diff = min(abs(hue1 - hue2), 180 - abs(hue1 - hue2))

                # 4. Color Threshold Check
                if hue_diff < HUE_MATCH_THRESHOLD:
                    good_matches.append(m)

    return good_matches, kp1, kp2


def draw_matches(good_matches, kp1, kp2, img1, img2, matches_mask, save_path):
    # Draw only the inliers
    draw_params = dict(
        matchColor=(0, 255, 0),  # Green for inliers
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    res_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    plt.imshow(res_img)
    plt.title("ORB Matches")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def filter_matched_points_by_depth(
    pts1, depth_map1, pts2, depth_map2, max_depth_meters
) -> tuple[list, list]:
    # Filter points based on depth
    filtered_pts1 = []
    filtered_pts2 = []

    for (u1, v1), (u2, v2) in zip(pts1, pts2):
        z1 = depth_map1[int(v1), int(u1)]
        z2 = depth_map2[int(v2), int(u2)]

        if 0.5 < z1 < max_depth_meters * 0.9 and 0.5 < z2 < max_depth_meters * 0.9:
            filtered_pts1.append((u1, v1))
            filtered_pts2.append((u2, v2))

    return np.array(filtered_pts1), np.array(filtered_pts2)


def get_points3d_world(
    pixel_coords, depth, cam_prop, rot_mat: np.ndarray, pos: np.ndarray
) -> np.ndarray:
    """
    Convert pixel coordinates and depth to 3D world coordinates.

    Depth is 0 - 255
    """
    z, x, y = measure_3d_coordinate(
        depth, pixel_coords[:, 0], pixel_coords[:, 1], cam_prop
    )
    points3d_local = np.array(list(zip(x.tolist(), y.tolist(), z.tolist())))
    points3d_world = local_to_world(points3d_local, rot_mat)
    points3d_world += pos
    return points3d_world


def get_points3d_world_for_cam(
    pixel_coords, depth, cam_prop, rot_mat: np.ndarray, pos: np.ndarray, cam_name: str
) -> np.ndarray:
    if cam_name not in [Camera.FRONT.value, Camera.LEFT.value, Camera.RIGHT.value]:
        raise ValueError(
            f"Unsupported cam_name: {cam_name}. Expected one of front/left/right."
        )

    axis_f, axis_r, axis_d = get_cam_axes()[cam_name]
    forward, right, down = measure_3d_coordinate(
        depth, pixel_coords[:, 0], pixel_coords[:, 1], cam_prop
    )

    points3d_local = (
        np.outer(forward, axis_f) + np.outer(right, axis_r) + np.outer(down, axis_d)
    )
    points3d_world = (rot_mat @ points3d_local.T).T
    points3d_world += pos
    return points3d_world


def print_points3d_distance(points3d_1, points3d_2):
    dist = np.linalg.norm(points3d_1 - points3d_2, axis=1)
    print("Total 3D point pairs: ", len(dist))
    print("Average distance between 3D points: ", np.mean(dist))
    print("Distance < 5m: ", np.sum(dist < 5))
    print("Distance < 10m: ", np.sum(dist < 10))
    print("Distance < 20m: ", np.sum(dist < 20))
    print("Distance < 40m: ", np.sum(dist < 40))


def filter_matched_points_by_distance(points3d_1, points3d_2, max_distance):
    dist = np.linalg.norm(points3d_1 - points3d_2, axis=1)
    mask = dist <= max_distance
    return points3d_1[mask], points3d_2[mask]


def estimate_3d_to_3d_motion_affine(
    points3d_1, points3d_2
) -> tuple[np.ndarray, np.ndarray]:
    # Use RANSAC to find the rigid transformation (R, t)
    # estimateAffine3D returns a 3x4 matrix [R | t]
    # estimateAffine3D requires at least 5 points, otherwise it returns None in output
    retval, M, inliers = cv2.estimateAffine3D(
        points3d_1, points3d_2, ransacThreshold=0.05, confidence=0.995
    )

    if retval and M is not None:
        R = M[:, :3]
        t = M[:, 3]
        return R, t

    return None, None


def estimate_3d_to_3d_motion_avg(
    points3d_1, points3d_2
) -> tuple[np.ndarray, np.ndarray]:
    # Calculate the average translation and rotation
    t = np.mean(points3d_2 - points3d_1, axis=0)
    R = np.eye(3)
    return R, t


class Localizer:
    def __init__(self, batch_state: EvalBatchState, scene_ids: list[str]):
        self.batch_state = batch_state
        self.scene_ids = scene_ids
        self.time_str = datetime.now().strftime("%m%d_%H%M%S")

    def correct_drift(
        self, active_indices: list[int], active_scene_ids: list[str]
    ) -> dict:
        # Compare current frame with last few frames
        # If there is a good number of match, try to localize
        # Estimate amount of drift
        for ia, i in enumerate(active_indices):
            if len(self.batch_state.episodes[i]) >= 2:
                # We assume prev obs points coordinates are correct
                # Now, we measure the change in point coordinates from prev obs to cur obs
                # We subtract the change from the current position to get the corrected position
                prev_obs = self.batch_state.episodes[i][-NUM_WAYPOINTS_TO_MOVE - 1]
                cur_obs = self.batch_state.episodes[i][-1]
                step_num = len(self.batch_state.episodes[i]) // NUM_WAYPOINTS_TO_MOVE
                prev_pos_drift = prev_obs["sensors"]["state"]["position_drift"]
                cur_pos_drift = cur_obs["sensors"]["state"]["position_drift"]
                prev_rot_mat = prev_obs["sensors"]["imu"]["rotation"]
                cur_rot_mat = cur_obs["sensors"]["imu"]["rotation"]
                cam_prop = OPENUAV_CAM_PROP

                ts = []
                cam_list = [Camera.FRONT.value, Camera.LEFT.value, Camera.RIGHT.value]
                # for cam_name1, cam_name2 in product(cam_list, cam_list):   # takes time
                for cam_name1, cam_name2 in zip(cam_list, cam_list):
                    frame_prev = prev_obs["rgb"][cam_name1]
                    frame_cur = cur_obs["rgb"][cam_name2]
                    depth_map_prev = prev_obs["depth"][cam_name1]
                    depth_map_cur = cur_obs["depth"][cam_name2]

                    good_matches, kp1, kp2 = match_and_filter_orb_with_color(
                        frame_prev, frame_cur
                    )
                    print(f"Ep {i}: Total Matches: {len(good_matches)}")

                    draw_matches(
                        good_matches,
                        kp1,
                        kp2,
                        frame_prev,
                        frame_cur,
                        np.ones(len(good_matches)).astype(int).tolist(),
                        f"localizer/{self.time_str}/{active_scene_ids[ia]}/{step_num}_{cam_name1}_{cam_name2}.png",
                    )

                    if len(good_matches) < MIN_GOOD_MATCHES:
                        continue  # Not enough matches to reliably estimate motion

                    # Convert depth maps to 2D arrays of distances in meters
                    depth_map_meters_prev = (
                        depth_map_prev.astype(np.float32) / 255.0
                    ) * cam_prop.max_depth_meters
                    depth_map_meters_cur = (
                        depth_map_cur.astype(np.float32) / 255.0
                    ) * cam_prop.max_depth_meters

                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                    pts1, pts2 = filter_matched_points_by_depth(
                        pts1,
                        depth_map_meters_prev,
                        pts2,
                        depth_map_meters_cur,
                        cam_prop.max_depth_meters,
                    )
                    
                    if len(pts1) < 5:
                        continue

                    # Get 3D points for Frame 1 and Frame 2
                    points3d_1 = get_points3d_world_for_cam(
                        pts1,
                        depth_map_prev,
                        cam_prop,
                        prev_rot_mat,
                        prev_pos_drift,
                        cam_name1,
                    )
                    points3d_2 = get_points3d_world_for_cam(
                        pts2,
                        depth_map_cur,
                        cam_prop,
                        cur_rot_mat,
                        cur_pos_drift,
                        cam_name2,
                    )

                    points3d_1, points3d_2 = filter_matched_points_by_distance(
                        points3d_1, points3d_2, max_distance=MATCHED_POINTS_MAX_DISTANCE
                    )

                    if len(points3d_1) == 0 or len(points3d_2) == 0:
                        continue

                    R, t = estimate_3d_to_3d_motion_avg(points3d_1, points3d_2)
                    if R is None or t is None:
                        raise RuntimeError(
                            "3D-3D motion estimation failed (insufficient valid correspondences or affine estimation failure)."
                        )

                    ts.append(t)
                    print("Cam pair: ", cam_name1, cam_name2, " Estimated translation (drift): ", t)

                if len(ts) == 0:
                    continue

                t = np.array(ts).mean(axis=0)

                print(
                    f"Pos drift: {self.batch_state.cur_pos_error[i]}, Measured drift: \n{t}"
                )
                self.batch_state.episodes[i][-1]["sensors"]["state"][
                    "position_drift"
                ] = [cur_pos_drift[0] - t[0], cur_pos_drift[1] - t[1], cur_pos_drift[2]]


if __name__ == "__main__":
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    traj_dir = os.path.join(
        cur_file_dir,
        "../../Model/LLaMA-UAV/work_dirs/eval_closeloop/eval_test_seen_valset_one_dir_follow_20260305_164819/11ec7f68-c071-4a68-87ed-dab885910aaa",
    )
    img_file1 = os.path.join(traj_dir, "frontcamera/000400.png")
    img_file2 = os.path.join(traj_dir, "frontcamera/000405.png")
    depth_map_file1 = os.path.join(traj_dir, "frontcamera_depth/000400.png")
    depth_map_file2 = os.path.join(traj_dir, "frontcamera_depth/000405.png")

    log_reader = SceneLogReader(traj_dir)
    log1 = log_reader.read_log(400 // 5)
    log2 = log_reader.read_log(405 // 5)
    pos1 = np.array(log1["sensors"]["state"]["position"])
    pos2 = np.array(log2["sensors"]["state"]["position"])
    rot_mat1 = np.array(log1["sensors"]["imu"]["rotation"])
    rot_mat2 = np.array(log2["sensors"]["imu"]["rotation"])
    drift = np.array([2, 3, 0])
    pos2_drift = pos2 + drift
    print("Position 1: ", pos1)
    print("Position 2: ", pos2, " with drift: ", pos2_drift)
    print("Drift: ", drift)

    frame1 = cv2.imread(img_file1, 0)
    frame2 = cv2.imread(img_file2, 0)
    depth_map1 = cv2.imread(
        depth_map_file1, cv2.IMREAD_GRAYSCALE
    )  # Assuming single-channel 8-bit image
    depth_map2 = cv2.imread(depth_map_file2, cv2.IMREAD_GRAYSCALE)

    if frame1 is None or frame2 is None:
        raise FileNotFoundError("Could not load one or both RGB frames.")
    if depth_map1 is None or depth_map2 is None:
        raise FileNotFoundError("Could not load one or both depth maps.")

    img_width, img_height = frame1.shape[1], frame1.shape[0]
    assert img_width == img_height  # We are working with square images

    cam_prop = CamProp(
        img_width=img_width, img_height=img_height, fov_deg=90, max_depth_meters=100
    )
    f, cx, cy = get_cam_intrinsic_params(cam_prop)

    good_matches, kp1, kp2 = match_and_filter_orb_with_color(frame1, frame2)
    print(f"Total Matches: {len(good_matches)}")

    # Randomly sample 10 matches
    # good_matches = list(np.random.choice(good_matches, size=min(100, len(good_matches)), replace=False))

    draw_matches(
        good_matches,
        kp1,
        kp2,
        frame1,
        frame2,
        np.ones(len(good_matches)).astype(int).tolist(),
        "orb_matches.png",
    )

    print("Estimating motion using 3D-3D correspondences...")

    # Convert depth maps to 2D arrays of distances in meters
    depth_map_meters1 = (
        depth_map1.astype(np.float32) / 255.0
    ) * cam_prop.max_depth_meters
    depth_map_meters2 = (
        depth_map2.astype(np.float32) / 255.0
    ) * cam_prop.max_depth_meters

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    pts1, pts2 = filter_matched_points_by_depth(
        pts1, depth_map_meters1, pts2, depth_map_meters2, cam_prop.max_depth_meters
    )

    print("Pts1 after depth filtering: ", pts1)
    print("Pts2 after depth filtering: ", pts2)

    # Get 3D points for Frame 1 and Frame 2
    points3d_1 = get_points3d_world(pts1, depth_map1, cam_prop, rot_mat1, pos1)
    points3d_2 = get_points3d_world(pts2, depth_map2, cam_prop, rot_mat2, pos2_drift)

    print("3D Points from Frame 1: ", points3d_1[:100])
    print("3D Points from Frame 2: ", points3d_2[:100])
    print_points3d_distance(points3d_1, points3d_2)
    points3d_1, points3d_2 = filter_matched_points_by_distance(
        points3d_1, points3d_2, max_distance=5
    )
    if len(points3d_1) == 0:
        print("No valid 3D point correspondences after distance filtering.")
        exit(0)

    R, t = estimate_3d_to_3d_motion_avg(points3d_1, points3d_2)

    if R is None or t is None:
        raise RuntimeError(
            "3D-3D motion estimation failed (insufficient valid correspondences or affine estimation failure)."
        )

    print(f"Rotation Matrix R: \n{R}")
    print(f"Translation Vector (t): \n{t}")
