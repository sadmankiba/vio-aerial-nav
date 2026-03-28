import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ai_dev.ad_robot.cam_util import get_cam_intrinsic_params, CamProp
from ai_dev.ad_robot.depth_util import get_3d_coord_from_pixel

# Gemini chat : https://gemini.google.com/share/17c1796f2ab9


def match_and_filter_orb(img1, img2):
    # 1. Initialize ORB detector
    # nfeatures: 1000-2000 is usually good for drone simulations
    orb = cv2.ORB_create(nfeatures=1500)

    # 2. Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return [], kp1, kp2

    # 3. Initialize Brute-Force Matcher with Hamming Distance
    # Binary descriptors (ORB, BRIEF) MUST use NORM_HAMMING
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 4. Use k-Nearest Neighbors (k=2) to allow for Ratio Test
    matches = bf.knnMatch(des1, des2, k=2)

    # 5. Apply Lowe's Ratio Test
    # If the closest match isn't much better than the second closest, discard it.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, kp1, kp2

def draw_matches(good_matches, kp1, kp2, img1, img2, matches_mask):
    # Draw only the inliers
    draw_params = dict(
        matchColor=(0, 255, 0),  # Green for inliers
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    res_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None, **draw_params
    )
    
    plt.imshow(res_img)
    plt.title("ORB Matches")
    plt.savefig("orb_matches.png")


def get_essential_matrix_and_inliers(good_matches, kp1, kp2, img1, img2, camera_matrix):
    # 6. Geometric Verification with RANSAC
    # We need at least 5 matches to estimate the Essential Matrix for a drone
    if len(good_matches) > 10:
        # Convert keypoints to coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Find Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is None or mask is None:
            print("Essential matrix estimation failed.")
            return None

        # 'mask' tells us which matches are "inliers" (physically possible)
        inlier_mask = mask.ravel().astype(bool)
        matches_mask = inlier_mask.astype(np.uint8).tolist()
        pts1_inliers = pts1[inlier_mask]
        pts2_inliers = pts2[inlier_mask]

        print(f"Inliers: {sum(matches_mask)}")
        
        return matches_mask, E, pts1_inliers, pts2_inliers
    else:
        print("Not enough matches found.")
        return None


# Assuming pts1, pts2 are the inlier points from your RANSAC mask
# and E is the Essential Matrix calculated previously.


def estimate_drone_motion(E, pts1, pts2, camera_matrix):
    # 1. Recover Pose
    # This function internally solves the "Cheirality Constraint"
    # (ensuring points are in front of the camera) to pick the 1 correct
    # solution out of the 4 possible decompositions of E.
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    # 2. Convert R to Euler Angles (Optional - for human readability)
    # This helps you see Pitch, Roll, and Yaw in degrees
    def rotation_matrix_to_euler(R):
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.rad2deg(np.array([x, y, z]))

    euler_angles = rotation_matrix_to_euler(R)

    print(f"Rotation (Euler deg): {euler_angles}")
    print("Rotation Matrix R: \n", R)
    print(f"Translation Vector (Direction): \n{t.flatten()}")

    return R, t


# Example output interpretation:
# t = [0, 0, 1] means the drone moved exactly forward.
# t = [0, -1, 0] means the drone moved "up" in the camera frame.


def estimate_3d_to_3d_motion(pts1, depth1, pts2, depth2, camera_matrix, max_depth_meters):
    # 1. Back-project 2D matches to 3D for BOTH frames
    def back_project(pts, depth_map, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_3d = []
        valid_indices = []

        h, w = depth_map.shape
        for i, (u, v) in enumerate(pts):
            if not (0 <= u < w and 0 <= v < h):
                continue

            z = depth_map[int(v), int(u)]
            if 0.5 < z < max_depth_meters * 0.9:  # Filter out sky/infinity
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                pts_3d.append([x, y, z])
                valid_indices.append(i)
        return np.array(pts_3d, dtype=np.float32), valid_indices

    def estimate_rigid_transform_svd(src_pts, dst_pts):
        src_centroid = np.mean(src_pts, axis=0)
        dst_centroid = np.mean(dst_pts, axis=0)

        src_centered = src_pts - src_centroid
        dst_centered = dst_pts - dst_centroid
        H = src_centered.T @ dst_centered

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = dst_centroid - R @ src_centroid
        return R.astype(np.float32), t.astype(np.float32)

    # Get 3D points for Frame 1 and Frame 2
    points3d_F1, idx1 = back_project(pts1, depth1, camera_matrix)
    points3d_F2, idx2 = back_project(pts2, depth2, camera_matrix)

    # Intersection: keep only points that have valid depth in BOTH frames
    common_idx = sorted(set(idx1) & set(idx2))

    if len(common_idx) < 4:
        return None, None

    src_pts = np.array([points3d_F1[idx1.index(i)] for i in common_idx])
    dst_pts = np.array([points3d_F2[idx2.index(i)] for i in common_idx])

    # 2. Use RANSAC to find the rigid transformation (R, t)
    # estimateAffine3D returns a 3x4 matrix [R | t]
    # estimateAffine3D requires at least 5 points, otherwise it returns None in output
    retval, M, inliers = cv2.estimateAffine3D(src_pts, dst_pts)

    if retval and M is not None:
        R = M[:, :3]
        t = M[:, 3]
        return R, t

    if len(src_pts) >= 3:
        return estimate_rigid_transform_svd(src_pts, dst_pts)

    return None, None


if __name__ == "__main__":
    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    img_file1 = os.path.join(cur_file_dir, "../../../sample-data/drone_sim/1.png")
    img_file2 = os.path.join(cur_file_dir, "../../../sample-data/drone_sim/2.png")
    depth_map_file1 = os.path.join(
        cur_file_dir, "../../../sample-data/drone_sim/1_depth.png"
    )
    depth_map_file2 = os.path.join(
        cur_file_dir, "../../../sample-data/drone_sim/2_depth.png"
    )

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

    # camera_matrix, K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam_prop = CamProp(
        img_width=img_width, img_height=img_height, fov_deg=90, max_depth_meters=100
    )
    f, cx, cy = get_cam_intrinsic_params(cam_prop)

    camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    good_matches, kp1, kp2 = match_and_filter_orb(frame1, frame2)
    print(f"Total Matches: {len(good_matches)}")
    
    # random sample 5 good matches
    # good_matches = random.sample(good_matches, min(5, len(good_matches)))
    
    draw_matches(good_matches, kp1, kp2, frame1, frame2, np.ones(len(good_matches)).astype(int).tolist())

    if len(good_matches) == 0:
        raise RuntimeError("No ORB matches found.")

    estimate_motion_method = "3D-3D"  # Options: "2D-2D", "3D-2D", or "3D-3D"

    if estimate_motion_method == "2D-2D":
        print(
            "Estimating motion using 2D-2D (Essential Matrix and RANSAC in image space) ..."
        )
        match_result = get_essential_matrix_and_inliers(
            good_matches, kp1, kp2, frame1, frame2, camera_matrix
        )
        if match_result is None:
            raise RuntimeError("Could not compute ORB matches / essential matrix.")

        matches_mask, essential_matrix, pts1_inliers, pts2_inliers = match_result
        print("matches_mask type:", type(matches_mask))
        draw_matches(good_matches, kp1, kp2, frame1, frame2, matches_mask)

        estimate_drone_motion(
            essential_matrix, pts1_inliers, pts2_inliers, camera_matrix
        )
    elif estimate_motion_method == "3D-2D":
        print("Estimating motion using 3D-2D (correspondences and PnP)...")
        # Convert depth maps to 2D arrays of distances in meters
        depth_map1 = (depth_map1.astype(np.float32) / 255.0) * cam_prop.max_depth_meters
        depth_map2 = (depth_map2.astype(np.float32) / 255.0) * cam_prop.max_depth_meters

        object_points = []
        image_points = []

        h, w = depth_map1.shape
        for m in good_matches:
            u, v = kp1[m.queryIdx].pt  # Pixel coords in Frame 1
            if not (0 <= u < w and 0 <= v < h):
                continue
            d = depth_map1[int(v), int(u)]  # Depth at that pixel
            
            if 0.5 < d < cam_prop.max_depth_meters * 0.9:  # Filter out sky/infinity
                # Convert pixel (u,v,z) to 3D (X,Y,Z) using camera intrinsics
                z, x, y = get_3d_coord_from_pixel(u, v, f, cx, cy, d)
                object_points.append([x, y, z])
                image_points.append(kp2[m.trainIdx].pt)  # 2D match in Frame 2

        # 2. Solve for motion using PnP
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        num_corr = len(object_points)
        if num_corr < 4:
            raise RuntimeError(
                f"Not enough valid 3D-2D correspondences for PnP: got {num_corr} (need >= 4)."
            )

        pnp_flag = cv2.SOLVEPNP_ITERATIVE if num_corr >= 6 else cv2.SOLVEPNP_EPNP
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, None, flags=pnp_flag
        )
        if not success:
            raise RuntimeError("solvePnP failed to estimate camera pose.")

        print(f"PnP Success: {success}")
        print(f"Rotation Vector (rvec): \n{rvec}")
        print(f"Translation Vector (tvec): \n{tvec}")

        # 3. Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        print(f"Rotation Matrix R: \n{R}")
    elif estimate_motion_method == "3D-3D":
        print(
            "Estimating motion using 3D-3D (depth correspondences and rigid transform)..."
        )

        # Convert depth maps to 2D arrays of distances in meters
        depth_map1 = (depth_map1.astype(np.float32) / 255.0) * cam_prop.max_depth_meters
        depth_map2 = (depth_map2.astype(np.float32) / 255.0) * cam_prop.max_depth_meters

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        R, t = estimate_3d_to_3d_motion(
            pts1, depth_map1, pts2, depth_map2, camera_matrix, cam_prop.max_depth_meters
        )
        if R is None or t is None:
            raise RuntimeError(
                "3D-3D motion estimation failed (insufficient valid correspondences or affine estimation failure)."
            )

        print(f"Rotation Matrix R: \n{R}")
        print(f"Translation Vector (t): \n{t}")
    else:
        raise ValueError(
            f"Unknown estimate_motion_method='{estimate_motion_method}'. Use one of: '2D-2D', '3D-2D', '3D-3D'."
        )


"""
Output:

Estimating motion using Essential Matrix and RANSAC in image space...
Total Matches: 29 | Inliers: 25
Rotation (Euler deg): [ 70.49077547 -36.40236485 -48.62169493]
Rotation Matrix R: 
 [[ 0.53204098 -0.11917651 -0.83828954]
 [-0.60394287  0.64049361 -0.47436372]
 [ 0.59345211  0.75865994  0.26879304]]
Translation Vector (Direction): 
[0.5855195  0.36002559 0.72632533]

Estimating motion using 3D-2D (correspondences and PnP)...
PnP Success: True
Rotation Vector (rvec): 
[[2.46958455]
 [1.39986674]
 [0.46580135]]
Translation Vector (tvec): 
[[-6.8082994 ]
 [ 5.49058554]
 [32.66619775]]
 
Estimating motion using 3D-3D (depth correspondences and rigid transform)...
Rotation Matrix R: 
[[  2.21832789  -3.91311575  -1.38412181]
 [  0.17693369   1.0431785    0.23195552]
 [  9.26614288 -30.83541547  -9.38298908]]
Translation Vector (t): 
[ 24.1987204    0.47695881 208.21929295]
"""
