import cv2
import math
import numpy as np 
from sklearn.cluster import DBSCAN

def lidar_scan(msgScan):
    # print(msgScan.range_max, msgScan.range_min, msgScan.angle_max - msgScan.angle_min, msgScan.angle_increment)
    """
    Convert LaserScan msg to array
    """
    distances = np.array([])
    angles = np.array([])
    information = np.array([])
    
    for i in range(len(msgScan.ranges)):
        # angle calculation
        ang = msgScan.angle_min + i * msgScan.angle_increment

        #if ang < 6.02139:
        #    continue

        # distance calculation
        if ( msgScan.ranges[i] > msgScan.range_max ):
            dist = msgScan.range_max
        elif ( msgScan.ranges[i] < msgScan.range_min ):
            dist = msgScan.range_min
        else:
            dist = msgScan.ranges[i]

        #dist = msgScan.ranges[i]

        # smaller the distance, bigger the information (measurement is more confident)
        inf = ((msgScan.range_max - dist) / msgScan.range_max) ** 2 

        distances = np.append(distances, dist)
        angles = np.append(angles, normalize_angle(ang))
        information = np.append(information, inf)

    # distances in [m], angles in [radians], information [0-1]
    return ( distances, angles, information,  msgScan.range_max, msgScan.angle_max - msgScan.angle_min)

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def extract_features(image):
    # Use a feature detector like ORB
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

'''
    eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
'''
def extract_features_dbscan(image, eps=5, min_samples=3):
    # Use a feature detector like ORB
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if keypoints:
        # Extract x, y coordinates of the keypoints
        coords = np.array([kp.pt for kp in keypoints])

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coords)

        print("Num feats", len(keypoints), "Unique labels", len(set(labels)))

        # Initialize lists to hold the representative keypoints and descriptors
        representative_keypoints = []
        representative_descriptors = []

        # Iterate over each cluster label
        for label in set(labels):
            if label == -1:
                continue  # Skip noise points

            # Indices of keypoints in the current cluster
            indices = [i for i, lbl in enumerate(labels) if lbl == label]

            #print("Cluster", label, "members", len(indices))

            # Cluster's keypoints and their descriptors
            cluster_keypoints = [keypoints[i] for i in indices]
            cluster_descriptors = descriptors[indices] if descriptors is not None else []

            #print(len(cluster_keypoints), len(cluster_descriptors))

            # Calculate the centroid of the cluster
            centroid = np.mean([kp.pt for kp in cluster_keypoints], axis=0)
            #print("centroid", centroid)

            # Find the keypoint closest to the centroid
            closest_index = np.argmin([np.linalg.norm(np.array(kp.pt) - centroid) for kp in cluster_keypoints])
            #print("closest index", closest_index)
            representative_keypoint = cluster_keypoints[closest_index]
            representative_descriptor = cluster_descriptors[closest_index]

            representative_keypoints.append(representative_keypoint)
            representative_descriptors.append(representative_descriptor)

        return representative_keypoints, np.array(representative_descriptors)

    return keypoints, descriptors

def filter_keypoints_in_vertical_band(keypoints, descriptors, image_height, vertical_band_center, vertical_band_height):
    """
    Filter keypoints to keep only those within a specified vertical band of the image.

    :param keypoints: List of keypoints.
    :param descriptors: Corresponding descriptors.
    :param image_height: Height of the image.
    :param vertical_band_center: Center y-coordinate of the vertical band.
    :param vertical_band_height: Height of the vertical band.
    :return: Filtered keypoints and descriptors.
    """
    filtered_keypoints = []
    filtered_descriptors = []
    top_bound = vertical_band_center - vertical_band_height / 2
    bottom_bound = vertical_band_center + vertical_band_height / 2

    for i, kp in enumerate(keypoints):
        if top_bound <= kp.pt[1] <= bottom_bound:
            filtered_keypoints.append(kp)
            filtered_descriptors.append(descriptors[i])

    return filtered_keypoints, filtered_descriptors

def convert_x_to_angle(x, image_width, fov_horizontal):
    x_norm = (x / image_width) * 2 - 1  # Normalize x to [-1, 1]
    angle = x_norm * (fov_horizontal / 2)  # Convert to angle
    return -normalize_angle(math.radians(angle)) # negate to change to clockwise as per ROS standards

def compute_xy_location(distance, angle, offset_y):
    # Initial computation as if there's no offset
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)

    # Adjust for the horizontal offset
    y += offset_y

    return x, y

def transform_point_to_global_frame(local_x, local_y, robot_x, robot_y, robot_yaw):
    # Convert local (robot-centric) coordinates to global coordinates based on odometry data

    # Apply rotation
    R = [ [math.cos(robot_yaw), -math.sin(robot_yaw)],
          [math.sin(robot_yaw),  math.cos(robot_yaw)] ]
    
    # apply rotation with matrix multiplication
    global_x, global_y = np.dot(R, np.array([local_x, local_y]))

    # Apply translation
    global_x += robot_x
    global_y += robot_y

    return global_x, global_y

def match_features_to_laserscan(feature_angles, laser_angles, laser_ranges, descriptors, overlapping_sector, max_lidar_range = 3.0, max_angular_difference = 0.0174533):
    matched_ranges = []
    matched_angles = []
    matched_descriptors = []
    indices = []
    start_angle, end_angle = overlapping_sector

    # Iterate over each feature angle
    for i, feature_angle in enumerate(feature_angles):
        # Check if feature is within the overlapping sector
        if start_angle <= feature_angle <= end_angle:
            # Find the closest angle in the LaserScan data to the feature angle
            closest_angle_index = min(range(len(laser_angles)), key=lambda i: abs(laser_angles[i] - feature_angle))
            closest_angle = laser_angles[closest_angle_index]
            range_for_feature = laser_ranges[closest_angle_index]

            # Check if the closest angle is within the acceptable range
            if abs(closest_angle - feature_angle) <= max_angular_difference and range_for_feature < max_lidar_range:
                # Get the corresponding range from the LaserScan data
                
                matched_ranges.append(range_for_feature)
                matched_angles.append(feature_angle)
                matched_descriptors.append(descriptors[i])
                indices.append(i)
            #else:
                # If the difference is too large, append an invalid marker (like None or -1)
                #matched_ranges.append(None)

    return matched_ranges, matched_angles, matched_descriptors, indices