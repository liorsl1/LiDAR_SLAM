import cv2
import numpy as np

import pickle
import math

from features_utils import normalize_angle

class VisualHashMap:
    class Feature:
        def __init__(self, idx, x, y, descriptor):
            self.idx = idx
            self.coords = [(x, y)]
            self.descrs = [descriptor]
            self.avg_coords = (x, y)

        def update(self, x, y, descriptor):
            self.coords.append((x, y))
            self.descrs.append(descriptor)
            # Update to average the coordinates with the new one
            avg_x = sum(coord[0] for coord in self.coords) / len(self.coords)
            avg_y = sum(coord[1] for coord in self.coords) / len(self.coords)
            self.avg_coords = (avg_x, avg_y)


    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

        self.last_feature_idx = -1

        # Create BFMatcher object with Hamming distance
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # For Turtlebot
        # self.descriptor_distance_threshold = 120
        # self.cluster_proximity_threshold = 0.2 

        # For Kitti
        self.descriptor_distance_threshold = 200
        self.cluster_proximity_threshold = 3.0 

    def get_grid_cell(self, x, y):
        return (int(x / self.cell_size), int(y / self.cell_size))

    def add_feature(self, x, y, descriptor, idx):
        if idx < 0: 
            self.last_feature_idx += 1
            idx = self.last_feature_idx
        
        cell = self.get_grid_cell(x, y)
        if cell not in self.grid:
            self.grid[cell] = []

        # Check for nearby cluster to update
        for feature in self.grid[cell]:
            if feature.idx == idx:
                feature.update(x, y, descriptor)
                return idx

        # Create new cluster if no match found
        new_feature = self.Feature(idx, x, y, descriptor)
        self.grid[cell].append(new_feature)

        return idx

    def get_nearby_features(self, x, y):
        cell = self.get_grid_cell(x, y)
        # Optionally, check adjacent cells as well
        # ...
        return self.grid.get(cell, [])

    def get_all_features(self):
        all_feature_clusters = []
        for cell in self.grid.values():
            all_feature_clusters.extend(cell)

        all_features = []
        for feature in all_feature_clusters:
            all_features.append((feature.avg_coords[0], feature.avg_coords[1], feature.descrs[0]))
        return all_features

    def match_and_update(self, robot_pose, new_feature, overlapping_sector, max_distance, localize = False):
        
        new_x, new_y, new_descriptor, new_r, new_phi = new_feature

        # Find nearby features in the map
        nearby_features = self.get_nearby_features(new_x, new_y)

        for feature in nearby_features:
            match_count = 0

            if not self.is_within_overlapping_sector_and_distance(robot_pose, feature.avg_coords , overlapping_sector, max_distance):
                continue

            # Check proximity to the feature cluster's average coordinates
            if self.euclidean_distance((new_x, new_y), feature.avg_coords) > self.cluster_proximity_threshold:
                continue

            # Iterate through each descriptor in the cluster
            for existing_descriptor in feature.descrs:
                # Calculate descriptor distance
                distance = self.calculate_descriptor_distance(new_descriptor, existing_descriptor)
    
                if distance < self.descriptor_distance_threshold:
                    match_count += 1
                    if match_count >= 1: #len(feature.descrs) / 5.0:  # self.match_count_threshold:
                        # Update the existing cluster with the new feature
                        feature.update(new_x, new_y, new_descriptor)
                        return (feature.avg_coords[0], feature.avg_coords[1], new_descriptor, feature.idx)

        
                
        # If no match is found, add the new feature to the map
        idx = self.add_feature(new_x, new_y, new_descriptor, -1)
        return (new_x, new_y, new_descriptor, idx)

    def is_within_overlapping_sector_and_distance(self, robot_pose, landmark_position, overlapping_sector, max_distance):
        # Calculate relative position of the landmark from the robot's perspective
        rel_x = landmark_position[0] - robot_pose[0]
        rel_y = landmark_position[1] - robot_pose[1]
        
        # Calculate distance and angle to the landmark
        distance = math.sqrt(rel_x**2 + rel_y**2)
        angle = math.atan2(rel_y, rel_x) - robot_pose[2]  # Robot's orientation is robot_pose[2]

        # Normalize angle to be within -pi to pi
        angle = normalize_angle(angle)

        # Check if within overlapping sector and within maximum distance
        return overlapping_sector[0] <= angle <= overlapping_sector[1] and distance <= max_distance

    def calculate_descriptor_distance(self, descriptor1, descriptor2):
        # Convert descriptors to the format expected by BFMatcher
        # Note: Descriptors should be in a list of numpy arrays
        descriptor1 = np.array([descriptor1])
        descriptor2 = np.array([descriptor2])

        # Match descriptors
        matches = self.bf.match(descriptor1, descriptor2)

        # If there is a match, return its distance; otherwise return a large number
        if matches:
            # Return distance of the best match
            #if matches[0].distance > 2:
            #print("Distance:", matches[0].distance)
            return matches[0].distance
        else:
            # Return a large number to indicate no match
            return float('inf')

    def euclidean_distance(self, coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

    def save_map(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.grid, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_map(self, filename):
        with open(filename, 'rb') as file:
            self.grid = pickle.load(file)

