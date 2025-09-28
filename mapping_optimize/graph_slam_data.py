import pickle

class GraphSLAMData:
    def __init__(self):
        self.poses = []
        self.controls = []
        self.landmarks = []
        self.observations = []

    def add_control_and_pose(self, v, w, x, y, theta, timestamp):
        self.poses.append((x, y, theta))
        self.controls.append((v, w, timestamp))
        return len(self.controls) - 1

    def add_landmark(self, x, y, feature):
        self.landmarks.append((x, y, feature))
        #return len(self.landmarks) - 1

    def add_observation(self, pose_index, landmark_index, r, phi):
        self.observations.append((pose_index, landmark_index, r, phi))

    def find_landmark(self, feature_descriptor):
        # Implement logic to find a landmark based on the feature descriptor
        # This could be based on similarity measures, nearest neighbor, etc.
        for i, (_, _, feature) in enumerate(self.landmarks):
            if feature_descriptor == feature:  # Simple example, replace with actual matching logic
                return i
        return None

    def save_to_file(self, filename):
        """ Save the data to a file. """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename):
        """ Load data from a file. """
        with open(filename, 'rb') as file:
            return pickle.load(file)

if __name__ == '__main__':
    # Example usage
    slam_data = GraphSLAMData.load_from_file("tb3_03_16_slow.pkl")

    print(len(slam_data.controls))
    print(slam_data.landmarks)

    observations_by_pose = {}
    for observation in slam_data.observations:
        pose_index, landmark_index, r, phi = observation
        if pose_index not in observations_by_pose:
            observations_by_pose[pose_index] = []
        observations_by_pose[pose_index].append(landmark_index)

    for pose_index in sorted(observations_by_pose.keys()):
    
        # LINE 12 START
        observations = observations_by_pose[pose_index]  # List of observations for this pose
        # For all observed features z_t_i
        print(pose_index, observations)