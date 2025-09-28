import matplotlib.pyplot as plt
import numpy as np 
import math

import cv2
from cv_bridge import CvBridge

class Visualizer:
    def __init__(self):
        # Initial plot setup
        #plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.sc_map, = self.ax.plot([], [], 'o', color='blue')  # For map features
        self.sc_robot, = self.ax.plot([], [], 'o', color='green')  # For robot position
        self.sc_predicted, = self.ax.plot([], [], 'o', color='red')  # For robot position
        self.sc_loop_closure, = self.ax.plot([], [], 'o', color='yellow')  # For loop closure landmarks
        self.yaw_line, = self.ax.plot([], [], color='green', linewidth=2)  # For robot yaw
        self.predicted_yaw_line, = self.ax.plot([], [], color='red', linewidth=2)  # For robot yaw
        self.ax.set_title("Sensor Fusion Map with Robot Position")
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.grid(True)
        self.ax.axis('equal')

        self.bridge = CvBridge()

        self.robot_xs = []
        self.robot_ys = []

    def update(self, map, robot_x, robot_y, robot_yaw, predicted_x, predicted_y, predicted_yaw, localize = False):
        #features = map.get_all_features()

        xs = [x for x, _, _ in map]
        ys = [y for _, y, _ in map]

        # Include robot's position in the list of coordinates
        xs.append(robot_x)
        ys.append(robot_y)

        self.robot_xs.append(robot_x)
        self.robot_ys.append(robot_y)

        # Update map and robot's position
        self.sc_map.set_data(xs, ys)
        self.sc_robot.set_data(self.robot_xs, self.robot_ys)
        
        # Update the yaw line
        line_length = 0.5  # Adjust length of the yaw line
        end_x = robot_x + line_length * math.cos(robot_yaw)
        end_y = robot_y + line_length * math.sin(robot_yaw)
        self.yaw_line.set_data([robot_x, end_x], [robot_y, end_y])

        if localize:
            # Plot the predicted pose of the robot
            predicted_end_x = predicted_x + line_length * math.cos(predicted_yaw)
            predicted_end_y = predicted_y + line_length * math.sin(predicted_yaw)

            xs.append(predicted_x)
            ys.append(predicted_y)

            self.sc_predicted.set_data(predicted_x, predicted_y)
            self.predicted_yaw_line.set_data([predicted_x, predicted_end_x], [predicted_y, predicted_end_y])

        # Dynamically adjust axes limits
        buffer = 5  # Buffer to add around the outermost features and robot
        
        # For Turtlebot
        # self.ax.set_xlim(min(xs) - buffer, max(xs) + buffer)
        # self.ax.set_ylim(min(ys) - buffer, max(ys) + buffer)

        # For Kitti
        self.ax.set_xlim(-10, 150)
        self.ax.set_ylim(-100, 150)
        
        # Instead of plt.draw(), update the figure canvas directly
        self.fig.canvas.draw_idle()

        # Save plot to a numpy array
        self.fig.canvas.draw()
        
        plt.pause(0.01)

    def update_loop_closure(self, loop_closure_landmarks):
        """
        Update the plot to show loop closure landmarks in yellow.

        :param loop_closure_landmarks: List of landmark coordinates for loop closure (list of tuples/lists).
        """
        if loop_closure_landmarks:
            # Unpack the landmark coordinates
            lc_xs, lc_ys = zip(*loop_closure_landmarks)
        else:
            lc_xs, lc_ys = [], []
        #print("Loop closure", lc_xs)
        # Update loop closure landmarks
        self.sc_loop_closure.set_data(lc_xs, lc_ys)

        # Redraw the plot
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

