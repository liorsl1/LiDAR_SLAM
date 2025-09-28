import gc
import os
import sys
import numpy as np
import math

# Adjust path if needed
import g2o
import g2opy

from features_utils import normalize_angle, compute_xy_location
from graph_slam_data import GraphSLAMData

import plotly.graph_objects as go


class GraphSLAM:
    def __init__(self, slam_data, verbose=False, method="gauss-newton", 
                 noise_level=1.0, add_noise_to_initial=True) -> None:
        """
        GraphSLAM in 2D with g2o.
        method: "gauss-newton" | "levenberg" | "dogleg"
        noise_level: scaling factor for noise (1.0 = default noise)
        add_noise_to_initial: whether to add noise to initial state
        """
        # Optimizer core
        self.optimizer = g2o.SparseOptimizer()
        self.solver = g2o.BlockSolverSE2(g2o.LinearSolverDenseSE2())

        method_lc = method.lower()
        if method_lc in ("gauss-newton", "gauss", "gn"):
            self.algorithm = g2o.OptimizationAlgorithmGaussNewton(self.solver)
        elif method_lc in ("levenberg", "lm", "levenberg-marquardt"):
            self.algorithm = g2o.OptimizationAlgorithmLevenberg(self.solver)
        elif method_lc in ("dogleg", "dl"):
            self.algorithm = g2o.OptimizationAlgorithmDogleg(self.solver)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        self.optimizer.set_algorithm(self.algorithm)

        # Internal counters
        self.vertex_count = 0
        self.edge_count = 0
        self.verbose = verbose
        self.vertex_count_before_gt = None
        
        # Noise parameters
        self.noise_level = noise_level
        self.add_noise_to_initial = add_noise_to_initial

        # Noise / information (tunable) - scaled by noise_level
        self.epsilon = 1e-9
        self.pose_std_dev = np.asarray([0.5, 0.5, 0.1]) * noise_level  # x,y,theta std dev
        self.pose_information = np.linalg.inv(np.diag(self.pose_std_dev ** 2))
        self.landmark_std_dev = np.asarray([1.0, 1.0]) * noise_level
        self.landmark_information = np.linalg.inv(np.diag(self.landmark_std_dev ** 2))
        
        # For adding noise to initial state
        self.initial_pose_noise_std = np.array([2.0, 2.0, 0.3]) * noise_level  # x, y, theta
        self.initial_landmark_noise_std = np.array([3.0, 3.0]) * noise_level  # x, y

        # Data container
        self.slam_data = slam_data

    def vertex_pose(self, id):
        '''
        Get position of vertex by id
        '''
        return self.optimizer.vertex(id).estimate()

    def vertex(self, id):
        '''
        Get vertex by id
        '''
        return self.optimizer.vertex(id)

    def edge(self, id):
        '''
        Get edge by id
        '''
        return self.optimizer.edge(id)

    def add_fixed_pose(self, pose, vertex_id=None):
        '''
        Add fixed pose to the graph
        '''
        v_se2 = g2o.VertexSE2()
        if vertex_id is None:
            vertex_id = self.vertex_count
        v_se2.set_id(vertex_id)
        if self.verbose:
            print("Adding fixed pose vertex with ID", vertex_id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(True)
        self.optimizer.add_vertex(v_se2)
        self.vertex_count += 1

    def add_odometry(self, northings, eastings, heading, information, add_noise=True):
        '''
        Add odometry edge using RELATIVE motion measurement.
        Vertex estimate is kept as absolute pose; edge measurement = T_{i-1}^{-1} * T_i.
        '''
        vertices = self.optimizer.vertices()
        if len(vertices) == 0:
            raise ValueError("Need an initial fixed pose before adding odometry.")
        # Find last pose vertex (largest id below current vertex_count)
        last_pose_id = self.vertex_count - 1
        last_pose = self.vertex(last_pose_id).estimate()

        # Absolute pose for new vertex
        abs_pose = g2o.SE2(northings, eastings, heading)
        # Relative measurement
        rel_measure = last_pose.inverse() * abs_pose

        v_se2 = g2o.VertexSE2()
        v_se2.set_id(self.vertex_count)
        
        # Add significant noise to initial guess to make optimization meaningful
        if add_noise and self.add_noise_to_initial:
            noisy_abs_pose = g2o.SE2(
                abs_pose.translation()[0] + np.random.normal(0, self.initial_pose_noise_std[0]),
                abs_pose.translation()[1] + np.random.normal(0, self.initial_pose_noise_std[1]),
                abs_pose.rotation().angle() + np.random.normal(0, self.initial_pose_noise_std[2])
            )
            v_se2.set_estimate(noisy_abs_pose)
        else:
            v_se2.set_estimate(abs_pose)
            
        self.optimizer.add_vertex(v_se2)

        e_se2 = g2o.EdgeSE2()
        e_se2.set_vertex(0, self.vertex(last_pose_id))
        e_se2.set_vertex(1, self.vertex(self.vertex_count))
        e_se2.set_measurement(rel_measure)
        e_se2.set_information(information)
        self.optimizer.add_edge(e_se2)

        self.vertex_count += 1
        self.edge_count += 1
        if self.verbose:
            print("Added odom edge (rel) between", last_pose_id, "and", self.vertex_count-1)

    def add_landmark(self, x, y, information, pose_id, landmark_id=None, add_noise=True):
        '''
        Add landmark to the graph
        '''
        relative_measurement = np.array([x, y])
        
        # Check that the pose_id is of type VertexSE2
        if type(self.optimizer.vertex(pose_id)) != g2o.VertexSE2:
            raise ValueError("The pose_id that you have provided does not correspond to a VertexSE2")
        
        trans0 = self.optimizer.vertex(pose_id).estimate()
        measurement = trans0 * relative_measurement
        
        if landmark_id is None:
            landmark_id = self.vertex_count
            v_pointxy = g2o.VertexPointXY()
            # Add significant noise to landmark initial guess
            if add_noise and self.add_noise_to_initial:
                noisy_meas = measurement + np.random.normal(0, self.initial_landmark_noise_std, 2)
                v_pointxy.set_estimate(noisy_meas)
            else:
                v_pointxy.set_estimate(measurement)
            v_pointxy.set_id(landmark_id)
            if self.verbose:
                print("Adding landmark vertex", landmark_id, "observed by", pose_id)
                print("Rel / global:", relative_measurement, measurement)
            self.optimizer.add_vertex(v_pointxy)
            self.vertex_count += 1
        # add edge
        e_pointxy = g2o.EdgeSE2PointXY()
        e_pointxy.set_vertex(0, self.vertex(pose_id))
        e_pointxy.set_vertex(1, self.vertex(landmark_id))
        self.edge_count += 1
        e_pointxy.set_measurement(relative_measurement)
        e_pointxy.set_information(information)
        self.optimizer.add_edge(e_pointxy)
        if self.verbose:
            print("Adding landmark edge between", pose_id, landmark_id)

        return landmark_id

    def GraphSLAM_initialize(self):
        print("GraphSLAM Initialize")
        
        self.mus = self.slam_data.poses

        # Add initial fixed pose without noise
        self.add_fixed_pose(g2o.SE2())

        for i in range(1, len(self.slam_data.controls)):
            mu_t_x, mu_t_y, mu_t_theta = self.mus[i]
            self.add_odometry(mu_t_x, mu_t_y, mu_t_theta, self.pose_information, 
                            add_noise=self.add_noise_to_initial)

        print("Done adding poses to the graph")
        print(f"Number of vertices: {len(self.optimizer.vertices())}")
        print(f"Number of edges {len(self.optimizer.edges())}")

    def GraphSLAM_linearize_and_reduce(self):
        print("GraphSLAM Linearize")

        observations_by_pose = {}
        for observation in self.slam_data.observations:
            pose_index, landmark_index, r, phi = observation
            if pose_index not in observations_by_pose:
                observations_by_pose[pose_index] = []
            observations_by_pose[pose_index].append((landmark_index, r, phi))
        
        # For all measurements z_t do
        added = {}
        for pose_index in sorted(observations_by_pose.keys()):
            
            observations = observations_by_pose[pose_index]  # List of observations for this pose
            
            # For all observed features z_t_i
            for i, (landmark_index, r_t_i, phi_t_i) in enumerate(observations):

                mu_j_x, mu_j_y = compute_xy_location(r_t_i, phi_t_i, 0.0)

                if landmark_index not in added:

                    landmark_g2o_index = self.add_landmark(
                        mu_j_x, mu_j_y, self.landmark_information, 
                        pose_id=pose_index, add_noise=self.add_noise_to_initial
                    )
                    added[landmark_index] = landmark_g2o_index
                else:
                    landmark_g2o_index = added[landmark_index]
                    self.add_landmark(
                        mu_j_x, mu_j_y, self.landmark_information, 
                        pose_id=pose_index, landmark_id=landmark_g2o_index,
                        add_noise=False  # Don't add noise again to same landmark
                    )
                
        print("Done adding landmarks to the graph")
        print(f"Number of vertices: {len(self.optimizer.vertices())}")
        print(f"Number of edges {len(self.optimizer.edges())}")    
        self.vertex_count_before_gt = len(self.optimizer.vertices())  


    def GraphSLAM_solve(self, iterations=10, verbose=None):
        '''
        Optimize the graph with memory safety
        '''
        try:
            print(f"Starting optimization with {iterations} iterations...")
            
            # DEBUG: Check graph structure before optimization
            print(f"Vertices: {len(self.optimizer.vertices())}")
            print(f"Edges: {len(self.optimizer.edges())}")
            
            self.optimizer.initialize_optimization()
            initial_chi2 = self.optimizer.chi2()
            
            if verbose is None:
                verbose = self.verbose
                
            # Set verbosity for debugging
            self.optimizer.set_verbose(False)  # Reduced verbosity to avoid output issues
            
            print("Starting optimization iterations...")
            # Use smaller batches to avoid memory issues
            max_iterations_per_batch = 10
            total_iterations = 0
            
            while total_iterations < iterations:
                batch_size = min(max_iterations_per_batch, iterations - total_iterations)
                self.optimizer.optimize(batch_size)
                total_iterations += batch_size
                print(f"Completed {total_iterations}/{iterations} iterations")
            
            final_chi2 = self.optimizer.chi2()
            print(f"Chi2: initial={initial_chi2:.3f} final={final_chi2:.3f} delta={initial_chi2-final_chi2:.3f}")
            
            return final_chi2
            
        except Exception as e:
            print(f"Optimization error: {e}")
            return None
        
    
    def cleanup(self):
        """Safe cleanup to prevent memory corruption"""
        try:
            if hasattr(self, 'optimizer') and self.optimizer:
                # Clear all vertices and edges
                self.optimizer.clear()
                self.optimizer = None
            
            # Clear other components
            self.solver = None
            self.algorithm = None
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Cleanup warning: {e}")


    def plot_ground_truth(self):
        """Plot ground truth for comparison"""
        # Store current state
        current_vertices = len(self.optimizer.vertices())
        
        # Create a clean optimizer for ground truth
        gt_optimizer = g2o.SparseOptimizer()
        gt_optimizer.set_algorithm(self.algorithm)
        
        # Add ground truth poses (no noise)
        vertex_count = 0
        for i, pose in enumerate(self.slam_data.poses):
            v_se2 = g2o.VertexSE2()
            v_se2.set_id(vertex_count)
            v_se2.set_estimate(g2o.SE2(pose[0], pose[1], pose[2]))
            if i == 0:
                v_se2.set_fixed(True)
            gt_optimizer.add_vertex(v_se2)
            vertex_count += 1
        
        # Add ground truth landmarks (no noise)
        landmark_positions = {}
        for obs in self.slam_data.observations:
            pose_index, landmark_index, r, phi = obs
            if landmark_index not in landmark_positions:
                # Compute landmark position in global frame
                pose = self.slam_data.poses[pose_index]
                pose_se2 = g2o.SE2(pose[0], pose[1], pose[2])
                rel_pos = compute_xy_location(r, phi, 0.0)
                global_pos = pose_se2 * rel_pos
                landmark_positions[landmark_index] = global_pos
                
                v_pointxy = g2o.VertexPointXY()
                v_pointxy.set_id(vertex_count)
                v_pointxy.set_estimate(global_pos)
                gt_optimizer.add_vertex(v_pointxy)
                vertex_count += 1
        
        return self.plot_comparison(
            self.optimizer, 
            gt_optimizer, 
            "Ground Truth vs Optimized"
        )

    def plot_comparison(self, optimized_optimizer, gt_optimizer, title):
        """Plot comparison between optimized result and ground truth"""
        fig = go.Figure()

        # Plot optimized poses
        opt_pose_vertices = [v for v in optimized_optimizer.vertices().values() 
                           if isinstance(v, g2o.VertexSE2)]
        fig.add_trace(
            go.Scatter(
                x=[v.estimate()[0] for v in opt_pose_vertices],
                y=[v.estimate()[1] for v in opt_pose_vertices],
                mode="markers+lines",
                marker=dict(color="red", size=9),
                line=dict(color="red", width=2),
                opacity=0.75,
                name="Optimized Poses"
            )
        )

        # Plot ground truth poses
        gt_pose_vertices = [v for v in gt_optimizer.vertices().values() 
                          if isinstance(v, g2o.VertexSE2)]
        fig.add_trace(
            go.Scatter(
                x=[v.estimate()[0] for v in gt_pose_vertices],
                y=[v.estimate()[1] for v in gt_pose_vertices],
                mode="markers+lines",
                marker=dict(color="green", size=7),
                line=dict(color="green", width=2, dash="dash"),
                name="Ground Truth Poses"
            )
        )

        # Plot optimized landmarks
        opt_landmark_vertices = [v for v in optimized_optimizer.vertices().values() 
                               if isinstance(v, g2o.VertexPointXY)]
        fig.add_trace(
            go.Scatter(
                x=[v.estimate()[0] for v in opt_landmark_vertices],
                y=[v.estimate()[1] for v in opt_landmark_vertices],
                mode="markers",
                marker=dict(color="blue", size=7, symbol="circle"),
                name="Optimized Landmarks"
            )
        )

        # Plot ground truth landmarks  
        gt_landmark_vertices = [v for v in gt_optimizer.vertices().values() 
                              if isinstance(v, g2o.VertexPointXY)]
        fig.add_trace(
            go.Scatter(
                x=[v.estimate()[0] for v in gt_landmark_vertices],
                y=[v.estimate()[1] for v in gt_landmark_vertices],
                mode="markers",
                marker=dict(color="orange", size=7, symbol="x"),
                name="Ground Truth Landmarks"
            )
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            title=title,
            showlegend=True,
            width=1000,
            height=800
        )

        return fig
    

    def plot_slam2d(self, optimizer, title, plot_edges=True, edges_opacity=0.4, covariances=None):
        def edges_coord(edges, dim):
            for e in edges:
                yield e.vertices()[0].estimate()[dim]
                yield e.vertices()[1].estimate()[dim]
                yield None

        fig = go.Figure()

        # poses of the vertices
        vertices = optimizer.vertices()
        
        pose_vertices = [v for v in vertices.values() if isinstance(v, g2o.VertexSE2)]
        landmark_vertices = [v for v in vertices.values() if isinstance(v, g2o.VertexPointXY)]

        # Plot each category with a different color
        fig.add_trace(
            go.Scatter(
                x=[v.estimate()[0] for v in pose_vertices],
                y=[v.estimate()[1] for v in pose_vertices],
                mode="markers+lines",
                marker=dict(color="red", size=9),
                line=dict(color="red", width=2),
                opacity=0.75,
                name="Robot Poses"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[v.estimate()[0] for v in landmark_vertices],
                y=[v.estimate()[1] for v in landmark_vertices],
                mode="markers",
                marker=dict(color="blue", size=7),
                name="Landmarks"
            )
        )

        # edges
        if plot_edges:
            edges = optimizer.edges()  # get set once to have same order
            fig.add_trace(
                go.Scatter(
                    x=list(edges_coord(edges, 0)),
                    y=list(edges_coord(edges, 1)),
                    mode="lines",
                    line=dict(width=0.55, color="pink"),
                    opacity=edges_opacity,
                    name="Constraints"
                )
            )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            go.Layout({
                "title": title, 
                "showlegend": True,
                "width": 1000,
                "height": 800
            })
        )

        return fig
    

def main():
    
    # Example usage
    slam_data = GraphSLAMData.load_from_file("firstmap.pkl")

    print("Controls", len(slam_data.controls), "Landmarks", len(slam_data.landmarks), "Observations", len(slam_data.observations))
    
    # Test both optimization methods
    methods = ["levenberg", "dogleg"]

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()} method")
        print(f"{'='*50}")
        
        # Create GraphSLAM with significant noise
        graph_slam = GraphSLAM(
            slam_data, 
            verbose=False, 
            method=method,
            noise_level=1.5,  # Increased noise level
            add_noise_to_initial=True
        )
        # Initialize with noise
        graph_slam.GraphSLAM_initialize()
        graph_slam.GraphSLAM_linearize_and_reduce()

        # Show initial noisy state
        initial_fig = graph_slam.plot_slam2d(graph_slam.optimizer, f"Initial State ({method})")
        initial_fig.show()

        # Optimize
        print(f"Optimizing with {method}...")
        chi2 = graph_slam.GraphSLAM_solve(iterations=100)

        # Show optimized state
        optimized_fig = graph_slam.plot_slam2d(graph_slam.optimizer, f"Optimized State ({method})")
        optimized_fig.show()

        # Show comparison with ground truth
        comparison_fig = graph_slam.plot_ground_truth()
        comparison_fig.show()
        graph_slam.cleanup()

        print(f"{method} optimization completed. Final chi2: {chi2:.3f}")
    
if __name__ == '__main__':
    main()