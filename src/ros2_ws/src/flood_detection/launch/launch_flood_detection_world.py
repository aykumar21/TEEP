# launch_flood_detection_world.py

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import launch_ros

def generate_launch_description():
    return LaunchDescription([
        # Launch the Gazebo simulation with the iris_downward_depth_camera model
        Node(
            package='gazebo_ros',
            executable='gazebo',
            name='gazebo',
            output='screen',
            parameters=[{
                'use_sim_time': True
            }],
            arguments=['-s', 'libgazebo_ros_init.so']
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_model',
            name='spawn_model',
            output='screen',
            arguments=[
                '-file', '/home/ayushkumar/TEEP/src/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds/flood_detection.world',
                '-sdf', '-model', 'iris_downward_depth_camera'
            ]
        ),
        # Launch MAVROS for communication
        Node(
            package='mavros',
            executable='mavros_node',
            name='mavros',
            output='screen',
            parameters=[{
                'fcu_url': 'udp://:14540'
            }]
        ),
    ])
