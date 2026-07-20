#!/usr/bin/env python3

"""
Launch file for the Vehicle Odometry CSV Recorder node.

Usage:
    ros2 launch uav_sim vehicle_odometry_recorder.launch.py
    ros2 launch uav_sim vehicle_odometry_recorder.launch.py output_dir:=./data
    ros2 launch uav_sim vehicle_odometry_recorder.launch.py output_dir:=./data filename_prefix:=drone1
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    """Generate launch description for vehicle_odometry_recorder."""

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='.',
        description='Directory to save the CSV output file.',
    )

    filename_prefix_arg = DeclareLaunchArgument(
        'filename_prefix',
        default_value='vehicle_odometry',
        description='Prefix for the CSV filename (timestamp will be appended).',
    )

    vehicle_odometry_recorder_node = Node(
        package='uav_sim',
        executable='vehicle_odometry_recorder',
        name='vehicle_odometry_recorder',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'filename_prefix': LaunchConfiguration('filename_prefix'),
        }],
    )

    return LaunchDescription([
        output_dir_arg,
        filename_prefix_arg,
        vehicle_odometry_recorder_node,
    ])
