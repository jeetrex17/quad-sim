from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    dynamics = Node(
        package='drone_sim',
        executable='drone_dynamics',
        output='screen',
    )

    mekf = Node(
        package='drone_sim',
        executable='mekf',
        output='screen',
    )

    pid = Node(
        package='drone_sim',
        executable='pid_controller.py',
        output='screen',
    )

    viz = Node(
        package='drone_sim',
        executable='rerun_viz.py',
        output='screen',
    )

    return LaunchDescription([
        dynamics,
        mekf,
        viz,
        # wait for dynamics to be up before PID sends motor commands
        TimerAction(period=1.5, actions=[pid]),
    ])
