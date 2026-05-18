#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import QuaternionStamped
import rerun as rr

HOST = "host.docker.internal"

# X-config arm half-length (QD_L=0.17 / sqrt(2))
_ARM = 0.17 / math.sqrt(2)

# Motor positions in body frame: front-right, front-left, back-left, back-right
_MOTORS_BODY = np.array([
    [ _ARM, -_ARM, 0.0],
    [ _ARM,  _ARM, 0.0],
    [-_ARM,  _ARM, 0.0],
    [-_ARM, -_ARM, 0.0],
], dtype=np.float64)


def _qrot(qw, qx, qy, qz, pts):
    """Rotate Nx3 array of points from body to world frame."""
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])
    return (R @ pts.T).T


def quat_to_euler(w, x, y, z):
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _log_world():
    """Static world geometry logged once at startup."""
    s = 5.0
    rr.log("world/ground", rr.Mesh3D(
        vertex_positions=[[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]],
        triangle_indices=[[0,1,2],[0,2,3]],
        vertex_colors=[[45,45,45,220]]*4,
    ), static=True)

    # Grid lines along X
    grid_lines = []
    for i in range(-5, 6):
        grid_lines.append([[-5.0, float(i), 0.001], [5.0, float(i), 0.001]])
        grid_lines.append([[float(i), -5.0, 0.001], [float(i),  5.0, 0.001]])
    rr.log("world/grid", rr.LineStrips3D(
        strips=grid_lines,
        radii=0.005,
        colors=[[80, 80, 80]],
    ), static=True)


class RerunVizNode(Node):
    def __init__(self):
        super().__init__("rerun_viz")

        rr.init("quad_sim", spawn=False)
        rr.connect_grpc(f"rerun+http://{HOST}:9876/proxy")

        _log_world()

        self.create_subscription(Odometry,          "/drone/odom",   self.odom_cb,     10)
        self.create_subscription(Imu,               "/imu/data_raw", self.imu_cb,      10)
        self.create_subscription(QuaternionStamped, "/imu/attitude", self.attitude_cb, 10)
        self.get_logger().info(f"rerun_viz streaming to {HOST}:9876")

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        center = np.array([p.x, p.y, p.z])
        motors = _qrot(q.w, q.x, q.y, q.z, _MOTORS_BODY) + center

        # Arms: cross from motor 0<->2 and 1<->3
        rr.log("drone/arms", rr.LineStrips3D(
            strips=[
                [motors[0], motors[2]],
                [motors[1], motors[3]],
            ],
            radii=0.008,
            colors=[[200, 200, 200]],
        ))

        # Motor hubs (propeller positions)
        rr.log("drone/motors", rr.Points3D(
            motors,
            radii=0.022,
            colors=[[255, 60, 60]],
        ))

        # Center body
        rr.log("drone/body_center", rr.Points3D(
            [center],
            radii=0.018,
            colors=[[30, 144, 255]],
        ))

        # Thrust arrow: body +Z in world frame
        z_world = _qrot(q.w, q.x, q.y, q.z, np.array([[0, 0, 0.18]]))[0]
        rr.log("drone/thrust", rr.Arrows3D(
            origins=[center],
            vectors=[z_world],
            radii=0.006,
            colors=[[0, 230, 100]],
        ))

        rr.log("drone/altitude",   rr.Scalars(p.z))
        rr.log("drone/velocity/x", rr.Scalars(v.x))
        rr.log("drone/velocity/y", rr.Scalars(v.y))
        rr.log("drone/velocity/z", rr.Scalars(v.z))

        roll, pitch, yaw = quat_to_euler(q.w, q.x, q.y, q.z)
        rr.log("attitude/true/roll",  rr.Scalars(roll))
        rr.log("attitude/true/pitch", rr.Scalars(pitch))
        rr.log("attitude/true/yaw",   rr.Scalars(yaw))

    def imu_cb(self, msg: Imu):
        rr.log("imu/gyro/x",  rr.Scalars(msg.angular_velocity.x))
        rr.log("imu/gyro/y",  rr.Scalars(msg.angular_velocity.y))
        rr.log("imu/gyro/z",  rr.Scalars(msg.angular_velocity.z))
        rr.log("imu/accel/z", rr.Scalars(msg.linear_acceleration.z))

    def attitude_cb(self, msg: QuaternionStamped):
        q = msg.quaternion
        roll, pitch, yaw = quat_to_euler(q.w, q.x, q.y, q.z)
        rr.log("attitude/mekf/roll",  rr.Scalars(roll))
        rr.log("attitude/mekf/pitch", rr.Scalars(pitch))
        rr.log("attitude/mekf/yaw",   rr.Scalars(yaw))


def main():
    rclpy.init()
    node = RerunVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
