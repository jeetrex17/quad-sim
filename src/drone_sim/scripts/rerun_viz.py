#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import QuaternionStamped
import rerun as rr

HOST = "host.docker.internal"


def quat_to_euler(w, x, y, z):
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1, min(1, 2*(w*y-z*x))))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class RerunVizNode(Node):
    def __init__(self):
        super().__init__("rerun_viz")

        rr.init("quad_sim", spawn=False)
        rr.connect_grpc(f"rerun+http://{HOST}:9876/proxy")

        self.create_subscription(Odometry,         "/drone/odom",    self.odom_cb,    10)
        self.create_subscription(Imu,              "/imu/data_raw",  self.imu_cb,     10)
        self.create_subscription(QuaternionStamped,"/imu/attitude",  self.attitude_cb,10)
        self.get_logger().info(f"rerun_viz streaming to {HOST}:9876")

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        rr.log("drone/position", rr.Points3D(
            [[p.x, p.y, p.z]],
            radii=[0.05],
            colors=[[0, 180, 255]],
        ))

        rr.log("drone/transform", rr.Transform3D(
            translation=[p.x, p.y, p.z],
            quaternion=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
        ))

        rr.log("drone/altitude",   rr.Scalars(p.z))
        rr.log("drone/velocity/x", rr.Scalars(v.x))
        rr.log("drone/velocity/y", rr.Scalars(v.y))
        rr.log("drone/velocity/z", rr.Scalars(v.z))

        # True attitude from dynamics (ground truth)
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
