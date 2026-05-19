#!/usr/bin/env python3
import sys, os
import warnings
import numpy as np

try:
    from ament_index_python.packages import get_package_prefix as _gpp
    _lib_dir = os.path.join(_gpp('drone_sim'), 'lib', 'drone_sim')
except Exception:
    _lib_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _lib_dir)

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import QuaternionStamped

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
from stable_baselines3 import PPO

W_HOVER          = 197.92
W_SCALE          = 85.0
ENGAGE_QW        = 0.707   # engage when tilt > 45 deg
DISENGAGE_QW     = 0.940   # disengage when tilt < 20 deg (hysteresis)
POLICY_PATH      = os.path.join(_lib_dir, 'quad_recovery_policy')


class RLRecoveryNode(Node):
    def __init__(self):
        super().__init__('rl_recovery')

        self._model = PPO.load(POLICY_PATH)
        self.get_logger().info(f'Policy loaded from {POLICY_PATH}.zip')

        # 13-dim state: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
        self._obs = np.zeros(13, dtype=np.float32)
        self._obs[2] = 1.0   # pz = 1 m
        self._obs[6] = 1.0   # qw = 1 (upright)

        self._active = False

        self._motor_pub = self.create_publisher(
            Float64MultiArray, '/drone/motor_speeds', 10)

        self.create_subscription(Odometry,          '/drone/odom',   self._odom_cb,     10)
        self.create_subscription(QuaternionStamped, '/imu/attitude', self._attitude_cb, 10)

        self.create_timer(0.005, self._step)
        self.get_logger().info('RL recovery node running at 200 Hz')

    # ------------------------------------------------------------------
    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        q = msg.pose.pose.orientation

        self._obs[0:3]   = [p.x, p.y, p.z]
        self._obs[3:6]   = [v.x, v.y, v.z]
        self._obs[6:10]  = [q.w, q.x, q.y, q.z]
        self._obs[10:13] = [w.x, w.y, w.z]

    def _attitude_cb(self, msg):
        q = msg.quaternion
        self._obs[6:10] = [q.w, q.x, q.y, q.z]

    def _step(self):
        qw = float(self._obs[6])

        if not self._active and qw < ENGAGE_QW:
            self._active = True
            self.get_logger().warn(
                f'RL RECOVERY ENGAGED  tilt={self._tilt_deg(qw):.1f} deg')

        elif self._active and qw > DISENGAGE_QW:
            self._active = False
            self.get_logger().info(
                f'RL recovery disengaged  tilt={self._tilt_deg(qw):.1f} deg')

        if not self._active:
            return

        action, _ = self._model.predict(self._obs, deterministic=True)
        motors = [float(np.clip(W_HOVER + a * W_SCALE, 50.0, 400.0))
                  for a in action]

        msg = Float64MultiArray()
        msg.data = motors
        self._motor_pub.publish(msg)

    @staticmethod
    def _tilt_deg(qw):
        return 2.0 * np.degrees(np.arccos(np.clip(abs(qw), 0.0, 1.0)))


def main():
    rclpy.init()
    node = RLRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
