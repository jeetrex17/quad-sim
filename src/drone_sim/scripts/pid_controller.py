#!/usr/bin/env python3
"""
Cascaded PID position controller for the quadrotor simulator.

Outer loop (50 Hz): position error  -> desired thrust + desired attitude
Inner loop (200 Hz): attitude error -> torques -> motor speeds

Motor mixing inverse matches drone_dynamics_node.cpp exactly.
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

# Must match drone_dynamics_node.cpp
MASS    = 0.5
GRAVITY = 9.81
KT      = 3.13e-5
KQ      = 7.5e-7
L       = 0.17
DT      = 0.005   # 200 Hz odom rate


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def quat_to_euler(qw, qx, qy, qz):
    roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = math.asin(clamp(2*(qw*qy - qz*qx), -1.0, 1.0))
    yaw   = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return roll, pitch, yaw


def motor_mixing(F, tau_x, tau_y, tau_z):
    """(thrust [N], body torques [N*m]) -> motor speeds [rad/s]."""
    w_sq = [
        F/(4*KT) - tau_y/(2*KT*L) - tau_z/(4*KQ),   # M1
        F/(4*KT) + tau_x/(2*KT*L) + tau_z/(4*KQ),   # M2
        F/(4*KT) + tau_y/(2*KT*L) - tau_z/(4*KQ),   # M3
        F/(4*KT) - tau_x/(2*KT*L) + tau_z/(4*KQ),   # M4
    ]
    return [math.sqrt(max(0.0, w)) for w in w_sq]


class PID:
    def __init__(self, kp, ki, kd, i_limit=5.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self._i = 0.0

    def step(self, error, rate, dt):
        self._i = clamp(self._i + error * dt, -self.i_limit, self.i_limit)
        return self.kp * error + self.ki * self._i - self.kd * rate


class Waypoints:
    # (time_seconds, x, y, z)
    WP = [
        ( 0.0,  0.0,  0.0, 1.0),
        ( 5.0,  0.0,  0.0, 3.0),   # climb
        ( 9.0,  2.0,  0.0, 3.0),   # right
        (13.0,  2.0,  2.0, 3.0),   # forward
        (17.0,  0.0,  2.0, 3.0),   # left
        (21.0,  0.0,  0.0, 3.0),   # back
        (25.0,  0.0,  0.0, 1.0),   # descend
    ]

    def at(self, t):
        wps = self.WP
        if t >= wps[-1][0]:
            return wps[-1][1], wps[-1][2], wps[-1][3]
        for i in range(len(wps) - 1):
            t0, x0, y0, z0 = wps[i]
            t1, x1, y1, z1 = wps[i+1]
            if t0 <= t < t1:
                a = (t - t0) / (t1 - t0)
                return x0+a*(x1-x0), y0+a*(y1-y0), z0+a*(z1-z0)
        return wps[0][1], wps[0][2], wps[0][3]


class QuadPID(Node):
    def __init__(self):
        super().__init__("quad_pid")

        self.traj    = Waypoints()
        self.t0      = None

        # Outer loop: position
        self.pid_z = PID(kp=6.0, ki=0.3, kd=4.0)
        self.pid_x = PID(kp=0.5, ki=0.0, kd=1.5)
        self.pid_y = PID(kp=0.5, ki=0.0, kd=1.5)

        # Inner loop: attitude
        self.pid_roll  = PID(kp=6.0, ki=0.0, kd=1.0, i_limit=0.5)
        self.pid_pitch = PID(kp=6.0, ki=0.0, kd=1.0, i_limit=0.5)
        self.pid_yaw   = PID(kp=2.0, ki=0.0, kd=0.5, i_limit=0.5)

        self.motor_pub = self.create_publisher(Float64MultiArray, "/drone/motor_speeds", 10)
        self.create_subscription(Odometry, "/drone/odom", self.odom_cb, 10)

        self.get_logger().info("PID controller ready, following waypoint trajectory")

    def odom_cb(self, msg: Odometry):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = stamp
        t = stamp - self.t0

        # Current state
        p  = msg.pose.pose.position
        v  = msg.twist.twist.linear
        q  = msg.pose.pose.orientation
        wb = msg.twist.twist.angular

        roll, pitch, yaw = quat_to_euler(q.w, q.x, q.y, q.z)

        # Setpoint from trajectory
        xd, yd, zd = self.traj.at(t)

        # --- Outer loop: position -> thrust + desired attitude ---
        F = MASS * GRAVITY + self.pid_z.step(zd - p.z, v.z, DT)
        F = clamp(F, 0.0, 2.5 * MASS * GRAVITY)

        ax_d = self.pid_x.step(xd - p.x, v.x, DT)
        ay_d = self.pid_y.step(yd - p.y, v.y, DT)

        # Positive pitch (nose down) tilts thrust toward +X -> forward acceleration
        # Positive roll tilts thrust toward -Y -> need negative roll for +Y motion
        pitch_d = clamp( ax_d / GRAVITY, -0.35, 0.35)
        roll_d  = clamp(-ay_d / GRAVITY, -0.35, 0.35)

        # --- Inner loop: attitude -> torques ---
        tau_x = self.pid_roll.step( roll_d  - roll,  wb.x, DT)
        tau_y = self.pid_pitch.step(pitch_d - pitch, wb.y, DT)
        tau_z = self.pid_yaw.step(  0.0     - yaw,   wb.z, DT)

        speeds = motor_mixing(F, tau_x, tau_y, tau_z)

        out = Float64MultiArray()
        out.data = speeds
        self.motor_pub.publish(out)

        if not hasattr(self, "_last_log") or (t - self._last_log) >= 2.0:
            self._last_log = t
            self.get_logger().info(
                f"t={t:.1f}s  pos=({p.x:.2f},{p.y:.2f},{p.z:.2f})"
                f"  target=({xd:.1f},{yd:.1f},{zd:.1f})"
            )


def main():
    rclpy.init()
    node = QuadPID()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
