#!/usr/bin/env python3
import sys, tty, termios, threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

STEP_XY = 0.5   # metres per keypress (horizontal)
STEP_Z  = 0.3   # metres per keypress (vertical)
Z_MIN   = 0.2   # don't command below ground

BINDS = {
    'w': ( STEP_XY, 0,       0),
    's': (-STEP_XY, 0,       0),
    'd': (0,        STEP_XY, 0),  # ROS: +Y is left in world frame
    'a': (0,       -STEP_XY, 0),
    'q': (0,        0,       STEP_Z),
    'e': (0,        0,      -STEP_Z),
}


def _get_key():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop')
        self._pub = self.create_publisher(Point, '/drone/setpoint', 1)
        self._pos = [0.0, 0.0, 1.5]
        self._publish()
        self.get_logger().info(
            'Teleop ready\n'
            '  W/S  forward / back\n'
            '  A/D  left / right\n'
            '  Q/E  up / down\n'
            '  Ctrl+C  quit'
        )

    def _publish(self):
        msg = Point()
        msg.x, msg.y, msg.z = self._pos
        self._pub.publish(msg)

    def handle_key(self, key):
        if key not in BINDS:
            return
        dx, dy, dz = BINDS[key]
        self._pos[0] += dx
        self._pos[1] += dy
        self._pos[2] = max(Z_MIN, self._pos[2] + dz)
        self._publish()
        self.get_logger().info(
            f'setpoint  x={self._pos[0]:.1f}  y={self._pos[1]:.1f}  z={self._pos[2]:.1f}'
        )


def main():
    rclpy.init()
    node = TeleopNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            key = _get_key()
            if key in ('\x03', '\x1b'):   # Ctrl+C or Escape
                break
            node.handle_key(key)
    except Exception:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
