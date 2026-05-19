#!/usr/bin/env python3
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, QuaternionStamped
from std_msgs.msg import Float64MultiArray
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

_WAYPOINTS = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 3.0],
    [2.0, 0.0, 3.0],
    [2.0, 2.0, 3.0],
    [0.0, 2.0, 3.0],
    [0.0, 0.0, 3.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

_WAYPOINT_TIMES = np.array([0.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0])


def _qrot(qw, qx, qy, qz, pts):
    """Rotate Nx3 array of points from body to world frame."""
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ])
    return (R @ pts.T).T


def _stamp_seconds(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def quat_to_euler(w, x, y, z):
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _scalar(value):
    if hasattr(rr, "Scalars"):
        return rr.Scalars(value)
    return rr.Scalar(value)


def _waypoint_at(t):
    if t <= _WAYPOINT_TIMES[0]:
        return _WAYPOINTS[0]
    if t >= _WAYPOINT_TIMES[-1]:
        return _WAYPOINTS[-1]

    idx = int(np.searchsorted(_WAYPOINT_TIMES, t, side="right") - 1)
    t0 = _WAYPOINT_TIMES[idx]
    t1 = _WAYPOINT_TIMES[idx + 1]
    a = (t - t0) / (t1 - t0)
    return _WAYPOINTS[idx] + a * (_WAYPOINTS[idx + 1] - _WAYPOINTS[idx])


def _mesh_box(center, size, color):
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, dtype=np.float64) / 2.0
    vertices = np.array([
        [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
        [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
        [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
        [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint32)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1))
    return vertices, triangles, colors


def _mesh_disc(center, radius, color, segments=48):
    cx, cy, cz = center
    vertices = [[cx, cy, cz]]
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        vertices.append([cx + radius * math.cos(a), cy + radius * math.sin(a), cz])

    triangles = []
    for i in range(1, segments + 1):
        triangles.append([0, i, 1 if i == segments else i + 1])

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.uint32)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1))
    return vertices, triangles, colors


def _mesh_cylinder(center, radius, height, color, segments=32):
    cx, cy, cz = center
    z0 = cz - height / 2.0
    z1 = cz + height / 2.0
    vertices = [[cx, cy, z0], [cx, cy, z1]]

    for z in (z0, z1):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            vertices.append([cx + radius * math.cos(a), cy + radius * math.sin(a), z])

    triangles = []
    bottom_start = 2
    top_start = 2 + segments
    for i in range(segments):
        j = 0 if i + 1 == segments else i + 1
        triangles.append([0, bottom_start + j, bottom_start + i])
        triangles.append([1, top_start + i, top_start + j])
        triangles.append([bottom_start + i, bottom_start + j, top_start + j])
        triangles.append([bottom_start + i, top_start + j, top_start + i])

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.uint32)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1))
    return vertices, triangles, colors


def _combine_meshes(meshes):
    vertices = []
    triangles = []
    colors = []
    offset = 0
    for verts, tris, cols in meshes:
        vertices.append(verts)
        triangles.append(tris + offset)
        colors.append(cols)
        offset += len(verts)

    return (
        np.vstack(vertices),
        np.vstack(triangles),
        np.vstack(colors),
    )


def _transform_mesh(mesh, center, q):
    vertices, triangles, colors = mesh
    qw, qx, qy, qz = q
    return _qrot(qw, qx, qy, qz, vertices) + center, triangles, colors


def _flight_cage_lines(xy=3.2, z=4.2):
    corners = [
        [-xy, -xy, 0.0], [xy, -xy, 0.0], [xy, xy, 0.0], [-xy, xy, 0.0],
        [-xy, -xy, z],   [xy, -xy, z],   [xy, xy, z],   [-xy, xy, z],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [[corners[a], corners[b]] for a, b in edges]


def _gate_mesh(x, y, z, width=1.0, height=1.1, post=0.04):
    color = [0, 165, 255, 230]
    return _combine_meshes([
        _mesh_box([x, y - width / 2.0, z + height / 2.0], [post, post, height], color),
        _mesh_box([x, y + width / 2.0, z + height / 2.0], [post, post, height], color),
        _mesh_box([x, y, z + height], [post, width + post, post], color),
    ])


def _drone_body_mesh():
    meshes = [
        _mesh_box([0.0, 0.0, 0.0], [0.25, 0.18, 0.055], [28, 32, 40, 255]),
        _mesh_box([0.055, 0.0, 0.035], [0.12, 0.10, 0.025], [0, 145, 255, 255]),
        _mesh_box([-0.08, 0.0, -0.035], [0.16, 0.095, 0.025], [55, 60, 72, 255]),
    ]
    for motor in _MOTORS_BODY:
        meshes.append(_mesh_cylinder([motor[0], motor[1], 0.0], 0.038, 0.055, [32, 32, 36, 255], 28))
        meshes.append(_mesh_disc([motor[0], motor[1], 0.045], 0.125, [70, 170, 255, 72], 48))
    return _combine_meshes(meshes)


def _log_world():
    """Static world geometry logged once at startup."""
    try:
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    except AttributeError:
        pass

    s = 5.0
    rr.log("world/ground", rr.Mesh3D(
        vertex_positions=[[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]],
        triangle_indices=[[0,1,2],[0,2,3]],
        vertex_colors=[[32, 34, 38, 255]] * 4,
    ), static=True)

    pad_meshes = [
        _mesh_disc([0.0, 0.0, 0.012], 0.48, [35, 35, 35, 255], 72),
        _mesh_disc([0.0, 0.0, 0.014], 0.34, [245, 245, 245, 255], 72),
        _mesh_disc([0.0, 0.0, 0.016], 0.24, [35, 35, 35, 255], 72),
        _mesh_box([0.0, 0.0, 0.02], [0.09, 0.45, 0.01], [255, 190, 40, 255]),
        _mesh_box([0.0, 0.0, 0.022], [0.40, 0.08, 0.01], [255, 190, 40, 255]),
    ]
    pad_v, pad_t, pad_c = _combine_meshes(pad_meshes)
    rr.log("world/landing_pad", rr.Mesh3D(
        vertex_positions=pad_v,
        triangle_indices=pad_t,
        vertex_colors=pad_c,
    ), static=True)

    grid_lines = []
    for i in range(-5, 6):
        grid_lines.append([[-5.0, float(i), 0.001], [5.0, float(i), 0.001]])
        grid_lines.append([[float(i), -5.0, 0.001], [float(i),  5.0, 0.001]])
    rr.log("world/grid", rr.LineStrips3D(
        strips=grid_lines,
        radii=0.005,
        colors=[[80, 80, 80]],
    ), static=True)

    rr.log("world/flight_cage", rr.LineStrips3D(
        strips=_flight_cage_lines(),
        radii=0.011,
        colors=[[90, 140, 255, 190]],
    ), static=True)

    obstacle_meshes = [
        _mesh_box([1.15, -1.15, 0.45], [0.18, 0.18, 0.90], [220, 70, 55, 235]),
        _mesh_box([-1.25, 1.25, 0.65], [0.20, 0.20, 1.30], [220, 70, 55, 235]),
        _gate_mesh(1.95, 0.85, 0.80),
        _gate_mesh(-1.85, -0.85, 0.80),
    ]
    obs_v, obs_t, obs_c = _combine_meshes(obstacle_meshes)
    rr.log("world/test_obstacles", rr.Mesh3D(
        vertex_positions=obs_v,
        triangle_indices=obs_t,
        vertex_colors=obs_c,
    ), static=True)

    rr.log("mission/waypoints", rr.Points3D(
        _WAYPOINTS,
        radii=0.055,
        colors=[[255, 210, 0]],
    ), static=True)
    rr.log("mission/reference_path", rr.LineStrips3D(
        strips=[_WAYPOINTS.tolist()],
        radii=0.012,
        colors=[[255, 210, 0]],
    ), static=True)

    rr.log("world/axes", rr.Arrows3D(
        origins=[[0.0, 0.0, 0.03], [0.0, 0.0, 0.03], [0.0, 0.0, 0.03]],
        vectors=[[0.55, 0.0, 0.0], [0.0, 0.55, 0.0], [0.0, 0.0, 0.55]],
        radii=0.012,
        colors=[[255, 60, 60], [80, 220, 80], [80, 150, 255]],
    ), static=True)


class RerunVizNode(Node):
    def __init__(self):
        super().__init__("rerun_viz")

        rr.init("quad_sim", spawn=False)
        rr.connect_grpc(f"rerun+http://{HOST}:9876/proxy")

        _log_world()
        self._drone_mesh = _drone_body_mesh()
        self._trail = deque(maxlen=1400)
        self._t0 = None
        self._last_motors = None
        self._manual_target = None

        self.create_subscription(Odometry,          "/drone/odom",   self.odom_cb,     10)
        self.create_subscription(Imu,               "/imu/data_raw", self.imu_cb,      10)
        self.create_subscription(QuaternionStamped, "/imu/attitude", self.attitude_cb, 10)
        self.create_subscription(Float64MultiArray, "/drone/motor_speeds", self.motors_cb, 10)
        self.create_subscription(Point,             "/drone/setpoint", self.setpoint_cb, 10)
        self.get_logger().info(f"rerun_viz streaming to {HOST}:9876")

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        center = np.array([p.x, p.y, p.z])
        quat = [q.w, q.x, q.y, q.z]
        motors = _qrot(q.w, q.x, q.y, q.z, _MOTORS_BODY) + center

        stamp = _stamp_seconds(msg)
        if self._t0 is None:
            self._t0 = stamp
        t = stamp - self._t0
        target = self._manual_target if self._manual_target is not None else _waypoint_at(t)
        err = float(np.linalg.norm(center - target))

        self._trail.append(center.tolist())

        body_v, body_t, body_c = _transform_mesh(self._drone_mesh, center, quat)
        rr.log("world/drone/model", rr.Mesh3D(
            vertex_positions=body_v,
            triangle_indices=body_t,
            vertex_colors=body_c,
        ))

        rr.log("drone/arms", rr.LineStrips3D(
            strips=[
                [motors[0], motors[2]],
                [motors[1], motors[3]],
            ],
            radii=0.018,
            colors=[[215, 220, 225]],
        ))

        rr.log("drone/motors", rr.Points3D(
            motors,
            radii=0.035,
            colors=[[255, 80, 70], [80, 170, 255], [255, 80, 70], [80, 170, 255]],
        ))

        axes = _qrot(q.w, q.x, q.y, q.z, np.eye(3) * 0.32)
        z_world = _qrot(q.w, q.x, q.y, q.z, np.array([[0, 0, 0.18]]))[0]
        vel_vec = np.array([v.x, v.y, v.z]) * 0.18
        rr.log("drone/thrust", rr.Arrows3D(
            origins=[center],
            vectors=[z_world],
            radii=0.006,
            colors=[[0, 230, 100]],
        ))
        rr.log("drone/body_axes", rr.Arrows3D(
            origins=[center, center, center],
            vectors=axes,
            radii=0.005,
            colors=[[255, 60, 60], [80, 220, 80], [80, 150, 255]],
        ))
        rr.log("drone/velocity_vector", rr.Arrows3D(
            origins=[center],
            vectors=[vel_vec],
            radii=0.005,
            colors=[[255, 255, 255]],
        ))

        shadow_v, shadow_t, shadow_c = _mesh_disc([p.x, p.y, 0.02], 0.18, [0, 0, 0, 95], 48)
        rr.log("drone/shadow", rr.Mesh3D(
            vertex_positions=shadow_v,
            triangle_indices=shadow_t,
            vertex_colors=shadow_c,
        ))

        if len(self._trail) >= 2:
            rr.log("mission/actual_trail", rr.LineStrips3D(
                strips=[list(self._trail)],
                radii=0.01,
                colors=[[30, 190, 255]],
            ))

        rr.log("mission/current_target", rr.Points3D(
            [target],
            radii=0.065,
            colors=[[255, 80, 220]],
        ))
        rr.log("mission/tracking_error", _scalar(err))

        rr.log("drone/altitude",   _scalar(p.z))
        rr.log("drone/velocity/x", _scalar(v.x))
        rr.log("drone/velocity/y", _scalar(v.y))
        rr.log("drone/velocity/z", _scalar(v.z))

        roll, pitch, yaw = quat_to_euler(q.w, q.x, q.y, q.z)
        rr.log("attitude/true/roll",  _scalar(roll))
        rr.log("attitude/true/pitch", _scalar(pitch))
        rr.log("attitude/true/yaw",   _scalar(yaw))

    def imu_cb(self, msg: Imu):
        rr.log("imu/gyro/x",  _scalar(msg.angular_velocity.x))
        rr.log("imu/gyro/y",  _scalar(msg.angular_velocity.y))
        rr.log("imu/gyro/z",  _scalar(msg.angular_velocity.z))
        rr.log("imu/accel/z", _scalar(msg.linear_acceleration.z))

    def attitude_cb(self, msg: QuaternionStamped):
        q = msg.quaternion
        roll, pitch, yaw = quat_to_euler(q.w, q.x, q.y, q.z)
        rr.log("attitude/mekf/roll",  _scalar(roll))
        rr.log("attitude/mekf/pitch", _scalar(pitch))
        rr.log("attitude/mekf/yaw",   _scalar(yaw))

    def motors_cb(self, msg: Float64MultiArray):
        if len(msg.data) < 4:
            return
        self._last_motors = [float(x) for x in msg.data[:4]]
        for i, speed in enumerate(self._last_motors, start=1):
            rr.log(f"motors/m{i}_rad_s", _scalar(speed))

    def setpoint_cb(self, msg: Point):
        self._manual_target = np.array([msg.x, msg.y, msg.z], dtype=np.float64)


def main():
    rclpy.init()
    node = RerunVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
