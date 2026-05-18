#pragma once
#include <cmath>
#include <array>

// DJI F450-class quadrotor physical constants
inline constexpr double QD_MASS    = 0.5;
inline constexpr double QD_GRAVITY = 9.81;
inline constexpr double QD_L       = 0.17;
inline constexpr double QD_KT      = 3.13e-5;
inline constexpr double QD_KQ      = 7.5e-7;
inline constexpr double QD_IXX     = 2.3e-3;
inline constexpr double QD_IYY     = 2.3e-3;
inline constexpr double QD_IZZ     = 4.5e-3;

// Hover motor speed: sqrt(m*g / (4*KT))
inline constexpr double QD_W_HOVER = 197.92;   // rad/s  (pre-computed)

struct QuadState {
    double px, py, pz;      // position, world frame (m)
    double vx, vy, vz;      // velocity, world frame (m/s)
    double qw, qx, qy, qz;  // attitude quaternion body->world
    double wx, wy, wz;       // angular rate, body frame (rad/s)
};

inline void qd_quat_rot(double qw, double qx, double qy, double qz,
                         double vx, double vy, double vz,
                         double &ox, double &oy, double &oz)
{
    ox = (1-2*(qy*qy+qz*qz))*vx +   2*(qx*qy-qw*qz)*vy +   2*(qx*qz+qw*qy)*vz;
    oy =   2*(qx*qy+qw*qz)*vx + (1-2*(qx*qx+qz*qz))*vy +   2*(qy*qz-qw*qx)*vz;
    oz =   2*(qx*qz-qw*qy)*vx +   2*(qy*qz+qw*qx)*vy + (1-2*(qx*qx+qy*qy))*vz;
}

inline void qd_normalize(QuadState &s)
{
    double n = std::sqrt(s.qw*s.qw + s.qx*s.qx + s.qy*s.qy + s.qz*s.qz);
    if (n < 1e-10) { s.qw = 1; s.qx = s.qy = s.qz = 0; return; }
    s.qw /= n; s.qx /= n; s.qy /= n; s.qz /= n;
}

inline QuadState qd_derivative(const QuadState &s, const std::array<double,4> &omega)
{
    double w2[4] = {omega[0]*omega[0], omega[1]*omega[1],
                    omega[2]*omega[2], omega[3]*omega[3]};

    double thrust = QD_KT * (w2[0] + w2[1] + w2[2] + w2[3]);
    double tau_x  = QD_KT * QD_L * ( w2[1] - w2[3]);
    double tau_y  = QD_KT * QD_L * ( w2[2] - w2[0]);
    double tau_z  = QD_KQ        * (-w2[0] + w2[1] - w2[2] + w2[3]);

    double fx, fy, fz;
    qd_quat_rot(s.qw, s.qx, s.qy, s.qz, 0.0, 0.0, thrust, fx, fy, fz);

    QuadState d;
    d.px = s.vx; d.py = s.vy; d.pz = s.vz;
    d.vx = fx / QD_MASS;
    d.vy = fy / QD_MASS;
    d.vz = fz / QD_MASS - QD_GRAVITY;

    d.qw = 0.5*(-s.qx*s.wx - s.qy*s.wy - s.qz*s.wz);
    d.qx = 0.5*( s.qw*s.wx + s.qy*s.wz - s.qz*s.wy);
    d.qy = 0.5*( s.qw*s.wy - s.qx*s.wz + s.qz*s.wx);
    d.qz = 0.5*( s.qw*s.wz + s.qx*s.wy - s.qy*s.wx);

    double Iwx = QD_IXX*s.wx, Iwy = QD_IYY*s.wy, Iwz = QD_IZZ*s.wz;
    d.wx = (tau_x - (s.wy*Iwz - s.wz*Iwy)) / QD_IXX;
    d.wy = (tau_y - (s.wz*Iwx - s.wx*Iwz)) / QD_IYY;
    d.wz = (tau_z - (s.wx*Iwy - s.wy*Iwx)) / QD_IZZ;

    return d;
}

inline QuadState qd_rk4(const QuadState &s, const std::array<double,4> &omega, double dt)
{
    auto axpy = [](const QuadState &a, const QuadState &b, double h) -> QuadState {
        QuadState r;
        r.px = a.px+h*b.px; r.py = a.py+h*b.py; r.pz = a.pz+h*b.pz;
        r.vx = a.vx+h*b.vx; r.vy = a.vy+h*b.vy; r.vz = a.vz+h*b.vz;
        r.qw = a.qw+h*b.qw; r.qx = a.qx+h*b.qx;
        r.qy = a.qy+h*b.qy; r.qz = a.qz+h*b.qz;
        r.wx = a.wx+h*b.wx; r.wy = a.wy+h*b.wy; r.wz = a.wz+h*b.wz;
        return r;
    };

    QuadState k1 = qd_derivative(s,              omega);
    QuadState k2 = qd_derivative(axpy(s,k1,dt/2),omega);
    QuadState k3 = qd_derivative(axpy(s,k2,dt/2),omega);
    QuadState k4 = qd_derivative(axpy(s,k3,dt),  omega);

    QuadState next;
    #define F(f) next.f = s.f + (dt/6)*(k1.f + 2*k2.f + 2*k3.f + k4.f)
    F(px);F(py);F(pz); F(vx);F(vy);F(vz);
    F(qw);F(qx);F(qy);F(qz); F(wx);F(wy);F(wz);
    #undef F

    qd_normalize(next);
    return next;
}
