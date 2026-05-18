#include <array>
#include <cmath>
#include <random>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"

// DJI F450-class quadrotor physical constants
static constexpr double MASS    = 0.5;      // kg
static constexpr double GRAVITY = 9.81;     // m/s^2
static constexpr double L       = 0.17;     // m, motor-to-center arm length
static constexpr double KT      = 3.13e-5;  // N/(rad/s)^2, thrust coefficient
static constexpr double KQ      = 7.5e-7;   // N*m/(rad/s)^2, drag torque coefficient
static constexpr double IXX     = 2.3e-3;   // kg*m^2
static constexpr double IYY     = 2.3e-3;
static constexpr double IZZ     = 4.5e-3;

// X-config motor layout (top view):  2(CCW)  1(CW)
//                                        \  /
//                                        /  \
//                                    3(CW)  4(CCW)

// ADIS16470 IMU noise
static constexpr double GYRO_ARW       = 4.65e-4;  // rad/s/sqrt(Hz)
static constexpr double ACCEL_VRW      = 1.03e-3;  // m/s^2/sqrt(Hz)
static constexpr double GYRO_BWALK     = 1.0e-6;
static constexpr double ACCEL_BWALK    = 5.0e-5;
static constexpr double DT             = 0.005;     // 200 Hz

struct State {
    double px, py, pz;      // position, world frame
    double vx, vy, vz;      // velocity, world frame
    double qw, qx, qy, qz;  // attitude quaternion, body->world
    double wx, wy, wz;       // angular rate, body frame
};

// Rotate vector (vx,vy,vz) by quaternion (qw,qx,qy,qz)
static void quat_rot(double qw, double qx, double qy, double qz,
                     double vx, double vy, double vz,
                     double &ox, double &oy, double &oz)
{
    ox = (1-2*(qy*qy+qz*qz))*vx +   2*(qx*qy-qw*qz)*vy +   2*(qx*qz+qw*qy)*vz;
    oy =   2*(qx*qy+qw*qz)*vx + (1-2*(qx*qx+qz*qz))*vy +   2*(qy*qz-qw*qx)*vz;
    oz =   2*(qx*qz-qw*qy)*vx +   2*(qy*qz+qw*qx)*vy + (1-2*(qx*qx+qy*qy))*vz;
}

static void quat_normalize(State &s)
{
    double n = std::sqrt(s.qw*s.qw + s.qx*s.qx + s.qy*s.qy + s.qz*s.qz);
    if (n < 1e-10) { s.qw = 1; s.qx = s.qy = s.qz = 0; return; }
    s.qw /= n; s.qx /= n; s.qy /= n; s.qz /= n;
}

// Compute dx/dt given state x and motor speeds omega[4]
static State derivative(const State &s, const double omega[4])
{
    double w2[4] = {omega[0]*omega[0], omega[1]*omega[1],
                    omega[2]*omega[2], omega[3]*omega[3]};

    double thrust = KT * (w2[0] + w2[1] + w2[2] + w2[3]);

    // Body-frame torques from differential thrust and rotor drag
    double tau_x = KT * L * ( w2[1] - w2[3]);         // roll
    double tau_y = KT * L * ( w2[2] - w2[0]);         // pitch
    double tau_z = KQ      * (-w2[0] + w2[1] - w2[2] + w2[3]); // yaw

    // Thrust (body +Z) rotated into world frame
    double fx, fy, fz;
    quat_rot(s.qw, s.qx, s.qy, s.qz, 0.0, 0.0, thrust, fx, fy, fz);

    State d;

    // Translational kinematics / dynamics
    d.px = s.vx;
    d.py = s.vy;
    d.pz = s.vz;
    d.vx = fx / MASS;
    d.vy = fy / MASS;
    d.vz = fz / MASS - GRAVITY;   // gravity pulls down in world frame

    // Quaternion kinematics: dq/dt = 0.5 * q * [0, w_body]
    d.qw = 0.5*(-s.qx*s.wx - s.qy*s.wy - s.qz*s.wz);
    d.qx = 0.5*( s.qw*s.wx + s.qy*s.wz - s.qz*s.wy);
    d.qy = 0.5*( s.qw*s.wy - s.qx*s.wz + s.qz*s.wx);
    d.qz = 0.5*( s.qw*s.wz + s.qx*s.wy - s.qy*s.wx);

    // Euler's rotation equation: I*dw/dt = tau - w x (I*w)
    double Iwx = IXX*s.wx, Iwy = IYY*s.wy, Iwz = IZZ*s.wz;
    d.wx = (tau_x - (s.wy*Iwz - s.wz*Iwy)) / IXX;
    d.wy = (tau_y - (s.wz*Iwx - s.wx*Iwz)) / IYY;
    d.wz = (tau_z - (s.wx*Iwy - s.wy*Iwx)) / IZZ;

    return d;
}

// 4th-order Runge-Kutta integrator
static State rk4(const State &s, const double omega[4], double dt)
{
    auto axpy = [](const State &a, const State &b, double h) -> State {
        State r;
        r.px = a.px+h*b.px; r.py = a.py+h*b.py; r.pz = a.pz+h*b.pz;
        r.vx = a.vx+h*b.vx; r.vy = a.vy+h*b.vy; r.vz = a.vz+h*b.vz;
        r.qw = a.qw+h*b.qw; r.qx = a.qx+h*b.qx;
        r.qy = a.qy+h*b.qy; r.qz = a.qz+h*b.qz;
        r.wx = a.wx+h*b.wx; r.wy = a.wy+h*b.wy; r.wz = a.wz+h*b.wz;
        return r;
    };

    State k1 = derivative(s,           omega);
    State k2 = derivative(axpy(s,k1,dt/2), omega);
    State k3 = derivative(axpy(s,k2,dt/2), omega);
    State k4 = derivative(axpy(s,k3,dt),   omega);

    State next;
    #define RK4_FIELD(f) next.f = s.f + (dt/6)*(k1.f + 2*k2.f + 2*k3.f + k4.f)
    RK4_FIELD(px); RK4_FIELD(py); RK4_FIELD(pz);
    RK4_FIELD(vx); RK4_FIELD(vy); RK4_FIELD(vz);
    RK4_FIELD(qw); RK4_FIELD(qx); RK4_FIELD(qy); RK4_FIELD(qz);
    RK4_FIELD(wx); RK4_FIELD(wy); RK4_FIELD(wz);
    #undef RK4_FIELD

    quat_normalize(next);
    return next;
}

class DroneDynamicsNode : public rclcpp::Node {
public:
    DroneDynamicsNode()
    : Node("drone_dynamics"),
      rng_(std::random_device{}()),
      gyro_noise_ (0.0, GYRO_ARW    * std::sqrt(1.0/DT)),
      accel_noise_(0.0, ACCEL_VRW   * std::sqrt(1.0/DT)),
      gyro_bwalk_ (0.0, GYRO_BWALK),
      accel_bwalk_(0.0, ACCEL_BWALK)
    {
        // Start at z=1m, level, stationary
        state_ = {0,0,1, 0,0,0, 1,0,0,0, 0,0,0};

        // Hover motor speed: thrust = weight  =>  4*KT*w^2 = m*g
        double w_hover = std::sqrt(MASS * GRAVITY / (4.0 * KT));
        for (int i = 0; i < 4; ++i) motor_speeds_[i] = w_hover;

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/odom", 10);
        imu_pub_  = create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", 10);

        motor_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/drone/motor_speeds", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() >= 4)
                    for (int i = 0; i < 4; ++i) motor_speeds_[i] = msg->data[i];
            });

        timer_ = create_wall_timer(
            std::chrono::milliseconds(5),
            [this]() { step(); });

        RCLCPP_INFO(get_logger(),
            "Drone dynamics node running, hover motor speed = %.1f rad/s", w_hover);
    }

private:
    void step()
    {
        state_ = rk4(state_, motor_speeds_, DT);

        for (int i = 0; i < 3; ++i) {
            gyro_bias_[i]  += gyro_bwalk_(rng_);
            accel_bias_[i] += accel_bwalk_(rng_);
        }

        publish_odom();
        publish_imu();
    }

    void publish_odom()
    {
        auto msg = nav_msgs::msg::Odometry();
        msg.header.stamp    = now();
        msg.header.frame_id = "world";
        msg.child_frame_id  = "base_link";

        msg.pose.pose.position.x    = state_.px;
        msg.pose.pose.position.y    = state_.py;
        msg.pose.pose.position.z    = state_.pz;
        msg.pose.pose.orientation.w = state_.qw;
        msg.pose.pose.orientation.x = state_.qx;
        msg.pose.pose.orientation.y = state_.qy;
        msg.pose.pose.orientation.z = state_.qz;
        msg.twist.twist.linear.x    = state_.vx;
        msg.twist.twist.linear.y    = state_.vy;
        msg.twist.twist.linear.z    = state_.vz;
        msg.twist.twist.angular.x   = state_.wx;
        msg.twist.twist.angular.y   = state_.wy;
        msg.twist.twist.angular.z   = state_.wz;

        odom_pub_->publish(msg);
    }

    void publish_imu()
    {
        // Accelerometer measures specific force in body frame: a_body = R^T * (a_world + g_world)
        // At hover: a_world ~ 0, so specific force = R^T * [0, 0, g]
        // R^T is rotation by conjugate quaternion q* = (qw, -qx, -qy, -qz)
        double ax, ay, az;
        quat_rot(state_.qw, -state_.qx, -state_.qy, -state_.qz,
                 0.0, 0.0, GRAVITY, ax, ay, az);

        auto msg = sensor_msgs::msg::Imu();
        msg.header.stamp    = now();
        msg.header.frame_id = "imu_link";

        msg.angular_velocity.x    = state_.wx + gyro_bias_[0]  + gyro_noise_(rng_);
        msg.angular_velocity.y    = state_.wy + gyro_bias_[1]  + gyro_noise_(rng_);
        msg.angular_velocity.z    = state_.wz + gyro_bias_[2]  + gyro_noise_(rng_);
        msg.linear_acceleration.x = ax        + accel_bias_[0] + accel_noise_(rng_);
        msg.linear_acceleration.y = ay        + accel_bias_[1] + accel_noise_(rng_);
        msg.linear_acceleration.z = az        + accel_bias_[2] + accel_noise_(rng_);
        msg.orientation_covariance[0] = -1.0;

        imu_pub_->publish(msg);
    }

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr   imu_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr motor_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::mt19937 rng_;
    std::normal_distribution<double> gyro_noise_, accel_noise_;
    std::normal_distribution<double> gyro_bwalk_, accel_bwalk_;
    double gyro_bias_[3]  = {};
    double accel_bias_[3] = {};

    State  state_;
    double motor_speeds_[4];
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DroneDynamicsNode>());
    rclcpp::shutdown();
    return 0;
}
