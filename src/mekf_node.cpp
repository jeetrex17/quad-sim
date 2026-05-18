/*
 * Multiplicative Extended Kalman Filter (MEKF) for attitude estimation.
 *
 * Problem: gyro integrates well short-term but drifts. Accel gives gravity
 * direction (roll/pitch) but is noisy and corrupted during motion.
 * The filter fuses them optimally.
 *
 * Error state: dx = [dtheta (3), db (3)]
 *   dtheta — small 3D rotation error on top of nominal quaternion q
 *   db     — gyro bias error
 *
 * After each correction: q <- q * q(dtheta), then reset dtheta to zero.
 * This "multiplicative" reset is what keeps the quaternion unit-norm.
 */
#include <cmath>
#include <Eigen/Dense>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/quaternion_stamped.hpp"

using Vec3  = Eigen::Vector3d;
using Vec4  = Eigen::Vector4d;
using Mat3  = Eigen::Matrix3d;
using Mat6  = Eigen::Matrix<double,6,6>;
using Mat36 = Eigen::Matrix<double,3,6>;
using Vec6  = Eigen::Matrix<double,6,1>;

// ── Quaternion math (storage: w, x, y, z) ────────────────────────────────

static Vec4 qmul(const Vec4& p, const Vec4& q)
{
    return {p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3],
            p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2],
            p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1],
            p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]};
}

static Vec4 qnorm(Vec4 q)
{
    double n = q.norm();
    return (n < 1e-10) ? Vec4(1,0,0,0) : q/n;
}

// Body-to-world rotation matrix from quaternion (w,x,y,z)
static Mat3 qR(const Vec4& q)
{
    double w=q[0],x=q[1],y=q[2],z=q[3];
    Mat3 R;
    R << 1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),
           2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),
           2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y);
    return R;
}

// Cross-product (skew-symmetric) matrix: skew(v)*w == v cross w
static Mat3 skew(const Vec3& v)
{
    Mat3 S;
    S <<  0,    -v[2],  v[1],
          v[2],  0,    -v[0],
         -v[1],  v[0],  0;
    return S;
}

// Exact quaternion integration: q * exp(omega * dt)
static Vec4 qintegrate(const Vec4& q, const Vec3& omega, double dt)
{
    double angle = omega.norm() * dt;
    Vec4 dq;
    if (angle < 1e-10) {
        dq = Vec4(1, omega[0]*dt/2, omega[1]*dt/2, omega[2]*dt/2);
    } else {
        Vec3 ax = omega / omega.norm();
        double s = std::sin(angle/2);
        dq = Vec4(std::cos(angle/2), s*ax[0], s*ax[1], s*ax[2]);
    }
    return qnorm(qmul(q, dq));
}

// ── MEKF parameters ──────────────────────────────────────────────────────

static constexpr double GRAVITY   = 9.81;
static constexpr double DT        = 0.005;    // 200 Hz

// ADIS16470 specs (must match dynamics node)
static constexpr double GYRO_ARW  = 4.65e-4;  // rad/s/sqrt(Hz)
static constexpr double ACCEL_VRW = 1.03e-3;  // m/s^2/sqrt(Hz)
static constexpr double BIAS_WALK = 1.0e-6;   // rad/s per step

static const double VAR_GYRO  = GYRO_ARW  * GYRO_ARW  / DT;
static const double VAR_ACCEL = ACCEL_VRW * ACCEL_VRW / DT;
static const double VAR_BIAS  = BIAS_WALK * BIAS_WALK;

class MekfNode : public rclcpp::Node {
public:
    MekfNode() : Node("mekf")
    {
        q_ = Vec4(1, 0, 0, 0);   // start level
        b_.setZero();              // assume zero initial bias

        P_.setZero();
        P_.block<3,3>(0,0) = 0.1  * Mat3::Identity();   // attitude uncertainty
        P_.block<3,3>(3,3) = 1e-4 * Mat3::Identity();   // bias uncertainty

        Q_.setZero();
        Q_.block<3,3>(0,0) = VAR_GYRO * Mat3::Identity();
        Q_.block<3,3>(3,3) = VAR_BIAS * Mat3::Identity();

        R_accel_ = VAR_ACCEL * Mat3::Identity();

        pub_ = create_publisher<geometry_msgs::msg::QuaternionStamped>(
            "/imu/attitude", 10);
        sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data_raw", 10,
            [this](const sensor_msgs::msg::Imu::SharedPtr m){ tick(m); });

        RCLCPP_INFO(get_logger(), "MEKF running");
    }

private:
    void tick(const sensor_msgs::msg::Imu::SharedPtr& msg)
    {
        Vec3 omega{msg->angular_velocity.x,
                   msg->angular_velocity.y,
                   msg->angular_velocity.z};
        Vec3 accel{msg->linear_acceleration.x,
                   msg->linear_acceleration.y,
                   msg->linear_acceleration.z};

        predict(omega);
        correct(accel);

        auto out = geometry_msgs::msg::QuaternionStamped();
        out.header = msg->header;
        out.quaternion.w = q_[0];
        out.quaternion.x = q_[1];
        out.quaternion.y = q_[2];
        out.quaternion.z = q_[3];
        pub_->publish(out);
    }

    // ── Predict: integrate gyro, grow covariance ─────────────────────────
    void predict(const Vec3& omega_raw)
    {
        Vec3 w = omega_raw - b_;    // remove estimated bias

        q_ = qintegrate(q_, w, DT);

        // F = d(error_dynamics)/d(error_state)
        // dtheta_dot = -[w x] * dtheta - db
        // db_dot     = 0
        Mat6 Phi = Mat6::Identity();
        Phi.block<3,3>(0,0) -= skew(w) * DT;
        Phi.block<3,3>(0,3)  = -Mat3::Identity() * DT;

        P_ = Phi * P_ * Phi.transpose() + Q_;
    }

    // ── Correct: gravity vector from accelerometer ────────────────────────
    void correct(const Vec3& accel)
    {
        // During aggressive maneuvers accel ≠ gravity alone — skip update.
        if (std::abs(accel.norm() - GRAVITY) > 2.0) return;

        // Predicted gravity in body frame using current quaternion estimate
        Vec3 g_body = qR(q_).transpose() * Vec3(0, 0, GRAVITY);

        // Measurement Jacobian (3x6).
        // Derivation: accel_pred = g_body + skew(g_body)*dtheta + 0*db
        // So H = [skew(g_body), 0]
        Mat36 H;
        H.setZero();
        H.block<3,3>(0,0) = skew(g_body);

        // Innovation
        Vec3 y = accel - g_body;

        // Kalman gain (3x3 innovation covariance)
        Eigen::Matrix3d S = H * P_ * H.transpose() + R_accel_;
        Eigen::Matrix<double,6,3> K = P_ * H.transpose() * S.inverse();

        // Error state correction
        Vec6 dx = K * y;

        // Apply attitude correction: q <- q * q(dtheta)
        Vec3 dtheta = dx.head<3>();
        Vec4 dq(1.0, dtheta[0]/2, dtheta[1]/2, dtheta[2]/2);
        q_ = qnorm(qmul(q_, dq));

        // Apply bias correction
        b_ += dx.tail<3>();

        // Joseph form covariance update (numerically stable)
        Mat6 IKH = Mat6::Identity() - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * R_accel_ * K.transpose();
    }

    rclcpp::Publisher<geometry_msgs::msg::QuaternionStamped>::SharedPtr pub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_;

    Vec4 q_;       // attitude quaternion (w,x,y,z)
    Vec3 b_;       // gyro bias estimate (body frame)
    Mat6 P_;       // error-state covariance (6x6)
    Mat6 Q_;       // process noise
    Mat3 R_accel_; // accelerometer measurement noise
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MekfNode>());
    rclcpp::shutdown();
    return 0;
}
