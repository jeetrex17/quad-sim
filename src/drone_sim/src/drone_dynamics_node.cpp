#include <random>
#include <chrono>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "drone_sim/quad_dynamics.hpp"

// ADIS16470 IMU noise
static constexpr double GYRO_ARW    = 4.65e-4;
static constexpr double ACCEL_VRW   = 1.03e-3;
static constexpr double GYRO_BWALK  = 1.0e-6;
static constexpr double ACCEL_BWALK = 5.0e-5;
static constexpr double DT          = 0.005;

class DroneDynamicsNode : public rclcpp::Node {
public:
    DroneDynamicsNode()
    : Node("drone_dynamics"),
      rng_(std::random_device{}()),
      gyro_noise_ (0.0, GYRO_ARW   * std::sqrt(1.0/DT)),
      accel_noise_(0.0, ACCEL_VRW  * std::sqrt(1.0/DT)),
      gyro_bwalk_ (0.0, GYRO_BWALK),
      accel_bwalk_(0.0, ACCEL_BWALK)
    {
        state_ = {0,0,1, 0,0,0, 1,0,0,0, 0,0,0};
        motors_.fill(QD_W_HOVER);

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/drone/odom", 10);
        imu_pub_  = create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", 10);

        motor_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "/drone/motor_speeds", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() >= 4)
                    for (int i = 0; i < 4; ++i) motors_[i] = msg->data[i];
            });

        timer_ = create_wall_timer(
            std::chrono::milliseconds(5),
            [this]() { step(); });

        RCLCPP_INFO(get_logger(),
            "Drone dynamics node running, hover motor speed = %.1f rad/s", QD_W_HOVER);
    }

private:
    void step()
    {
        state_ = qd_rk4(state_, motors_, DT);

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
        double ax, ay, az;
        qd_quat_rot(state_.qw, -state_.qx, -state_.qy, -state_.qz,
                    0.0, 0.0, QD_GRAVITY, ax, ay, az);

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

    QuadState           state_;
    std::array<double,4> motors_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DroneDynamicsNode>());
    rclcpp::shutdown();
    return 0;
}
