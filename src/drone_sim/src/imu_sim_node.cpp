#include <random>
#include <chrono>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"

// ADIS16470 datasheet noise specs
// Gyro:  Angular Random Walk = 0.16 °/√hr  → 4.65e-4 rad/s/√Hz
// Accel: Velocity Random Walk = 0.037 m/s/√hr → 1.03e-3 m/s²/√Hz
static constexpr double GYRO_ARW    = 4.65e-4;   // rad/s/√Hz
static constexpr double ACCEL_VRW   = 1.03e-3;   // m/s²/√Hz
static constexpr double GYRO_BIAS_WALK  = 1.0e-6; // rad/s per step (slow drift)
static constexpr double ACCEL_BIAS_WALK = 5.0e-5; // m/s² per step

class ImuSimNode : public rclcpp::Node {
public:
    ImuSimNode()
    : Node("imu_sim"),
      rng_(std::random_device{}()),
      // Scale noise density to 200 Hz sample rate: σ = density * √(sample_rate)
      gyro_noise_ (0.0, GYRO_ARW   * std::sqrt(200.0)),
      accel_noise_(0.0, ACCEL_VRW  * std::sqrt(200.0)),
      gyro_bwalk_ (0.0, GYRO_BIAS_WALK),
      accel_bwalk_(0.0, ACCEL_BIAS_WALK)
    {
        pub_ = create_publisher<sensor_msgs::msg::Imu>("/imu/data_raw", 10);
        timer_ = create_wall_timer(
            std::chrono::milliseconds(5),   // 200 Hz
            [this]() { publish(); });

        RCLCPP_INFO(get_logger(),
            "IMU simulator running at 200 Hz, ADIS16470 noise model");
    }

private:
    void publish() {
        // Bias drifts slowly each step (random walk)
        for (int i = 0; i < 3; ++i) {
            gyro_bias_[i]  += gyro_bwalk_(rng_);
            accel_bias_[i] += accel_bwalk_(rng_);
        }

        auto msg = sensor_msgs::msg::Imu();
        msg.header.stamp    = now();
        msg.header.frame_id = "imu_link";

        // Stationary robot: true angular velocity = 0
        msg.angular_velocity.x = gyro_bias_[0] + gyro_noise_(rng_);
        msg.angular_velocity.y = gyro_bias_[1] + gyro_noise_(rng_);
        msg.angular_velocity.z = gyro_bias_[2] + gyro_noise_(rng_);

        // Stationary robot: accelerometer measures reaction against gravity.
        // Body frame Z points up  →  az = +9.81 m/s²
        msg.linear_acceleration.x = accel_bias_[0] + accel_noise_(rng_);
        msg.linear_acceleration.y = accel_bias_[1] + accel_noise_(rng_);
        msg.linear_acceleration.z = 9.81 + accel_bias_[2] + accel_noise_(rng_);

        // Orientation unknown at this stage — convention: covariance[0] = -1
        msg.orientation_covariance[0] = -1.0;

        pub_->publish(msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::mt19937 rng_;
    std::normal_distribution<double> gyro_noise_, accel_noise_;
    std::normal_distribution<double> gyro_bwalk_, accel_bwalk_;
    double gyro_bias_[3]  = {0.0, 0.0, 0.0};
    double accel_bias_[3] = {0.0, 0.0, 0.0};
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuSimNode>());
    rclcpp::shutdown();
    return 0;
}
