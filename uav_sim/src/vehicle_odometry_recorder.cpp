/**
 * @brief PX4 Vehicle Odometry CSV Recorder
 * @file vehicle_odometry_recorder.cpp
 *
 * Subscribes to /fmu/out/vehicle_odometry and records all fields to a CSV file.
 */

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

class VehicleOdometryRecorder : public rclcpp::Node
{
public:
    explicit VehicleOdometryRecorder()
    : Node("vehicle_odometry_recorder"), msg_count_(0)
    {
        // Declare parameters
        this->declare_parameter("output_dir", ".");
        this->declare_parameter("filename_prefix", "vehicle_odometry");

        std::string output_dir = this->get_parameter("output_dir").as_string();
        std::string filename_prefix = this->get_parameter("filename_prefix").as_string();

        // Generate output filename with timestamp
        std::string csv_path = output_dir + "/" + filename_prefix + "_"
                             + current_timestamp_str() + ".csv";

        csv_file_.open(csv_path);
        if (!csv_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_path.c_str());
            rclcpp::shutdown();
            return;
        }

        // Write CSV header
        csv_file_ << "timestamp,timestamp_sample,pose_frame,"
                  << "x,y,z,"
                  << "q_w,q_x,q_y,q_z,"
                  << "velocity_frame,"
                  << "vx,vy,vz,"
                  << "angvel_x,angvel_y,angvel_z,"
                  << "pos_var_x,pos_var_y,pos_var_z,"
                  << "orient_var_x,orient_var_y,orient_var_z,"
                  << "vel_var_x,vel_var_y,vel_var_z,"
                  << "reset_counter,quality\n";
        csv_file_.flush();

        RCLCPP_INFO(this->get_logger(), "Recording vehicle odometry to: %s", csv_path.c_str());

        // QoS profile matching PX4 uORB→ROS2 bridge conventions
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

        // Subscribe to /fmu/out/vehicle_odometry
        subscription_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
            "/fmu/out/vehicle_odometry", qos,
            [this](px4_msgs::msg::VehicleOdometry::UniquePtr msg) {
                odometry_callback(std::move(msg));
            });
    }

    ~VehicleOdometryRecorder() override
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down. Total messages recorded: %zu", msg_count_);
        if (csv_file_.is_open()) {
            csv_file_.flush();
            csv_file_.close();
        }
    }

private:
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr subscription_;
    std::ofstream csv_file_;
    size_t msg_count_;
    static constexpr int LOG_INTERVAL = 100;  // Log every N messages

    /// Generate a timestamp string like "20260720_210000"
    static std::string current_timestamp_str()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now{};
        localtime_r(&time_t_now, &tm_now);

        std::ostringstream oss;
        oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
        return oss.str();
    }

    void odometry_callback(px4_msgs::msg::VehicleOdometry::UniquePtr msg)
    {
        csv_file_ << msg->timestamp << ","
                  << msg->timestamp_sample << ","
                  << static_cast<int>(msg->pose_frame) << ","
                  << msg->position[0] << "," << msg->position[1] << "," << msg->position[2] << ","
                  << msg->q[0] << "," << msg->q[1] << "," << msg->q[2] << "," << msg->q[3] << ","
                  << static_cast<int>(msg->velocity_frame) << ","
                  << msg->velocity[0] << "," << msg->velocity[1] << "," << msg->velocity[2] << ","
                  << msg->angular_velocity[0] << "," << msg->angular_velocity[1] << ","
                  << msg->angular_velocity[2] << ","
                  << msg->position_variance[0] << "," << msg->position_variance[1] << ","
                  << msg->position_variance[2] << ","
                  << msg->orientation_variance[0] << "," << msg->orientation_variance[1] << ","
                  << msg->orientation_variance[2] << ","
                  << msg->velocity_variance[0] << "," << msg->velocity_variance[1] << ","
                  << msg->velocity_variance[2] << ","
                  << static_cast<int>(msg->reset_counter) << ","
                  << static_cast<int>(msg->quality) << "\n";

        msg_count_++;

        if (msg_count_ % LOG_INTERVAL == 0) {
            csv_file_.flush();
            RCLCPP_INFO(this->get_logger(),
                        "Recorded %zu messages (latest timestamp: %lu us)",
                        msg_count_, msg->timestamp);
        }
    }
};

int main(int argc, char *argv[])
{
    std::cout << "Starting Vehicle Odometry CSV Recorder..." << std::endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);

    rclcpp::init(argc, argv);
    auto node = std::make_shared<VehicleOdometryRecorder>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
