#include <deque>
#include <fstream>

#ifdef USE_ROS1
#include <sensor_msgs/Imu.h>
#else
#include <sensor_msgs/msg/imu.hpp>
#endif

#include "common_lib.h"
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

static inline const bool time_list(PointType& x, PointType& y) {
  return (x.curvature < y.curvature);
};

/// *************IMU Process and undistortion
class ImuProcess {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
#ifdef USE_ROS1
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr& lastimu);
#else
  void Reset(double start_timestamp,
             const sensor_msgs::msg::Imu::ConstSharedPtr& lastimu);
#endif
  void set_extrinsic(const V3D& transl, const M3D& rot);
  void set_extrinsic(const V3D& transl);
  void set_extrinsic(const MD(4, 4) & T);
  void set_gyr_cov(const V3D& scaler);
  void set_acc_cov(const V3D& scaler);
  void set_gyr_bias_cov(const V3D& b_g);
  void set_acc_bias_cov(const V3D& b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup& meas,
               esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
               PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;
  int lidar_type;

 private:
  void IMU_init(const MeasureGroup& meas,
                esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state, int& N);
  void UndistortPcl(const MeasureGroup& meas,
                    esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
                    PointCloudXYZI& pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
#ifdef USE_ROS1
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
#else
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
#endif
  // Preintegrated IMU poses represented by local C++ struct Pose6D (not ROS
  // msg).
  vector<Pose6D> IMUpose;
  vector<M3D> v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int init_iter_num = 1;
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;
};
