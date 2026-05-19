/**
* This file is part of FAST-LIO2
*
* Copyright 2026 Xinle XI, MaRS Lab, University of Hong Kong, <mars.hku.hk>
* Developed by Xinle XI <xinlexi at connect dot hku dot hk>
* 
* This file is a refactor of FASTLIO2
*/
#include <condition_variable>
#include <mutex>
#include <omp.h>
#include <csignal>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>

#include <pcl/filters/voxel_grid.h>

#include "ikd-Tree/ikd_Tree.h"
#include "common_lib.h"
#include "use-ikfom.hpp"
#include "preprocess.h"
#include "IMU_Processing.hpp"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define PUBFRAME_PERIOD     (20)

void pointBodyToWorld(PointType const * const pi, PointType * const po);
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
void prop_imu_once(state_ikfom &prop_state, const double dt, const V3D &acc_raw, const V3D &gyro_raw);
bool try_init_gravity_aligned_transform(const state_ikfom &s);
void lasermap_fov_segment();
bool sync_packages(MeasureGroup &meas);
void map_incremental();
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

ros::Publisher pubImuPropOdom;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
nav_msgs::Odometry imuPropOdom;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

extern state_ikfom imu_prop_state;
extern state_ikfom latest_ekf_state;
extern Matrix4d G_T_I0;
extern bool   lidar_pushed, flg_EKF_inited;
extern int effct_feat_num;
extern deque<PointCloudXYZI::Ptr>        lidar_buffer;
extern deque<double>                     time_buffer;
extern deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
extern PointCloudXYZI::Ptr feats_undistort;
extern PointCloudXYZI::Ptr feats_down_body;
extern PointCloudXYZI::Ptr feats_down_world;
extern PointCloudXYZI::Ptr laserCloudOri;
extern PointCloudXYZI::Ptr featsFromMap;
extern PointCloudXYZI::Ptr _featsArray;
extern PointCloudXYZI::Ptr normvec;
extern pcl::VoxelGrid<PointType> downSizeFilterSurf;
extern pcl::VoxelGrid<PointType> downSizeFilterMap;
extern double lidar_end_time;
extern double first_lidar_time;
extern state_ikfom state_point;
extern float DET_RANGE;
extern double cube_len;
extern int lidar_type;
extern bool   point_selected_surf[100000];
extern KD_TREE<PointType> ikdtree;
extern double match_time;
extern double solve_time;
extern MeasureGroup Measures;
extern vect3 pos_lid;
extern double kdtree_incremental_time;
extern double kdtree_search_time;
extern double kdtree_delete_time;
extern vector<PointVector>  Nearest_Points;
extern int add_point_size;
extern int kdtree_delete_counter;
extern float res_last[100000];

mutex mtx_buffer;
mutex mtx_imu_prop;
condition_variable sig_buffer;

bool flg_first_scan = true;
bool flg_exit = false;

int time_log_counter = 0, scan_count = 0, publish_count = 0;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

#define MAXN                (720000)
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool   imu_unit_g = false;

double time_diff_lidar_to_imu = 0.0;

// Optional high-rate odometry propagation using IMU updates.
// Motivation: LiDAR odometry output is tied to LiDAR frame rate (e.g., ~10 Hz on MID360),
// while UAV control often needs higher-rate odometry.
// Reference: FAST_LIO issue #394.
bool imu_prop_enable = false;
bool imu_prop_use_imu_attitude = true;
string imu_prop_topic = "/imu_propagate";
state_ikfom imu_prop_state;
state_ikfom latest_ekf_state;
double latest_ekf_time = 0.0;
double last_prop_t_from_ekf = 0.0;
bool state_update_flg = false;
bool imu_prop_state_valid = false;
deque<sensor_msgs::Imu> prop_imu_buffer;

int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;

esekfom::esekf<state_ikfom, 12, input_ikfom> kf;

string map_file_path, lid_topic, imu_topic;

double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;

V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
vector<vector<int>>  pointSearchInd_surf;
int kdtree_size_st = 0, kdtree_size_end = 0;

bool gravity_align_initialized = false;

inline void ConvertPoseToGravFrame(V3D &pos, Quaterniond &q)
{
    M3D rot = q.toRotationMatrix();
    pos = G_T_I0.block<3, 3>(0, 0) * pos + G_T_I0.col(3).head(3);
    rot = G_T_I0.block<3, 3>(0, 0) * rot;
    q = Quaterniond(rot);
}

inline void ConvertTwistToGravFrame(V3D &vel)
{
    vel = G_T_I0.block<3, 3>(0, 0) * vel;
}

void publish_imu_propagate_odometry(const sensor_msgs::Imu &imu_msg)
{
    imuPropOdom.header.frame_id = "camera_init";
    imuPropOdom.child_frame_id = "body";
    imuPropOdom.header.stamp = imu_msg.header.stamp;
    V3D pos = imu_prop_state.pos;
    Quaterniond q(imu_prop_state.rot);
    V3D vel = imu_prop_state.vel;
    ConvertPoseToGravFrame(pos, q);
    ConvertTwistToGravFrame(vel);
    imuPropOdom.pose.pose.position.x = pos(0);
    imuPropOdom.pose.pose.position.y = pos(1);
    imuPropOdom.pose.pose.position.z = pos(2);
    imuPropOdom.pose.pose.orientation.x = q.x();
    imuPropOdom.pose.pose.orientation.y = q.y();
    imuPropOdom.pose.pose.orientation.z = q.z();
    imuPropOdom.pose.pose.orientation.w = q.w();
    imuPropOdom.twist.twist.linear.x = vel(0);
    imuPropOdom.twist.twist.linear.y = vel(1);
    imuPropOdom.twist.twist.linear.z = vel(2);
    pubImuPropOdom.publish(imuPropOdom);
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    static bool   timediff_set_flg = false;
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

inline bool is_valid_prop_dt(const double dt)
{
    return std::isfinite(dt) && dt > 0.0 && dt < 0.1;
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    if (imu_unit_g)
    {
        msg->linear_acceleration.x *= G_m_s2;
        msg->linear_acceleration.y *= G_m_s2;
        msg->linear_acceleration.z *= G_m_s2;
    }

    imu_buffer.push_back(msg);

    // Optional high-rate odometry by propagating the latest EKF state at IMU update rate.
    if (imu_prop_enable)
    {
        lock_guard<mutex> lk_imu_prop(mtx_imu_prop);
        if (imu_prop_use_imu_attitude)
        {
            Eigen::Quaterniond imu_q(msg->orientation.w,
                                     msg->orientation.x,
                                     msg->orientation.y,
                                     msg->orientation.z);
            if (imu_q.norm() > 1e-12)
            {
                imu_q.normalize();
                imu_prop_state.rot = SO3(imu_q.toRotationMatrix());
            }
        }
        prop_imu_buffer.push_back(*msg);
        if (imu_prop_state_valid)
        {
            if (state_update_flg)
            {
                imu_prop_state = latest_ekf_state;
                if (imu_prop_use_imu_attitude)
                {
                    Eigen::Quaterniond imu_q(msg->orientation.w,
                                             msg->orientation.x,
                                             msg->orientation.y,
                                             msg->orientation.z);
                    if (imu_q.norm() > 1e-12)
                    {
                        imu_q.normalize();
                        imu_prop_state.rot = SO3(imu_q.toRotationMatrix());
                    }
                }
                while ((!prop_imu_buffer.empty()) &&
                       (prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
                {
                    prop_imu_buffer.pop_front();
                }
                last_prop_t_from_ekf = 0.0;
                for (size_t i = 0; i < prop_imu_buffer.size(); ++i)
                {
                    const double t_from_ekf = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
                    const double dt = t_from_ekf - last_prop_t_from_ekf;
                    if (!is_valid_prop_dt(dt))
                    {
                        continue;
                    }
                    V3D acc(prop_imu_buffer[i].linear_acceleration.x,
                            prop_imu_buffer[i].linear_acceleration.y,
                            prop_imu_buffer[i].linear_acceleration.z);
                    V3D gyro(prop_imu_buffer[i].angular_velocity.x,
                             prop_imu_buffer[i].angular_velocity.y,
                             prop_imu_buffer[i].angular_velocity.z);
                    prop_imu_once(imu_prop_state, dt, acc, gyro);
                    last_prop_t_from_ekf = t_from_ekf;
                }
                state_update_flg = false;
            }
            else
            {
                const double t_from_ekf = msg->header.stamp.toSec() - latest_ekf_time;
                const double dt = t_from_ekf - last_prop_t_from_ekf;
                if (is_valid_prop_dt(dt))
                {
                    V3D acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                    V3D gyro(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
                    prop_imu_once(imu_prop_state, dt, acc, gyro);
                    last_prop_t_from_ekf = t_from_ekf;
                }
            }
            publish_imu_propagate_odometry(*msg);
        }
    }

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
            const V3D p_grav = G_T_I0.block<3, 3>(0, 0) * V3D(laserCloudWorld->points[i].x,
                                                              laserCloudWorld->points[i].y,
                                                              laserCloudWorld->points[i].z)
                               + G_T_I0.col(3).head(3);
            laserCloudWorld->points[i].x = p_grav(0);
            laserCloudWorld->points[i].y = p_grav(1);
            laserCloudWorld->points[i].z = p_grav(2);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
            const V3D p_grav = G_T_I0.block<3, 3>(0, 0) * V3D(laserCloudWorld->points[i].x,
                                                              laserCloudWorld->points[i].y,
                                                              laserCloudWorld->points[i].z)
                               + G_T_I0.col(3).head(3);
            laserCloudWorld->points[i].x = p_grav(0);
            laserCloudWorld->points[i].y = p_grav(1);
            laserCloudWorld->points[i].z = p_grav(2);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
        const V3D p_grav = G_T_I0.block<3, 3>(0, 0) * V3D(laserCloudWorld->points[i].x,
                                                          laserCloudWorld->points[i].y,
                                                          laserCloudWorld->points[i].z)
                           + G_T_I0.col(3).head(3);
        laserCloudWorld->points[i].x = p_grav(0);
        laserCloudWorld->points[i].y = p_grav(1);
        laserCloudWorld->points[i].z = p_grav(2);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    PointCloudXYZI::Ptr map_grav(new PointCloudXYZI(featsFromMap->size(), 1));
    for (size_t i = 0; i < featsFromMap->size(); ++i)
    {
        const auto &p = featsFromMap->points[i];
        const V3D p_grav = G_T_I0.block<3, 3>(0, 0) * V3D(p.x, p.y, p.z) + G_T_I0.col(3).head(3);
        auto &out = map_grav->points[i];
        out = p;
        out.x = p_grav(0);
        out.y = p_grav(1);
        out.z = p_grav(2);
    }
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*map_grav, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}



template<typename T>
void set_posestamp(T & out)
{
    Quaterniond q(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);
    V3D pos = state_point.pos;
    ConvertPoseToGravFrame(pos, q);
    out.pose.position.x = pos(0);
    out.pose.position.y = pos(1);
    out.pose.position.z = pos(2);
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    V3D vel = state_point.vel;
    ConvertTwistToGravFrame(vel);
    odomAftMapped.twist.twist.linear.x = vel(0);
    odomAftMapped.twist.twist.linear.y = vel(1);
    odomAftMapped.twist.twist.linear.z = vel(2);
    pubOdomAftMapped.publish(odomAftMapped);
    pubImuPropOdom.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

int main(int argc, char** argv)
{
    std::string root_dir = ROOT_DIR;
    std::vector<double>       extrinT(3, 0.0);
    std::vector<double>       extrinR(9, 0.0);
    double HALF_FOV_COS = 0, FOV_DEG = 0;

    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<bool>("preprocess/imu_unit_g", imu_unit_g, false);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<bool>("imu_propagate/enable", imu_prop_enable, false);
    nh.param<bool>("imu_propagate/use_imu_attitude", imu_prop_use_imu_attitude, true);
    nh.param<string>("imu_propagate/topic", imu_prop_topic, "/imu_propagate");

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    pubImuPropOdom = nh.advertise<nav_msgs::Odometry>(imu_prop_topic, 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            if (imu_prop_enable)
            {
                lock_guard<mutex> lk_imu_prop(mtx_imu_prop);
                // Hand off the latest fused state as propagation anchor for subsequent IMU callbacks.
                latest_ekf_state = state_point;
                latest_ekf_time = lidar_end_time;
                state_update_flg = true;
                imu_prop_state_valid = true;
            }
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            if (flg_EKF_inited && !gravity_align_initialized)
            {
                gravity_align_initialized = try_init_gravity_aligned_transform(state_point);
            }
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
