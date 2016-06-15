#ifndef DOWNSAMPLER_DOWNSAMPLER_H_
#define DOWNSAMPLER_DOWNSAMPLER_H_

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <downsampler/DownsamplerConfig.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ecl/geometry.hpp>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

namespace downsampler
{

class Downsampler : public nodelet::Nodelet
{
public:
  virtual void onInit();
  virtual void downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

protected:
  enum PlaneType
  {
    Ground, Ramp, NotATraversablePlane
  };

  virtual void reconfigureCB(DownsamplerConfig &config, uint32_t level);
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr doStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, Eigen::Vector3f& axis,
                                                      std::vector<pcl::PointXYZ>& ground_buffer_points);
  virtual pcl::PointIndices::Ptr extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, Eigen::Vector3f& axis,
                                             boost::shared_ptr<pcl::ModelCoefficients> coeff_out);

  virtual Eigen::Vector3f getRotatedNormal(boost::shared_ptr<pcl::ModelCoefficients> coeff, double degree);
  virtual std::vector<pcl::PointXYZ> filterIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                   boost::shared_ptr<pcl::ModelCoefficients> plane_coeff,
                                                   pcl::PointIndices::Ptr indices);
  virtual PlaneType checkPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               std::vector<pcl::PointXYZ>& ground_buffer_points, pcl::PointIndices::Ptr plane_inliers,
                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff);
  virtual nav_msgs::OdometryPtr coeffToOdom(boost::shared_ptr<pcl::ModelCoefficients> coeff, std::string name);
  virtual bool normalsOfPointsSupportThePlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                              boost::shared_ptr<pcl::ModelCoefficients> plane_coeff);
  virtual bool approximateNormal(pcl::Normal normal_out);

  ros::Subscriber sub_cloud_;
  ros::Publisher pub_downsampled_;
  ros::Publisher pub_filtered_;
  ros::Publisher pub_ground_;
  ros::Publisher pub_padded_ground_;
  ros::Publisher pub_ramp_;
  ros::Publisher pub_result_;
  ros::Publisher pub_ground_pose_;
  ros::Publisher pub_pose_;
  ros::Publisher pub_ramp_pose_;

  std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> > reconfigure_server_;

  double min_range_;
  double max_range_;
  double leaf_size_;
  double filter_radius_;
  int min_points_threshold_;

  int plane_fitting_type_;
  int plane_max_search_count_;

  int normal_neighbours_;
  double max_angle_error_;

  double plane_max_deviation_;
  double plane_max_angle_;
  bool lookedup_;

  tf::Vector3 robot_axis_in_camera_frame_;
  tf::Vector3 robot_center_in_camera_frame_;

  ros::Duration interval_;
  ros::Time next_call_time_;

  tf::TransformBroadcaster transform_broadcaster_;
};

} //end namespace

#endif /* DOWNSAMPLER_DOWNSAMPLER_H_ */
