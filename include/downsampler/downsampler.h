#ifndef DOWNSAMPLER_DOWNSAMPLER_H_
#define DOWNSAMPLER_DOWNSAMPLER_H_

#include <atomic>
#include <mutex>
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <downsampler/DownsamplerConfig.h>
#include <downsampler/DownsamplerRampConfig.h>
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
  virtual void reconfigureRampCB(DownsamplerRampConfig &config, uint32_t level);
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr doStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                      pcl::PointCloud<pcl::Normal>::Ptr input_normals,
                                                      Eigen::Vector3f& axis,
                                                      std::vector<pcl::PointXYZ>& ground_buffer_points);
  virtual pcl::PointIndices::Ptr extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                             pcl::PointCloud<pcl::Normal>::Ptr input_normals, Eigen::Vector3f& axis,
                                             boost::shared_ptr<pcl::ModelCoefficients> coeff_out);

  virtual std::vector<pcl::PointXYZ> filterIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                   boost::shared_ptr<pcl::ModelCoefficients> plane_coeff,
                                                   pcl::PointIndices::Ptr indices);
  virtual PlaneType checkPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               std::vector<pcl::PointXYZ>& ground_buffer_points, pcl::PointIndices::Ptr plane_inliers,
                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff);
  virtual bool checkCommon(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<pcl::PointXYZ>& ground_buffer_points,
                           pcl::PointIndices::Ptr plane_inliers);
  void setGroundNormal();

  pcl::PointCloud<pcl::PointXYZ>::Ptr removeFront(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

  ros::Subscriber sub_cloud_;
  ros::Publisher pub_downsampled_;
  ros::Publisher pub_filtered_;
  ros::Publisher pub_ground_;
  ros::Publisher pub_ground_free_;
  ros::Publisher pub_padding_;
  ros::Publisher pub_ramp_;

  std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> > reconfigure_server_;
  std::shared_ptr<dynamic_reconfigure::Server<DownsamplerRampConfig> > reconfigure_ramp_server_;

  double min_range_;
  double max_range_;
  double leaf_size_;
  double filter_radius_;
  int min_points_threshold_;

  std::atomic<bool> remove_ramp_;
  std::string sensor_frame_;
  std::string sensor_frame_overwrite_;
  std::mutex param_mutex_;
  double cut_off_distance_;

  int plane_fitting_type_;
  int plane_max_search_count_;
  int normal_neighbours_;

  double plane_max_deviation_;
  double ground_plane_padding_size_;
  double plane_max_degree_;
  double normal_distance_weight_;
  double normal_distance_weight_ramp_;

  std::vector<float> ground_plane_coeff_;
  std::vector<float> front_normal_;

  ros::Duration interval_;
  ros::Time next_call_time_;

  tf::TransformBroadcaster transform_broadcaster_;
};

} //end namespace

#endif /* DOWNSAMPLER_DOWNSAMPLER_H_ */
