#include <downsampler/downsampler.h>
#include <pluginlib/class_list_macros.h>

#include <boost/shared_ptr.hpp>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/console/print.h>

namespace downsampler
{

void Downsampler::onInit()
{
  ros::NodeHandle& nh = getNodeHandle();

  sub_cloud_ = nh.subscribe<sensor_msgs::PointCloud2>("input_cloud", 1, &Downsampler::downsample_cloud_cb, this);
  pub_downsampled_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("downsampled_points", 1);
  pub_filtered_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("filtered_points", 1);
  pub_ramp_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("ramp_points", 1);
  pub_ground_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("ground_points", 1);
  pub_result_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("result_points", 1);

  reconfigure_server_ = std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> >(
      new dynamic_reconfigure::Server<DownsamplerConfig>(nh));
  dynamic_reconfigure::Server<DownsamplerConfig>::CallbackType reconfigure_cb = boost::bind(&Downsampler::reconfigureCB,
                                                                                            this, _1, _2);
  reconfigure_server_->setCallback(reconfigure_cb);

  ros::NodeHandle& private_nh = getPrivateNodeHandle();

  private_nh.param("min_range", min_range_, 0.0);
  private_nh.param("max_range", max_range_, 4.0);
  private_nh.param("leaf_size", leaf_size_, 0.03);
  private_nh.param("filter_radius", filter_radius_, 0.03);
  private_nh.param("min_points_threshold", min_points_threshold_, 3);

  private_nh.param("plane_fitting_type", plane_fitting_type_, pcl::SAC_LMEDS);
  private_nh.param("plane_max_search_count", plane_max_search_count_, 200);
  private_nh.param("plane_max_deviation", plane_max_deviation_, 0.02);
  private_nh.param("plane_max_angle_degree", plane_max_angle_, 5.0);
  plane_max_angle_ = DEG2RAD(plane_max_angle_);

  double rate;
  private_nh.param("rate", rate, 30.0);

  if (rate == 0)
    interval_ = ros::Duration(0);
  else
    interval_ = ros::Duration(1.0 / rate);
  next_call_time_ = ros::Time::now();

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  ROS_INFO_STREAM("Downsampling points using a leaf size of '" << leaf_size_ << "' m, running at " << rate << " Hz.");
}

void Downsampler::reconfigureCB(DownsamplerConfig &config, uint32_t level)
{
  min_range_ = config.min_range;
  max_range_ = config.max_range;
  leaf_size_ = config.leaf_size;
  filter_radius_ = config.filter_radius;
  min_points_threshold_ = config.min_points_threshold;

  plane_fitting_type_ = config.plane_fitting_type;
  plane_max_search_count_ = config.plane_max_search_count;
  plane_max_deviation_ = config.plane_max_deviation;
  plane_max_angle_ = config.plane_max_angle_degree;
  plane_max_angle_ = DEG2RAD(plane_max_angle_);
}

void Downsampler::downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  if (ros::Time::now() <= next_call_time_)
    return;
  next_call_time_ = next_call_time_ + interval_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cut_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::PointCloud<pcl::PointXYZ> input_cloud;
  pcl::fromROSMsg(*cloud_msg, input_cloud);

  if (max_range_ != 0.0)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud.makeShared());
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_range_, max_range_);
    pass.filter(*cut_cloud);
  }
  else
  {
    cut_cloud = input_cloud.makeShared();
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (leaf_size_ == 0.0)
  {
    downsampled_cloud = cut_cloud;
  }
  else
  {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cut_cloud);
    sor.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    sor.filter(*downsampled_cloud);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (min_points_threshold_ > 0 && filter_radius_ > 0.0)
  {
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_removal;
    outlier_removal.setInputCloud(downsampled_cloud);
    outlier_removal.setRadiusSearch(filter_radius_);
    outlier_removal.setMinNeighborsInRadius(min_points_threshold_);
    outlier_removal.filter(*filtered_cloud);
  }
  else
  {
    filtered_cloud = downsampled_cloud;
  }

  if (pub_downsampled_.getNumSubscribers() > 0)
  {
    sensor_msgs::PointCloud2Ptr result_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered_cloud, *result_msg);

    pub_downsampled_.publish(result_msg);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr rampless_cloud = extractPlanes(filtered_cloud);

  sensor_msgs::PointCloud2Ptr result_msg(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*rampless_cloud, *result_msg);

  pub_result_.publish(result_msg);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_free(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr ramp(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr removed_planes(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());

  boost::shared_ptr<pcl::ModelCoefficients> ground_coefficients(new pcl::ModelCoefficients);
  boost::shared_ptr<pcl::ModelCoefficients> ramp_coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

  pcl::SACSegmentation<pcl::PointXYZ> segmentation;

  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setMethodType(plane_fitting_type_);
  segmentation.setDistanceThreshold(plane_max_deviation_); //we gotta catch'em all, so set that low
  segmentation.setMaxIterations(plane_max_search_count_); //wanna be the very best, so set that high

  segmentation.setAxis(Eigen::Vector3f(0, -0.095, -0.226)); ///TODO set automatically
  segmentation.setEpsAngle(plane_max_angle_);

  segmentation.setInputCloud(cloud);
  segmentation.segment(*inliers, *ground_coefficients);

  ///TODO make sure the ground is near the ground

  pcl::ExtractIndices<pcl::PointXYZ> remove_ground;
  remove_ground.setInputCloud(cloud);
  remove_ground.setIndices(inliers);
  remove_ground.setNegative(true);
  remove_ground.filter(*ground_free);

  if (pub_ground_.getNumSubscribers() > 0)
  {
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
    extract_ground.setInputCloud(cloud);
    extract_ground.setIndices(inliers);
    extract_ground.filter(*ground);

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ground, *ground_msg);
    pub_ground_.publish(ground_msg);
  }

  if (inliers->indices.size() == 0 || ground_coefficients->values.size() == 0)
  {
    ROS_INFO_THROTTLE(10, "[Downsampler]: Could not extract ground plane, no ground in field of view?");
    return cloud;
  }

  double ground_d = ground_coefficients->values[3];

  int i = 0;
  while (i <= 3)
  {
    pcl::PointIndices::Ptr ramp_inliers = extractRamp(ground_free, ground_coefficients->values, ramp_coefficients);
    ++i;

    if (ramp_coefficients->values.size() == 0)
    {
//      ROS_INFO("[Downsampler]: Could not extract ramp, no ramp in field of view?");
      return ground_free;
    }

//    if (std::abs(ramp_coefficients->values[3]) > 1.25) ////TODO calc value automatically
    {

      if (pub_ramp_.getNumSubscribers() > 0)
      {
        pcl::ExtractIndices<pcl::PointXYZ> extract_ramp;
        extract_ramp.setInputCloud(ground_free);
        extract_ramp.setIndices(ramp_inliers);
        extract_ramp.filter(*ramp);

        sensor_msgs::PointCloud2Ptr plane_msg(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(*ramp, *plane_msg);
        pub_ramp_.publish(plane_msg);
      }

      pcl::ExtractIndices<pcl::PointXYZ> extract_result;
      extract_result.setInputCloud(ground_free);
      extract_result.setNegative(true);
      extract_result.setIndices(ramp_inliers);
      extract_result.filter(*result);

      if (i > 1)
      {
        //move any non-ramp planes back in
        *result = *result + *removed_planes;
      }

      break;
    }

    //wrong plane extracted
    //remove it but keep it in the result
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
    extract_plane.setInputCloud(ground_free);
    extract_plane.setIndices(ramp_inliers);
    extract_plane.filter(*plane);

    *removed_planes = *removed_planes + *plane;

    pcl::PointCloud<pcl::PointXYZ>::Ptr new_ground_free(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_plane_invert;
    extract_plane.setInputCloud(ground_free);
    extract_plane.setIndices(ramp_inliers);
    extract_plane.setNegative(true);
    extract_plane.filter(*new_ground_free);

    ground_free = new_ground_free;
  }

  return result;
}

pcl::PointIndices::Ptr Downsampler::extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                std::vector<float>& axis,
                                                boost::shared_ptr<pcl::ModelCoefficients> coeff_out)
{
  pcl::SACSegmentation<pcl::PointXYZ> segmentation;

  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setOptimizeCoefficients(true);
  segmentation.setMethodType(plane_fitting_type_);
  segmentation.setMaxIterations(plane_max_search_count_);
  segmentation.setDistanceThreshold(plane_max_deviation_);

  segmentation.setAxis(Eigen::Vector3f(axis[0], axis[1], axis[2]));
  segmentation.setEpsAngle(plane_max_angle_); //5 degree = 0.0872665

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  segmentation.setInputCloud(input_cloud);
  segmentation.segment(*inliers, *coeff_out);

  return inliers;
}

} //end namespace

PLUGINLIB_EXPORT_CLASS(downsampler::Downsampler, nodelet::Nodelet)
