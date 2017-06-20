#include <downsampler/downsampler.h>
#include <pluginlib/class_list_macros.h>

#include <tf/transform_listener.h>
#include <limits>

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
#include <pcl/sample_consensus/sac_model_plane.h>

#include <std_msgs/Float32.h>

#include <algorithm>

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
  pub_ground_free_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("ground_free_points", 1);
  pub_padding_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("padding_points", 1);

  front_normal_.resize(4);

  reconfigure_server_ = std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> >(
      new dynamic_reconfigure::Server<DownsamplerConfig>());
  dynamic_reconfigure::Server<DownsamplerConfig>::CallbackType reconfigure_cb = boost::bind(&Downsampler::reconfigureCB,
                                                                                            this, _1, _2);
  reconfigure_server_->setCallback(reconfigure_cb);

  ros::NodeHandle ramp_nodehandle(nh, "ramp");
  reconfigure_ramp_server_ = std::shared_ptr<dynamic_reconfigure::Server<DownsamplerRampConfig> >(
      new dynamic_reconfigure::Server<DownsamplerRampConfig>(ramp_nodehandle));
  dynamic_reconfigure::Server<DownsamplerRampConfig>::CallbackType reconfigure_ramp_cb = boost::bind(&Downsampler::reconfigureRampCB,
                                                                                            this, _1, _2);
  reconfigure_ramp_server_->setCallback(reconfigure_ramp_cb);

  ros::NodeHandle& private_nh = getPrivateNodeHandle();

  private_nh.param("min_range", min_range_, min_range_);
  private_nh.param("max_range", max_range_, max_range_);
  private_nh.param("leaf_size", leaf_size_, leaf_size_);
  private_nh.param("filter_radius", filter_radius_, filter_radius_);
  private_nh.param("min_points_threshold", min_points_threshold_, min_points_threshold_);
  private_nh.param("sensor_frame", sensor_frame_, std::string("camera_depth_optical_frame"));
  private_nh.param("sensor_frame_overwrite", sensor_frame_overwrite_, std::string(""));

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
  std::lock_guard<std::mutex> lock(param_mutex_);

  min_range_ = config.min_range;
  max_range_ = config.max_range;
  leaf_size_ = config.leaf_size;
  filter_radius_ = config.filter_radius;
  min_points_threshold_ = config.min_points_threshold;
}

void Downsampler::reconfigureRampCB(DownsamplerRampConfig &config, uint32_t level)
{
  remove_ramp_ = config.remove_ramp;
  cut_off_distance_ = config.cut_off_distance;
  tf::Vector3 normal(front_normal_[0], front_normal_[1], front_normal_[2]);
  front_normal_[3] = -cut_off_distance_ * normal.length();

  plane_fitting_type_ = config.plane_fitting_type;
  plane_max_search_count_ = config.plane_max_search_count;
  plane_max_deviation_ = config.plane_max_deviation;
  ground_plane_padding_size_ = config.ground_plane_padding_size;
  plane_max_degree_ = config.plane_max_degree;

  normal_neighbours_ = config.normal_neighbours;
  normal_distance_weight_ = config.normal_distance_weight;
  normal_distance_weight_ramp_ = config.normal_distance_weight_ramp;
}

void Downsampler::downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  if (ros::Time::now() <= next_call_time_)
    return;
  next_call_time_ = next_call_time_ + interval_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cut_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::PointCloud<pcl::PointXYZ> input_cloud;
  pcl::fromROSMsg(*cloud_msg, input_cloud);

  std::lock_guard<std::mutex> lock(param_mutex_);

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

  pcl::PointCloud<pcl::PointXYZ>::Ptr result = filtered_cloud;

  if (remove_ramp_.load())
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr rampless_cloud = extractPlanes(filtered_cloud);
    result = removeFront(rampless_cloud);
  }

  sensor_msgs::PointCloud2Ptr result_msg(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*result, *result_msg);

  if (sensor_frame_overwrite_ != "")
  {
    result_msg->header.frame_id = sensor_frame_overwrite_;
  }

  pub_downsampled_.publish(result_msg);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_free(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::Normal>::Ptr ground_free_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());

  boost::shared_ptr<pcl::ModelCoefficients> ground_coefficients(new pcl::ModelCoefficients);

  pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());

  if (ground_plane_coeff_.empty())
  {
    setGroundNormal();
  }

  Eigen::Vector3f normal(ground_plane_coeff_[0], ground_plane_coeff_[1], ground_plane_coeff_[2]);

  Eigen::Vector3f ground_vec(-normal(1), normal(0), normal(2));

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;

  normal_estimation.setSearchMethod(tree);
  normal_estimation.setInputCloud(cloud);
  normal_estimation.setKSearch(normal_neighbours_);
  normal_estimation.compute(*cloud_normals);

  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> segmentation;

  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_NORMAL_PARALLEL_PLANE);
  segmentation.setInputNormals(cloud_normals);
  segmentation.setNormalDistanceWeight(normal_distance_weight_);
  segmentation.setMethodType(plane_fitting_type_);
  segmentation.setDistanceThreshold(plane_max_deviation_ + ground_plane_padding_size_); //we gotta catch'em all, so set that low
  segmentation.setMaxIterations(plane_max_search_count_); //wanna be the very best, so set that high

  segmentation.setAxis(normal);
  segmentation.setEpsAngle(DEG2RAD(plane_max_degree_));

  segmentation.setInputCloud(cloud);
  segmentation.segment(*ground_inliers, *ground_coefficients);

  ///TODO make sure the ground is near the ground

  std::vector<pcl::PointXYZ> ground_buffer_points;

  if (ground_inliers->indices.size() > 0 && ground_coefficients->values.size() != 0)
  {
    normal = Eigen::Vector3f(ground_coefficients->values[0], ground_coefficients->values[1],
                             ground_coefficients->values[2]);

    if (ground_coefficients->values[3] < 0)
    {
      for (int i = 0; i < ground_coefficients->values.size(); ++i)
      {
        ground_coefficients->values[i] *= -1;
      }
    }

    ground_buffer_points = filterIndices(cloud, ground_coefficients, ground_inliers);
  }

  if (pub_padding_.getNumSubscribers() > 0 && !ground_buffer_points.empty())
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr padding(new pcl::PointCloud<pcl::PointXYZ>());

    for (int i = 0; i < ground_buffer_points.size(); ++i)
    {
      padding->push_back(ground_buffer_points[i]);
    }
    padding->header = cloud->header;

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*padding, *ground_msg);
    pub_padding_.publish(ground_msg);
  }

  pcl::ExtractIndices<pcl::PointXYZ> remove_ground;
  remove_ground.setInputCloud(cloud);
  remove_ground.setIndices(ground_inliers);
  remove_ground.setNegative(true);
  remove_ground.filter(*ground_free);

  pcl::ExtractIndices<pcl::Normal> extract_normals;
  extract_normals.setInputCloud(cloud_normals);
  extract_normals.setIndices(ground_inliers);
  extract_normals.setNegative(true);
  extract_normals.filter(*ground_free_normals);

  if (pub_ground_free_.getNumSubscribers() > 0)
  {
    sensor_msgs::PointCloud2Ptr ground_free_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ground_free, *ground_free_msg);
    pub_ground_free_.publish(ground_free_msg);
  }

  if (pub_ground_.getNumSubscribers() > 0)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;

    extract_ground.setInputCloud(cloud);
    extract_ground.setIndices(ground_inliers);
    extract_ground.filter(*ground);

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ground, *ground_msg);
    pub_ground_.publish(ground_msg);
  }

  if (ground_buffer_points.empty())
  {
    return ground_free;
  }

  result = doStuff(ground_free, ground_free_normals, normal, ground_buffer_points);

  if (result->empty())
  {
    return ground_free;
  }

  return result;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::doStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                         pcl::PointCloud<pcl::Normal>::Ptr input_normals,
                                                         Eigen::Vector3f& axis,
                                                         std::vector<pcl::PointXYZ>& ground_buffer_points)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>(*input_cloud));
  pcl::PointCloud<pcl::PointXYZ>::Ptr ramp(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr removed_planes(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());

  boost::shared_ptr<pcl::ModelCoefficients> plane_coefficients(new pcl::ModelCoefficients);

  int i = 0;
  bool found = false;
  while (i <= 3)
  {
    pcl::PointIndices::Ptr plane_inliers = extractRamp(cloud, input_normals, axis, plane_coefficients);
    ++i;

    if (plane_coefficients->values.size() == 0)
    {
//      ROS_INFO("[Downsampler]: Could not extract ramp, no ramp in field of view?");
      break;
    }

    if (pub_ramp_.getNumSubscribers() > 0)
    {
      pcl::ExtractIndices<pcl::PointXYZ> extract_ramp;
      extract_ramp.setInputCloud(cloud);
      extract_ramp.setIndices(plane_inliers);
      extract_ramp.filter(*ramp);

      sensor_msgs::PointCloud2Ptr plane_msg(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(*ramp, *plane_msg);
      pub_ramp_.publish(plane_msg);
    }

    if (checkPlane(cloud, ground_buffer_points, plane_inliers, plane_coefficients) == Ramp)
    {
      result->clear();

      pcl::ExtractIndices<pcl::PointXYZ> extract_result;
      extract_result.setInputCloud(cloud);
      extract_result.setNegative(true);
      extract_result.setIndices(plane_inliers);
      extract_result.filter(*result);

      //move any non-ramp planes back in
      *result = *result + *removed_planes;

      found = true;
      break;
    }

    //wrong plane extracted
    //remove it but keep it in the result
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
    extract_plane.setInputCloud(cloud);
    extract_plane.setIndices(plane_inliers);
    extract_plane.filter(*plane);

    *removed_planes = *removed_planes + *plane;

    pcl::PointCloud<pcl::PointXYZ>::Ptr new_ground_free(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_plane_invert;
    extract_plane.setInputCloud(cloud);
    extract_plane.setIndices(plane_inliers);
    extract_plane.setNegative(true);
    extract_plane.filter(*new_ground_free);

    cloud = new_ground_free;
  }

  if (pub_ramp_.getNumSubscribers() > 0)
  {
    if (!found)
      ramp->clear();

    sensor_msgs::PointCloud2Ptr plane_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ramp, *plane_msg);
    pub_ramp_.publish(plane_msg);
  }

  if (!found)
    result->clear();

  return result;
}

std::vector<pcl::PointXYZ> Downsampler::filterIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                      boost::shared_ptr<pcl::ModelCoefficients> plane_coeff,
                                                      pcl::PointIndices::Ptr indices)
{
  std::vector<int> kept_indices;
  std::vector<pcl::PointXYZ> removed_points;

  double distance = 0;
  for (int i = 0; i < cloud->size(); ++i)
  {
    distance = pcl::pointToPlaneDistanceSigned(cloud->at(i), plane_coeff->values[0], plane_coeff->values[1],
                                               plane_coeff->values[2], plane_coeff->values[3]);

    if (distance <= plane_max_deviation_)
    {
      kept_indices.push_back(i);
    }
    else if (distance <= plane_max_deviation_ + ground_plane_padding_size_)
    {
      removed_points.push_back(cloud->at(i));
    }
  }

  indices->indices.clear();
  indices->indices.insert(indices->indices.begin(), kept_indices.begin(), kept_indices.end());

  return removed_points;
}

pcl::PointIndices::Ptr Downsampler::extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                pcl::PointCloud<pcl::Normal>::Ptr input_normals, Eigen::Vector3f& axis,
                                                boost::shared_ptr<pcl::ModelCoefficients> coeff_out)
{
  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> segmentation;

  segmentation.setModelType(pcl::SACMODEL_NORMAL_PARALLEL_PLANE);
  segmentation.setInputNormals(input_normals);
  segmentation.setNormalDistanceWeight(normal_distance_weight_ramp_);

  segmentation.setOptimizeCoefficients(true);
  segmentation.setMethodType(plane_fitting_type_);
  segmentation.setMaxIterations(plane_max_search_count_);
  segmentation.setDistanceThreshold(plane_max_deviation_);

  segmentation.setAxis(Eigen::Vector3f(axis(0), axis(1), axis(2)));
  segmentation.setEpsAngle(DEG2RAD(plane_max_degree_));

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  segmentation.setInputCloud(input_cloud);
  segmentation.segment(*inliers, *coeff_out);

  return inliers;
}

Downsampler::PlaneType Downsampler::checkPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                               std::vector<pcl::PointXYZ>& ground_buffer_points,
                                               pcl::PointIndices::Ptr plane_inliers,
                                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff)
{
  if (!checkCommon(cloud, ground_buffer_points, plane_inliers))
  {
    return NotATraversablePlane;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
  extract_plane.setInputCloud(cloud);
  extract_plane.setIndices(plane_inliers);
  extract_plane.filter(*plane);

  if (plane_inliers->indices.empty())
  {
    return NotATraversablePlane;
  }
  return Ramp;
}

bool Downsampler::checkCommon(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              std::vector<pcl::PointXYZ>& ground_buffer_points, pcl::PointIndices::Ptr plane_inliers)
{
  std::vector<int>::iterator it_inlier;
  std::vector<pcl::PointXYZ>::iterator it_ground;

  int count_common = 0;

  for (it_inlier = plane_inliers->indices.begin(); it_inlier != plane_inliers->indices.end(); ++it_inlier)
  {
    pcl::PointXYZ& point = cloud->at(*it_inlier);

    for (it_ground = ground_buffer_points.begin(); it_ground != ground_buffer_points.end(); ++it_ground)
    {
      if (it_ground->x == point.x && it_ground->y == point.y && it_ground->y == point.y)
      {
        ++count_common;
      }
    }
  }

  double min_indicies = std::min(ground_buffer_points.size(), plane_inliers->indices.size());

  double min_common_ratio = 0.1;
  double common_ratio = count_common / min_indicies;

//  ROS_INFO_STREAM(
//      "ground_buffer_points size: " << ground_buffer_points.size() << ", plane_inliers size: " << plane_inliers->indices.size());

  if (common_ratio < min_common_ratio)
  {
//    ROS_ERROR_STREAM("Common ratio < " << min_common_ratio << ": " << common_ratio << "; count: " << count_common);
    return false;
  }
  else
  {
//    ROS_INFO_STREAM("Common ratio < " << min_common_ratio << ": " << common_ratio << "; count: " << count_common);
  }

  return true;
}

void Downsampler::setGroundNormal()
{
  tf::TransformListener tf_listener;
  tf::StampedTransform tf_footprint_to_sensor;

  tf_listener.waitForTransform(sensor_frame_, "base_footprint", ros::Time::now(), ros::Duration(1.0));
  tf_listener.lookupTransform(sensor_frame_, "base_footprint", ros::Time(0), tf_footprint_to_sensor);

  tf::Vector3 origin = tf_footprint_to_sensor.getOrigin();

  tf::Vector3 z_offset_in_footprint(0.0, 0.0, origin.getZ());
  tf::Vector3 footprint_in_sensor = tf_footprint_to_sensor(tf::Vector3(0, 0, 0));
  tf::Vector3 z_offset_in_sensor = tf_footprint_to_sensor(z_offset_in_footprint);

  tf::Vector3 ground_normal = z_offset_in_sensor - footprint_in_sensor;

//  ROS_ERROR_STREAM_ONCE(
//      "ground_normal: " << ground_normal.getX() << ", " << ground_normal.getY() << ", " << ground_normal.getZ());

//  tf::Quaternion q(0, 0, 0, 1);
//  tf::Transform transform_test;
//  transform_test.setOrigin(ground_normal);
//  transform_test.setRotation(q);
//  transform_broadcaster_.sendTransform(
//      tf::StampedTransform(transform_test, ros::Time::now(), sensor_frame_, "ground_normal"));

  ground_plane_coeff_.push_back(ground_normal.getX());
  ground_plane_coeff_.push_back(ground_normal.getY());
  ground_plane_coeff_.push_back(ground_normal.getZ());
  ground_plane_coeff_.push_back(origin.getZ());

  tf::Vector3 sensor_ground_point_in_sensor = tf_footprint_to_sensor(tf::Vector3(origin.getX(), 0.0, 0.0));
  tf::Vector3 front_point_in_sensor = tf_footprint_to_sensor(tf::Vector3(origin.getX() + 1.0, 0.0, 0.0));

  tf::Vector3 front_normal = front_point_in_sensor - sensor_ground_point_in_sensor;

//  tf::Transform transform_front;
//  transform_front.setOrigin(front_normal);
//  transform_front.setRotation(q);
//  transform_broadcaster_.sendTransform(tf::StampedTransform(transform_front, ros::Time::now(), sensor_frame_, "front"));

  front_normal_[0] = front_normal.getX();
  front_normal_[1] = front_normal.getY();
  front_normal_[2] = front_normal.getZ();
  front_normal_[3] = -cut_off_distance_ * front_normal.length(); //see also http://mathworld.wolfram.com/Plane.html

//  ROS_ERROR_STREAM_ONCE(
//      "front_normal: " << front_normal_[0] << ", " << front_normal_[1] << ", " << front_normal_[2] << "; " << front_normal_[3]);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::removeFront(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  pcl::PointIndices::Ptr front_points(new pcl::PointIndices());

//  ROS_WARN_STREAM("cloud size: " << cloud->size());

  double distance = 0;
  for (int i = 0; i < cloud->size(); ++i)
  {
    distance = pcl::pointToPlaneDistanceSigned(cloud->at(i), front_normal_[0], front_normal_[1], front_normal_[2],
                                               front_normal_[3]);

//    ROS_INFO_STREAM_THROTTLE(
//        1,
//        i << ":" << cloud->at(i).x << ", " << cloud->at(i).y << ", " << cloud->at(i).z << " distance: " << distance << ", front normal: " << front_normal_[3]);
    if (distance >= 0)
    {
      front_points->indices.push_back(i);
    }
  }

//  ROS_INFO_STREAM("Removing " << front_points->indices.size() << " points");

  if (!front_points->indices.empty())
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr frontless_result(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract_front;
    extract_front.setInputCloud(cloud);
    extract_front.setIndices(front_points);
    extract_front.setNegative(true);
    extract_front.filter(*frontless_result);

    return frontless_result;
  }

  return cloud;
}

} //end namespace

PLUGINLIB_EXPORT_CLASS(downsampler::Downsampler, nodelet::Nodelet)
