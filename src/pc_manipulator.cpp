// system
#include <math.h>
#include <boost/shared_ptr.hpp>

// Eigen
#include <Eigen/Core>

// PCL
// general
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_ros/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/features/normal_3d.h>
//Filters and Downsampling
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
//Clustering
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>

// ROS
#include <ros/ros.h>
#include <pluginlib/class_list_macros.h>
#include <pcl_conversions/pcl_conversions.h>

boost::shared_ptr < pcl::PCLPointCloud2 > cloud_input(new pcl::PCLPointCloud2);
boost::shared_ptr < pcl::PCLPointCloud2 > cloud_filtered(new pcl::PCLPointCloud2);
boost::shared_ptr < pcl::PCLPointCloud2 > cloud_downsampled(new pcl::PCLPointCloud2);
boost::shared_ptr < pcl::PCLPointCloud2 > cloud_planes(new pcl::PCLPointCloud2);
boost::shared_ptr < pcl::PCLPointCloud2 > cloud_objects(new pcl::PCLPointCloud2);


void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  pcl_conversions::toPCL(*cloud_msg, *cloud_input);
//  ROS_INFO_STREAM("Received point cloud of size " << cloud_input->width << " (width) x " << cloud_input->height << " (height).");
}

void ground_extraction(const boost::shared_ptr<pcl::PCLPointCloud2> cloud_in,
                       boost::shared_ptr<pcl::PCLPointCloud2> cloud_out_plane,
                       boost::shared_ptr<pcl::PCLPointCloud2> cloud_out_objects,
                       const double cloud_proc_perc)
{
  boost::shared_ptr<pcl::PCLPointCloud2> temp(new pcl::PCLPointCloud2);
  *temp = *cloud_in;

  //// Did I mention pointers?
  pcl::PCLPointCloud2ConstPtr tempPtr(temp);
  pcl::PointCloud<pcl::PointXYZ>::Ptr groundcloudPtr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_o(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::ModelCoefficients::Ptr groundcoeffPtr(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr plane_inliners_ptr(new pcl::PointIndices);
  pcl::PointCloud<pcl::PointXYZ>::Ptr groundpointsPtr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr objectpointsPtr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudprojPtr(new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr groundhullPtr(new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::PointIndices::Ptr objectindicPtr(new pcl::PointIndices);

  // Find the planes using RANSAC
  // If the plane's slope is between 0 and 8 degrees consider it as a flat surface
  pcl::fromPCLPointCloud2(*temp, *groundcloudPtr);

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setMaxIterations (200);
  seg.setDistanceThreshold (0.015); // 0.0195
//  seg.setInputCloud (groundcloudPtr);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setModelType (pcl::SACMODEL_PLANE);

  // Look for all planes in the cloud
  // The RANSAC segmentation will always try to create a plane matching the majority of available points.
  // Hence, we exclude those points, once a plane with an accetable slope has been found.

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZ> extract;

  groundpointsPtr->header = groundcloudPtr->header;
  objectpointsPtr->header = groundcloudPtr->header;

  ROS_INFO_STREAM("---------------------");

  int i = 0, nr_points = (int) groundcloudPtr->points.size ();
  // While cloud_proc_perc of the original cloud is still there
  while (groundcloudPtr->points.size () > cloud_proc_perc * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (groundcloudPtr);
    seg.segment (*plane_inliners_ptr, *groundcoeffPtr);
    if (plane_inliners_ptr->indices.size () == 0)
    {
      ROS_INFO_STREAM("Could not find a plane in the remaining point cloud.");
      *objectpointsPtr += *groundcloudPtr;
      break;
    }
    else
    {
      // Extract the inliers
      extract.setInputCloud (groundcloudPtr);
      extract.setIndices (plane_inliners_ptr);
      extract.setNegative (false);
      extract.filter (*cloud_p);

      // check if the slope is acceptable
      Eigen::Vector3f normal_ground_plane(0, -0.7071, -0.7071); // ground plane vector
      Eigen::Vector3f normal_found_plane(groundcoeffPtr->values[0], groundcoeffPtr->values[1], groundcoeffPtr->values[2]);
      double dot, angle_rad, angle_deg;
      dot = normal_found_plane.normalized().dot(normal_ground_plane.normalized());
      if (dot > 0.0)
      {
        angle_rad = -1 * std::acos(dot);
      }
      else
      {
        angle_rad = std::acos(-1 * dot);
      }
      angle_deg = (angle_rad / M_PI) * 180;
      if ((angle_deg >= -10.0) && (angle_deg <= 10.0))
      {
        ROS_INFO_STREAM("Found plane with acceptable slope of " << angle_deg << " degress and added it to the ground cloud.");
        ROS_INFO_STREAM("Plane coefficients: " << groundcoeffPtr->values[0] << ", "
                                               << groundcoeffPtr->values[1] << ", "
                                               << groundcoeffPtr->values[2] << ", "
                                               << groundcoeffPtr->values[3]);
        *groundpointsPtr += *cloud_p;
      }
      else
      {
        ROS_INFO_STREAM("Found plane with non-acceptable slope of " << angle_deg << " degress and added it to the obstacle cloud.");
        *objectpointsPtr = *cloud_p;
      }
      // Create the filtering object
      extract.setNegative (true);
      extract.filter (*cloud_o);
      groundcloudPtr.swap (cloud_o);
      i++;
    }
  }
  ROS_INFO_STREAM("Fitted at least 10% of the whole point cloud into plains. Adding the remaining " << groundcloudPtr->points.size()
                  << " to the obstacle cloud.");
  *objectpointsPtr += *groundcloudPtr;

  // Project the ground inliers
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);
  proj.setInputCloud(groundpointsPtr);
  groundcoeffPtr->values[0] = 0;
  groundcoeffPtr->values[1] = -0.7071;
  groundcoeffPtr->values[2] = -0.7071;
  groundcoeffPtr->values[3] = 0.90;
  proj.setModelCoefficients(groundcoeffPtr);
  proj.filter(*cloudprojPtr);

//  pcl::toPCLPointCloud2(*groundpointsPtr, *cloud_out_plane);
  pcl::toPCLPointCloud2(*cloudprojPtr, *cloud_out_plane);

  pcl::toPCLPointCloud2(*objectpointsPtr, *cloud_out_objects);
}

int main(int argc, char * argv[])
{
  ros::init(argc, argv, "pc_manipulator");
  ros::NodeHandle nh, nh_priv("~");

  ros::Subscriber sub_cloud = nh.subscribe < sensor_msgs::PointCloud2 > ("input_cloud", 1, &cloud_cb);
  ros::Publisher pub_ground_cloud_ = nh.advertise < sensor_msgs::PointCloud2 > ("ground_cloud", 1);
  ros::Publisher pub_objects_cloud_ = nh.advertise < sensor_msgs::PointCloud2 > ("objects_cloud", 1);

  double min_range, max_range, leaf_size, cloud_proc_perc;
  nh_priv.param("min_range", min_range, 0.0);
  nh_priv.param("max_range", max_range, 10.0);
  nh_priv.param("leaf_size", leaf_size, 0.01);
  nh_priv.param("cloud_proc_perc", cloud_proc_perc, 0.01);

  ros::Duration interval;
  double rate;
  nh_priv.param("rate", rate, 30.0);
  if(rate == 0)
  {
    interval = ros::Duration(0);
  }
  else
  {
    interval = ros::Duration(1.0 / rate);
  }
  ROS_INFO_STREAM("Downsampling points using a leaf size of '" << leaf_size << "' m, running at " << rate << " Hz.");

  while(ros::ok())
  {
    ros::spinOnce();
    if (cloud_input && (cloud_input->height > 0) && (cloud_input->width > 0))
    {
      // limit z
      pcl::PassThrough < pcl::PCLPointCloud2 > pass;
      pass.setInputCloud(cloud_input);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(min_range, max_range);
      pass.filter(*cloud_filtered);

      // downsampling
      if (leaf_size != 0)
      {
        pcl::VoxelGrid < pcl::PCLPointCloud2 > sor;
        sor.setInputCloud(cloud_filtered);
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(*cloud_downsampled);
      }
      // filter planes
      ground_extraction(cloud_downsampled, cloud_planes, cloud_objects, cloud_proc_perc);

      // publish ground plane and objects
      if (pub_ground_cloud_.getNumSubscribers() > 0)
      {
        sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2);
        pcl_conversions::moveFromPCL(*cloud_planes, *cloud_msg);
        pub_ground_cloud_.publish(cloud_msg);
      }
      if (pub_objects_cloud_.getNumSubscribers() > 0)
      {
        sensor_msgs::PointCloud2Ptr cloud_msg(new sensor_msgs::PointCloud2);
        pcl_conversions::moveFromPCL(*cloud_objects, *cloud_msg);
        pub_objects_cloud_.publish(cloud_msg);
      }
    }
    interval.sleep();
  }

  return 0;
}

