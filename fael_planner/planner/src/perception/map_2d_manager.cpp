//
// Created by hjl on 2022/1/11.
//

#include "perception/map_2d_manager.h"

namespace perception {

    Map2DManager::Map2DManager(ros::NodeHandle &nh, ros::NodeHandle &nh_private)
            : nh_(nh), nh_private_(nh_private), is_map_updated_(false) {
        getParamsFromRos();
        grid_map_2d_pub_ = nh_private_.advertise<visualization_msgs::MarkerArray>("gird_map_2d", 1);
        wall_map_2d_pub_ = nh_private_.advertise<nav_msgs::OccupancyGrid>("wall_map_2d", 1);

        odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("base_odometry", 1, &Map2DManager::odomCallback, this);

        terrain_map_sub_ = nh_.subscribe<traversability_analysis::TerrainMap>("terrain_map", 1,
                                                                              &Map2DManager::terrainMapCallback, this);

        terrain_cloud_sub_.reset(
                new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, "terrain_point_cloud", 1));
        local_cloud_sub_.reset(
                new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, "voxelized_local_point_cloud", 1));
        sync_terrain_local_cloud_.reset(
                new message_filters::Synchronizer<SyncPolicyLocalCloud>(SyncPolicyLocalCloud(1), *terrain_cloud_sub_,
                                                                        *local_cloud_sub_));
        sync_terrain_local_cloud_->registerCallback(
                boost::bind(&Map2DManager::terrainLocalCloudCallback, this, _1, _2));

        wall_cloud_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("/trolley/lidar/wall", 1, &Map2DManager::wallCallback, this);

    }

    void Map2DManager::getParamsFromRos() {
        std::string ns = ros::this_node::getName() + "/GridMap2D";

        frame_id_ = "world";
        if (!ros::param::get(ns + "/frame_id", frame_id_)) {
            ROS_WARN("No frame_id specified. Looking for %s. Default is 'map'.",
                     (ns + "/frame_id").c_str());
        }

        grid_size_ = 0.3;
        if (!ros::param::get(ns + "/grid_size", grid_size_)) {
            ROS_WARN("No grid_size specified. Looking for %s. Default is 0.3.",
                     (ns + "/grid_size").c_str());
        }

        inflate_radius_ = 0.3;
        if (!ros::param::get(ns + "/inflate_radius", inflate_radius_)) {
            ROS_WARN("No inflate_radius specified. Looking for %s. Default is 0.3 .",
                     (ns + "/inflate_radius").c_str());
        }

        inflate_empty_radius_ = 0.6;
        if (!ros::param::get(ns + "/inflate_empty_radius", inflate_empty_radius_)) {
            ROS_WARN("No inflate_empty_radius specified. Looking for %s. Default is 0.6 .",
                     (ns + "/inflate_empty_radius").c_str());
        }

        lower_z_ = 0.03;
        if (!ros::param::get(ns + "/lower_z", lower_z_)) {
            ROS_WARN("No lower_z specified. Looking for %s. Default is 0.03 .",
                     (ns + "/lower_z").c_str());
        }

        connectivity_thre_ = 0.1;
        if (!ros::param::get(ns + "/connectivity_thre", connectivity_thre_)) {
            ROS_WARN("No connectivity_thre specified. Looking for %s. Default is 0.1 .",
                     (ns + "/connectivity_thre").c_str());
        }

        double fix_x = 100.0;
        double fix_y = 100.0;
        wall_map_.initialize(grid_size_, fix_x, fix_y, Status2D::Unknown);
        Eigen::Vector2d center_point(20.0, 20.0);
        wall_map_.setMapCenterAndBoundary(center_point);

    }

    void Map2DManager::odomCallback(const nav_msgs::OdometryConstPtr &odom) {
        current_pose_ = odom->pose.pose;
    }

    void Map2DManager::wallCallback(const sensor_msgs::PointCloud2ConstPtr &wall_cloud) {
        pcl::fromROSMsg(*wall_cloud, wall_cloud_);
        //ROS_WARN("CALL BACK CLOUD SIZE=%d,", wall_cloud_.points.size());
        nav_msgs::OccupancyGrid map_topic_msg;
        setMapTopicMsg(wall_cloud_, map_topic_msg);
        wall_map_2d_pub_.publish(map_topic_msg);

        //grid map to cv
        cv::Mat img(map_topic_msg.info.height, map_topic_msg.info.width, CV_8U);
        img.data = (unsigned char *)(&(map_topic_msg.data[0]) );
        cv::Mat received_image = img.clone();

        cv::imwrite("/home/sustech1411/img_in.png", received_image);

        cv::Mat image_cleaned = cv::Mat::zeros(received_image.size(), CV_8UC1);
        cv::Mat black_image   = cv::Mat::zeros(received_image.size(), CV_8UC1);
        image_cleaned = clean_image2(received_image, black_image);

        cv::imwrite("/home/sustech1411/img_out.png", image_cleaned);

        cv::Mat image_mid;
        cv::Canny(image_cleaned, image_mid, 50, 100);       
        cv::imwrite("/home/sustech1411/canny.png", image_mid);
        //to color img
        cv::Mat three_channel = cv::Mat::zeros(image_mid.rows,image_mid.cols,CV_8UC3);
        vector<cv::Mat> channels;
        for (int i=0;i<3;i++)
        {
            channels.push_back(image_mid);
        }
        cv::merge(channels,three_channel);

        vector<cv::Vec4i> lines;
        cv::HoughLinesP(image_mid, lines, 1, Pi/180, 20, 5, 5); 
        //ROS_WARN("size1=%d",lines.size());
        hough_lines.clear();
        for(size_t i = 0; i < lines.size(); i++ )  
        {      
            cv::Vec4i l = lines[i];  
            //cv::line(three_channel, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(155, 100, 50), 1);  

            HoughLine new_line;
            cv::Point2i start_id(l[0], l[1]), end_id(l[2], l[3]);
            new_line.start_p = start_id;
            new_line.end_p = end_id;
            //new_line.line_k = (l[3] - l[1]) / (l[2] - l[0]);
            new_line.id = 0;
            new_line.length = sqrt((l[0] - l[2]) * (l[0] - l[2]) + (l[1] - l[3]) * (l[1] - l[3]));
            hough_lines.emplace_back(new_line);
        }  
        //cv::imwrite("/home/sustech1411/hough1.png", three_channel);


  

        //cluster and extend lines
        ROS_WARN("line num before = %d", hough_lines.size());
        //MergeLines();
        //ExtendLines();
        ROS_WARN("line num after = %d", hough_lines.size());
        for(size_t i = 0; i < hough_lines.size(); i++ )  
        {   
            HoughLine l = hough_lines[i];  
            //cout<<l.start_p.x<<" "<<l.start_p.y<<" "<<l.end_p.x<<" "<<l.end_p.y<<endl;
            cv::line(three_channel, cv::Point(l.start_p.x, l.start_p.y), cv::Point(l.end_p.x, l.end_p.y), cv::Scalar(155, 100, 50), 1);  
        }  
        cv::imwrite("/home/sustech1411/cluster_extend.png", three_channel);

        //detect polygons
        std::vector<PolyLine> poly_lines;
        for (auto line : hough_lines) {           
            PolyLine poly_line({(float)line.start_p.x, (float)line.start_p.y}, {(float)line.end_p.x, (float)line.end_p.y});
            //cout<<"line "<<line.start_p.x<<" "<<line.start_p.y<<" "<<line.end_p.x<<" "<<line.end_p.y<<endl;
            poly_lines.emplace_back(poly_line);
        }
         
        PolyDetector pd; // can be reused
        for (auto &l : poly_lines)
            pd.AddLine(l);

        if (!pd.DetectPolygons()) // can be reused
        {
            logoutf("%s", "WARN: cannot detect polys!");
            //return -1;
        }


        for (auto &poly : pd.polys) // here are the detected polys
        {
            std::vector<cv::Point> pts;
            int count = 0;
            for (auto &p : poly.p)
            {
                logoutf("[%u] p:{%f %f}", poly.id, p.x, p.y);

                pts.push_back(cv::Point(p.x, p.y));
            }
            cv::polylines(three_channel, pts, true, cv::Scalar(255, 255, 0), 2);
        }
        cv::imwrite("/home/sustech1411/polys.png", three_channel);
        



    }

    void Map2DManager::ExtendLines()
    {
        for (auto& line : hough_lines) {
            //ROS_WARN(" len before = %f", line.length);
            Eigen::Vector2d vec1(line.end_p.x - line.start_p.x, line.end_p.y - line.start_p.y);
            vec1 *= ((line.length + 15) / line.length);

            Eigen::Vector2d vec2(line.start_p.x - line.end_p.x, line.start_p.y - line.end_p.y);
            vec2 *= ((line.length + 15) / line.length);
            int cur_end_x = line.end_p.x;
            int cur_end_y = line.end_p.y;

            line.end_p.x = vec1(0) + line.start_p.x;
            line.end_p.y = vec1(1) + line.start_p.y;

            line.start_p.x = vec2(0) + cur_end_x;
            line.start_p.y = vec2(1) + cur_end_y;

            line.length = sqrt((line.start_p.x - line.end_p.x) * (line.start_p.x - line.end_p.x) 
                + (line.start_p.y - line.end_p.y) * (line.start_p.y - line.end_p.y));
            //ROS_WARN(" len after = %f", line.length);
        }
    }

    inline bool Map2DManager::CanMergeLine(const HoughLine& line1, const HoughLine& line2)
    {
        cv::Point2i p11 = line1.start_p;
        cv::Point2i p12 = line1.end_p;

        //cout<<"lin1: "<<"("<<p11.x<<","<<p11.y<<")"<<"("<<p12.x<<","<<p12.y<<")"<<endl;
        double A1 = p12.y - p11.y;
        double B1 = p11.x - p12.x;
        double C1 = p12.x * p11.y - p11.x * p12.y;

        cv::Point2i p21 = line2.start_p;
        cv::Point2i p22 = line2.end_p;
        //cout<<"lin2: "<<"("<<p21.x<<","<<p21.y<<")"<<"("<<p22.x<<","<<p22.y<<")"<<endl;
        double d1 = fabs(A1 * p21.x + B1 * p21.y + C1) / sqrt(A1 * A1 + B1 * B1);
        double d2 = fabs(A1 * p22.x + B1 * p22.y + C1) / sqrt(A1 * A1 + B1 * B1);
        //cout<<"d="<<d1<<" "<<d2<<endl;
//ROS_WARN("di,d2=%f,%f",d1,d2);   
       
        if (d1 > 5 || d2 > 5 ) {
            return false;
        }
        //return true;

        HoughLine line_s, line_l;
        if (line1.length > line2.length) {
            line_s = line2;
            line_l = line1;
        } else {
            line_s = line1;
            line_l = line2;
        }
        cv::Point2i pl1 = line_l.start_p;
        cv::Point2i pl2 = line_l.end_p;
        cv::Point2i ps1 = line_s.start_p;
        cv::Point2i ps2 = line_s.end_p;

        double line_l_k, line_l_b, pro_p1_x, pro_p1_y, pro_p2_x, pro_p2_y;
        if (pl2.x != pl1.x) {
            line_l_k = (pl2.y - pl1.y) / (pl2.x - pl1.x);
            line_l_b = pl2.y - pl2.x * line_l_k;
            pro_p1_x = (line_l_k * (ps1.y - line_l_b) + ps1.x) / (1 + line_l_k * line_l_k);
            pro_p1_y = line_l_k * pro_p1_x + line_l_b;
            pro_p2_x = (line_l_k * (ps2.y - line_l_b) + ps2.x) / (1 + line_l_k * line_l_k);
            pro_p2_y = line_l_k * pro_p2_x + line_l_b;
        } else {
            pro_p1_x = pl2.x;
            pro_p2_x = pl2.x;
            pro_p1_y = ps1.y;
            pro_p2_y = ps2.y;
        }

        Eigen::Vector2d vec11(pl1.x - pro_p1_x, pl1.y - pro_p1_y);
        Eigen::Vector2d vec12(pl2.x - pro_p1_x, pl2.y - pro_p1_y);
        Eigen::Vector2d vec21(pl1.x - pro_p2_x, pl1.y - pro_p2_y);
        Eigen::Vector2d vec22(pl2.x - pro_p2_x, pl2.y - pro_p2_y);
        //ROS_WARN("finish canmerge %f  %f",vec11.dot(vec12),vec21.dot(vec22));
        if ((vec11.dot(vec12) < 0 && vec21.dot(vec22) < 0) ||
            (vec11.dot(vec12) * vec21.dot(vec22) <= 0))
            return true;
            
        return false;
    }

    void Map2DManager::MergeLines()
    {
        vector<HoughLine> lines_out;
        vector<vector<int>> rm_ids;
        rm_ids.resize(hough_lines.size());

        int cur_id = 1;
        for (int i = 0; i < hough_lines.size(); i++) {
            vector<int> rm_id;
            HoughLine cur_line = hough_lines[i];
            if (cur_line.id == 0) {
                //cout<<"new id="<<cur_id<<endl;
                hough_lines[i].id = cur_id;
                cur_id++;
            } else {
                continue;
            }
            
            for (int j = i + 1; j < hough_lines.size(); j++) {
                //cout<<j<<" "<<hough_lines.size()<<endl;
                HoughLine next_line = hough_lines[j];
                if (next_line.id != 0) continue; 
                if (CanMergeLine(cur_line, next_line)) {
                    //cout<<"canmerge id="<<cur_id<<endl;
                    hough_lines[j].id = cur_id;
                    
                    rm_id.emplace_back(j);
                } 
                if (!rm_id.empty()) {
                    rm_ids[i] = rm_id;
                }
            }
           
        }

        vector<int> del_id_set;
        for (int i = 0; i < rm_ids.size(); i++) {
            if (rm_ids[i].empty())
                continue;
            del_id_set.emplace_back(i);
            for (auto j : rm_ids[i])
                del_id_set.emplace_back(j);
        }
        for (int i = 0; i < hough_lines.size(); i++) {
            vector<int>::iterator it = find(del_id_set.begin(), del_id_set.end(), i);
            if (it != del_id_set.end())
                continue;
            lines_out.emplace_back(hough_lines[i]);
        }

        vector<HoughLine> line_rep;
        line_rep = getLineRep(rm_ids);
        //cout<<"size "<<line_rep.size()<<endl;
        for (auto line : line_rep) {
            lines_out.emplace_back(line);
        }



        /*for (int id = 1; id < cur_id; id++) {
            //cout<<"id="<<id<<endl;
            int max_id = 0;
            double max_len = -1e3;
            for (int i = 0; i < hough_lines.size(); i++) {
                if (hough_lines[i].id == id) {
                    //cout<<"i="<<i<<endl;
                    if (hough_lines[i].length > max_len) {
                        max_id = i;
                    }
                }
            }
            lines_out.emplace_back(hough_lines[max_id]);
            //cout<<"max id="<<max_id<<endl;
        }*/

        hough_lines = lines_out;
    }  
    
    std::vector<HoughLine> Map2DManager::getLineRep(vector<vector<int>> id_clusters)
    {
        vector<HoughLine> line_set;
        for (int i = 0; i < id_clusters.size(); i++) {
            if (id_clusters[i].empty())
                continue;

            id_clusters[i].emplace_back(i);
            //cout<<"new cluster"<<endl;
            double min_start_x{1e3}, min_start_y{1e3}, max_start_x{-1e3}, max_start_y{-1e3};
            for (int j : id_clusters[i]) {
                //cout<<"id="<<j<<endl;
                HoughLine cur_line = hough_lines[j];
                int min_x = min(cur_line.start_p.x, cur_line.end_p.x);
                int max_x = max(cur_line.start_p.x, cur_line.end_p.x);
                int min_y = min(cur_line.start_p.y, cur_line.end_p.y);
                int max_y = max(cur_line.start_p.y, cur_line.end_p.y);
                if (min_x < min_start_x)
                    min_start_x = min_x;
                if (min_y < min_start_y)
                    min_start_y = min_y;
                if (max_x > max_start_x)
                    max_start_x = max_x;
                if (max_y > max_start_y)
                    max_start_y = max_y;
            }

            HoughLine new_line;
            new_line.start_p.x = min_start_x;
            new_line.start_p.y = min_start_y;
            new_line.end_p.x = max_start_x;
            new_line.end_p.y = max_start_y;
            new_line.length = sqrt((min_start_x - max_start_x) * (min_start_x - max_start_x)
                 + (min_start_y - max_start_y) * (min_start_y - max_start_y));
            line_set.emplace_back(new_line);

        }
        return line_set;
    }

    cv::Mat Map2DManager::clean_image2(cv::Mat Occ_Image, cv::Mat &black_image){
        //Occupancy Image to Free Space	
        cv::Mat open_space = Occ_Image<10;
        black_image = Occ_Image>90 & Occ_Image<=100;		
        cv::Mat Median_Image, out_image, temp_image ;
        int filter_size=2;

        //cv::imwrite("/home/sustech1411/step0.png", black_image);
        cv::boxFilter(black_image, temp_image, -1, cv::Size(filter_size, filter_size), cv::Point(-1,-1), false, cv::BORDER_DEFAULT ); // filter open_space
        black_image = temp_image > filter_size*filter_size/2;  // threshold in filtered
        //cv::imwrite("/home/sustech1411/step1.png", black_image);

        //cv::dilate(black_image, black_image, cv::Mat(), cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue() );			// inflate obstacle
        //cv::imwrite("/home/sustech1411/step2.png", black_image);

        filter_size=10;
        cv::boxFilter(open_space, temp_image, -1, cv::Size(filter_size, filter_size), cv::Point(-1,-1), false, cv::BORDER_DEFAULT ); // filter open_space
        Median_Image = temp_image > filter_size*filter_size/2;  // threshold in filtered
        Median_Image = Median_Image | open_space ;
        //cv::imwrite("/home/sustech1411/step3.png", Median_Image);
        //cv::medianBlur(Median_Image, Median_Image, 3);
        cv::dilate(Median_Image, Median_Image,cv::Mat());
        //cv::imwrite("/home/sustech1411/step4.png", Median_Image);
        out_image = Median_Image & ~black_image;// Open space without obstacles

        return out_image;
    }

    void Map2DManager::setMapTopicMsg(const pcl::PointCloud<pcl::PointXYZ> cloud,
                    nav_msgs::OccupancyGrid &msg) {
                        //ROS_WARN("SET MSG");
        if (!have_wall_map) {
            //ROS_WARN("SET MSG FALSE");
            msg.header.seq = 0;
            msg.header.stamp = ros::Time::now();
            msg.header.frame_id = frame_id_;

            msg.info.map_load_time = ros::Time::now();
            msg.info.resolution = grid_size_;

            double x_min, x_max, y_min, y_max;
            /*double z_max_grey_rate = 0.05;
            double z_min_grey_rate = 0.95;
            //? ? ??
            double k_line =
                (z_max_grey_rate - z_min_grey_rate) / (thre_z_max - thre_z_min);
            double b_line =
                (thre_z_max * z_min_grey_rate - thre_z_min * z_max_grey_rate) /
                (thre_z_max - thre_z_min);*/

            if (cloud.points.empty()) {
                ROS_WARN("pcd is empty!\n");
                return;
            }

            for (int i = 0; i < cloud.points.size() - 1; i++) {
                if (i == 0) {
                x_min = x_max = cloud.points[i].x;
                y_min = y_max = cloud.points[i].y;
                }

                double x = cloud.points[i].x;
                double y = cloud.points[i].y;

                if (x < x_min)
                x_min = x;
                if (x > x_max)
                x_max = x;

                if (y < y_min)
                y_min = y;
                if (y > y_max)
                y_max = y;
            }
            // origin的确定
            msg.info.origin.position.x = -10.0;
            msg.info.origin.position.y = -10.0;
            msg.info.origin.position.z = 0.0;
            msg.info.origin.orientation.x = 0.0;
            msg.info.origin.orientation.y = 0.0;
            msg.info.origin.orientation.z = 0.0;
            msg.info.origin.orientation.w = 1.0;
            //设置栅格地图大小
            msg.info.width = 1000;
            msg.info.height = 1000;
            //实际地图中某点坐标为(x,y)，对应栅格地图中坐标为[x*map.info.width+y]
            msg.data.resize(msg.info.width * msg.info.height);
            msg.data.assign(msg.info.width * msg.info.height, 0);

            ROS_INFO("data size = %d", msg.data.size());

            for (int iter = 0; iter < cloud.points.size(); iter++) {
                int i = int((cloud.points[iter].x - msg.info.origin.position.x) / grid_size_);
                if (i < 0 || i >= msg.info.width)
                    continue;

                int j = int((cloud.points[iter].y - msg.info.origin.position.y) / grid_size_);
                if (j < 0 || j >= msg.info.height - 1)
                    continue;
                // 栅格地图的占有概率[0,100]，这里设置为占据
                msg.data[i + j * msg.info.width] = 100;
                //    msg.data[i + j * msg.info.width] = int(255 * (cloud.points[iter].z *
                //    k_line + b_line)) % 255;
                last_msg = msg;
            }
        } else {
            //ROS_WARN("SET MSG TRUE");
            for (int iter = 0; iter < cloud.points.size(); iter++) {
                int i = int((cloud.points[iter].x - last_msg.info.origin.position.x) / grid_size_);
                if (i < 0 || i >= last_msg.info.width)
                    continue;

                int j = int((cloud.points[iter].y - last_msg.info.origin.position.y) / grid_size_);
                if (j < 0 || j >= last_msg.info.height - 1)
                    continue;
                last_msg.data[i + j * last_msg.info.width] = 100;
                //ROS_WARN("id=%d",i + j * last_msg.info.width);
                msg = last_msg;
            }
        }
        have_wall_map = true;
    }

    void Map2DManager::terrainLocalCloudCallback(const sensor_msgs::PointCloud2ConstPtr &terrain_cloud,
                                                 const sensor_msgs::PointCloud2ConstPtr &local_cloud) {
        pcl::fromROSMsg(*terrain_cloud, terrain_cloud_);
        pcl::fromROSMsg(*local_cloud, local_cloud_);

        if (terrain_cloud->header.frame_id != frame_id_) {
            if (!pcl_ros::transformPointCloud(frame_id_, terrain_cloud_, terrain_cloud_, tf_listener_))//Convert to global frame.
                return;
        }

        if (local_cloud->header.frame_id != frame_id_) {
            if (!pcl_ros::transformPointCloud(frame_id_, local_cloud_, local_cloud_, tf_listener_))//Convert to global frame.
                return;
        }

        updateGridMap2D(terrain_cloud_, local_cloud_);//Use terrain point cloud to update 2D grid status, and alloc grid to local point cloud
        grid_map_2d_pub_.publish(inflate_map_.generateMapMarkers(inflate_map_.grids_, current_pose_));
    }

    void Map2DManager::updateWallGridMap2D(const pcl::PointCloud<pcl::PointXYZI> &wall_cloud) 
    {
        Eigen::Vector2d center_point(current_pose_.position.x, current_pose_.position.y);//center point relative to the global frame
        pcl::PointXYZI min_p;
        pcl::PointXYZI max_p;
        pcl::getMinMax3D(wall_cloud, min_p, max_p);

        /*Index2D min_pt(wall_map_.getIndexInMap2D(min_p));
        Index2D max_pt(wall_map_.getIndexInMap2D(max_p));
        for (int i = min_pt(0); i <= max_pt(0); i++) {
            for (int j = min_pt(1); j <= max_pt(1); j++) {
                wall_map_.setFree(i, j);
            }
        }*/

        for (auto &point: wall_cloud) { 
            if (wall_map_.isInMapRange2D(point)) {
                wall_map_.addPointInGrid(wall_map_.getIndexInMap2D(point), point);       
                wall_map_.setOccupied(wall_map_.getIndexInMap2D(point));
                cout<<"occ pt id: "<<wall_map_.getIndexInMap2D(point)(0)<<" "<<wall_map_.getIndexInMap2D(point)(1)<<endl;
            } 
        }
        wall_map_.inflateGridMap2D(inflate_radius_, inflate_empty_radius_);

        //have_wall_map = true;
    }

    void Map2DManager::terrainMapCallback(const traversability_analysis::TerrainMapConstPtr &terrain_map) {
        
        TerrainMap terrain_map_;

        pcl::PointXYZI bottom_point;
        pcl_conversions::toPCL(terrain_map->header, terrain_map_.bottom_points.header);
        for (const auto &item :terrain_map->grids) {
            terrain_map_.status.push_back(item.status);
            bottom_point.x = item.bottom_point.x;
            bottom_point.y = item.bottom_point.y;
            bottom_point.z = item.bottom_point.z;
            terrain_map_.bottom_points.push_back(bottom_point);
        }

        if (terrain_map->header.frame_id == frame_id_) { 
            terrain_map_.frame_id = terrain_map->header.frame_id;
            terrain_map_.min_x = terrain_map->min_x;
            terrain_map_.min_y = terrain_map->min_y;
            terrain_map_.z_value = terrain_map->z_value;
        } else {
            geometry_msgs::PoseStamped min_pose;
            min_pose.header = terrain_map->header;
            min_pose.pose.position.x = terrain_map->min_x;
            min_pose.pose.position.y = terrain_map->min_y;
            min_pose.pose.position.z = terrain_map->z_value;
            min_pose.pose.orientation.w = 1.0;
            try {
                tf_listener_.transformPose(frame_id_, min_pose, min_pose);
            }
            catch (const tf::TransformException &ex) {
                ROS_WARN_THROTTLE(1, " get terrain map acquire---    %s ", ex.what());
                return;
            }
            terrain_map_.frame_id = frame_id_;
            terrain_map_.min_x = min_pose.pose.position.x;
            terrain_map_.min_y = min_pose.pose.position.y;
            terrain_map_.z_value = min_pose.pose.position.z;

            if (!pcl_ros::transformPointCloud(frame_id_, terrain_map_.bottom_points, terrain_map_.bottom_points,
                                              tf_listener_))//Convert to global frame
                return;
        }

        terrain_map_.grid_size = terrain_map->grid_size;
        terrain_map_.grid_width_num = terrain_map->grid_width_num;
        terrain_map_.max_x = terrain_map_.min_x + terrain_map_.grid_size * terrain_map_.grid_width_num;
        terrain_map_.max_y = terrain_map_.min_y + terrain_map_.grid_size * terrain_map_.grid_width_num;


        updateGridMap2D(terrain_map_);
        grid_map_2d_pub_.publish(inflate_map_.generateMapMarkers(inflate_map_.grids_, current_pose_));

    }

    void Map2DManager::updateGridMap2D(const pcl::PointCloud<pcl::PointXYZI> &terrain_cloud,
                                       const pcl::PointCloud<pcl::PointXYZI> &local_cloud) {
        map_2d_update_mutex_.lock();
        map_.clearMap();
        inflate_map_.clearMap();
        pcl::PointXYZI min_p;
        pcl::PointXYZI max_p;
        pcl::getMinMax3D(terrain_cloud, min_p, max_p);
        double x_length = max_p.x - min_p.x;
        double y_length = max_p.y - min_p.y;
        map_.initialize(grid_size_, x_length, y_length, Status2D::Unknown);

        Eigen::Vector2d center_point(current_pose_.position.x, current_pose_.position.y);//center point relative to the global frame
        map_.setMapCenterAndBoundary(center_point);

        for (const auto &point: terrain_cloud) {//relative to the base link plane
            if (map_.isInMapRange2D(point)) {
                if (point.z < -0.1) continue;
                if (point.z - current_pose_.position.z > lower_z_) {//z is the relative height
                    if (point.intensity == 1) {
                        map_.setOccupied(map_.getIndexInMap2D(point));
                    }
                    if (point.intensity == 2) {
                        map_.setEmpty(map_.getIndexInMap2D(point));
                    }
                } else {
                    map_.setFree(map_.getIndexInMap2D(point));
                }
            }
        }

        for (auto &point: local_cloud) { 
            if (map_.isInMapRange2D(point))
                map_.addPointInGrid(map_.getIndexInMap2D(point), point);
        }

        inflate_map_ = map_;
        inflate_map_.inflateGridMap2D(inflate_radius_, inflate_empty_radius_);//Use inflated grid

        is_map_updated_ = true;

        map_2d_update_mutex_.unlock();
    }

    void Map2DManager::updateGridMap2D(const TerrainMap &terrain_map) {
        map_2d_update_mutex_.lock();
        map_.clearMap();
        inflate_map_.clearMap();
        double x_length = terrain_map.max_x - terrain_map.min_x;
        double y_length = terrain_map.max_y - terrain_map.min_y;
        map_.initialize(grid_size_, x_length, y_length, Status2D::Unknown);

        Eigen::Vector2d center_point(current_pose_.position.x, current_pose_.position.y);
        map_.setMapCenterAndBoundary(center_point);

        Point2D grid_center;
        for (int i = 0; i < map_.grid_x_num_; ++i) {
            for (int j = 0; j < map_.grid_y_num_; ++j) {
                grid_center = map_.getGridCenter(i, j);
                if (terrain_map.isInTerrainMap(grid_center)) {
                    if (terrain_map.status[terrain_map.getGridID(grid_center)] == 0)
                        map_.setFree(i, j);
                    if (terrain_map.status[terrain_map.getGridID(grid_center)] == 1)
                        map_.setOccupied(i, j);
                    if (terrain_map.status[terrain_map.getGridID(grid_center)] == 2)
                        map_.setEmpty(i, j);
                    if (terrain_map.status[terrain_map.getGridID(grid_center)] == 3)
                        map_.setUnknown(i, j);

                    map_.addPointInGrid(i, j, terrain_map.bottom_points[terrain_map.getGridID(grid_center)]);

                } else {
                    map_.setUnknown(i, j);
                }
            }
        }

        inflate_map_ = map_;
        inflate_map_.inflateGridMap2D(inflate_radius_, inflate_empty_radius_);//Use inflated grid

        is_map_updated_ = true;

        map_2d_update_mutex_.unlock();
    }

}

