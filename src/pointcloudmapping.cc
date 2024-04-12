#include "pointcloudmapping.h"
#include "PointCloude.h"
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
///获取当前日期
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <sstream>
#include <ctime>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>
int currentloopcount = 0;
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/common/angles.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/io/ply_io.h>

namespace ORB_SLAM2 {

    pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter;
    pcl::PointCloud<pcl::PointXYZRGBA> pcl_local_filter;
    pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud_local_kf;
    pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud_kf;
    PointCloudMapping::PointCloudMapping(double resolution)
    {
        mResolution = resolution;
        mCx = 0;
        mCy = 0;
        mFx = 0;
        mFy = 0;
        mbShutdown = false;
        mbFinish = false;

        voxel.setLeafSize( resolution, resolution, resolution);
        voxelDyna.setLeafSize( resolution, resolution, resolution);
        statistical_filter.setMeanK(50);
        statistical_filterDyna.setMeanK(50);
        statistical_filter.setStddevMulThresh(1.0); // The distance threshold will be equal to: mean + stddev_mult * stddev
        statistical_filterDyna.setStddevMulThresh(1.0); // The distance threshold will be equal to: mean + stddev_mult * stddev

        mPointCloud = boost::make_shared<PointCloud>();  // 用boost::make_shared<>
        mPointCloudDyna = boost::make_shared<PointCloud>();  // 用boost::make_shared<>
        mPointCloudStaic = boost::make_shared<PointCloud>();  // 用boost::make_shared<>
        tem_cloud1 = boost::make_shared<PointCloud>();  // 用boost::make_shared<>
        viewerThread = std::make_shared<std::thread>(&PointCloudMapping::NormalshowPointCloud, this);  // make_unique是c++14的

    }

    PointCloudMapping::~PointCloudMapping()
    {
        viewerThread->join();
    }

    void PointCloudMapping::requestFinish()
    {
        {
            unique_lock<mutex> locker(mKeyFrameMtx);
            mbShutdown = true;
            cout << "稠密点云图构建完成--结束程序" <<endl;
        }
        mKeyFrameUpdatedCond.notify_one();
    }

    bool PointCloudMapping::isFinished()
    {
        return mbFinish;
    }

    void PointCloudMapping::insertKeyFrame(KeyFrame* kf, const cv::Mat& color, const cv::Mat& depth,vector<KeyFrame*> vpKFs,int idk)
    {
        unique_lock<mutex> locker(mKeyFrameMtx);
        midk = idk;
        cout << GREEN <<"receive a keyframe, id = "<<idk<<" 第"<<kf->mnId<<"个"<<endl;
        mvKeyFrames.push(kf);
        keyframes.push_back( kf );
        /// 12.12
        currentvpKFs = vpKFs;
        // cout << "vpKFs 数量 =  " << vpKFs.size();
        ///
        cv::Mat colorImg_ ,depth_;
        mvColorImgs.push( color.clone() );  // clone()函数进行Mat类型的深拷贝，为什幺深拷贝？？
        mvDepthImgs.push( depth.clone() );
        mKeyFrameUpdatedCond.notify_one();
        //cout << "receive a keyframe, id = " << kf->mnId << endl;
    }

    void PointCloudMapping::NormalshowPointCloud()
    {
        pcl::visualization::CloudViewer viewer("Dense pointcloud viewer");
        PointCloude pointcloude;
        size_t N=0;
        while(true) {
            KeyFrame* kf;
            cv::Mat colorImg, depthImg;

            {
                std::unique_lock<std::mutex> locker(mKeyFrameMtx);
                while(mvKeyFrames.empty() && !mbShutdown){  // !mbShutdown为了防止所有关键帧映射点云完成后进入无限等待
                    mKeyFrameUpdatedCond.wait(locker);

                }
                // keyframe is updated

                {
                    unique_lock<mutex> lck( keyframeMutex );
                    N = keyframes.size();
                }
                if(loopbusy || bStop)
                {
                    continue;
                }
                if(lastKeyframeSize == N)
                    cloudbusy = false;
                cloudbusy = true;
                if (!(mvDepthImgs.size() == mvColorImgs.size() && mvKeyFrames.size() == mvColorImgs.size())) {
                    std::cout << RED << "这是不应该出现的情况！" << std::endl;
                    continue;
                }

                if (mbShutdown && mvColorImgs.empty() && mvDepthImgs.empty() && mvKeyFrames.empty()) {
                    break;
                }

                kf = mvKeyFrames.front();
                colorImg = mvColorImgs.front();
                depthImg = mvDepthImgs.front();
                mvKeyFrames.pop();
                mvColorImgs.pop();
                mvDepthImgs.pop();
            }

            if (mCx==0 || mCy==0 || mFx==0 || mFy==0) {
                mCx = kf->cx;
                mCy = kf->cy;
                mFx = kf->fx;
                mFy = kf->fy;
            }


            {
                std::unique_lock<std::mutex> locker(mPointCloudMtx);
                pointcloude.pcE=generatePointCloud(kf,colorImg, depthImg, kf->GetPose());
                pointcloude.pcID = kf->mnId;

                pointcloude.T = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
                pointcloud.push_back(pointcloude);
                //viewer.showCloud(mPointCloudStaic);
                viewer.showCloud(mPointCloud);
                if(pointcloude.pcE->empty())
                    continue;
                pcl_cloud_local_kf = *pointcloude.pcE;
                pcl_cloud_kf = *mPointCloud;
                std::cout << YELLOW << "show point cloud, size=" << mPointCloud->points.size() << std::endl;
                lastKeyframeSize++;
                cloudbusy = false;
            }

        }
        // 存储点云
        string save_path = "./VSLAMRGBD.pcd";
        pcl::io::savePCDFile(save_path, *mPointCloud, true);
        pcl::io::savePLYFile("./vslam.ply", *mPointCloud, true);

        cout << GREEN << "save pcd files to :  " << save_path << endl;

        mbFinish = true;
    }


    pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame *kf,const cv::Mat& imRGB, const cv::Mat& imD, const cv::Mat& pose)
    {
//    std::cout << "Converting image: " << nId;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        PointCloud::Ptr current(new PointCloud);
        PointCloud::Ptr view_current(new PointCloud);
        PointCloud::Ptr currentDyna(new PointCloud);
        ///正常稠密图 地图可回环
        for(size_t v = 0; v < imRGB.rows ; v+=5){
            for(size_t u = 0; u < imRGB.cols ; u+=5){
                cv::Point2i pt(u, v);
                float d = imD.ptr<float>(v)[u];
                if(d <0.01 || d>10){ // 深度值为0 表示测量失败
                    continue;
                }
                bool IsDynamic = false;
                for (auto area: kf->mvDynamicArea)
                    if (area.contains(pt)) IsDynamic = true;
                if(IsDynamic)
                {
                    PointT p_mvDynamic;
                    p_mvDynamic.z = d;
                    p_mvDynamic.x = (u - kf->cx) * p_mvDynamic.z / kf->fx;
                    p_mvDynamic.y = (v - kf->cy) * p_mvDynamic.z / kf->fy;
                    p_mvDynamic.b = 255;
                    p_mvDynamic.g = 255;
                    p_mvDynamic.r = 255;
                    p_mvDynamic.a = 1;
                    currentDyna->points.push_back(p_mvDynamic);
                }
                else
                {
                    PointT p;
                    p.z = d;
                    p.x = (u - kf->cx) * p.z / kf->fx;
                    p.y = (v - kf->cy) * p.z / kf->fy;

                    //ros下运行
                    p.r = imRGB.ptr<uchar>(v)[u * 3];
                    p.g = imRGB.ptr<uchar>(v)[u * 3 + 1];
                    p.b= imRGB.ptr<uchar>(v)[u * 3 + 2];

                    // p.b = imRGB.ptr<uchar>(v)[u * 3];
                    // p.g = imRGB.ptr<uchar>(v)[u * 3 + 1];
                    // p.r = imRGB.ptr<uchar>(v)[u * 3 + 2];
                    current->points.push_back(p);
                    view_current->points.push_back(p);
                }
            }
        }

        Eigen::Isometry3d T = Converter::toSE3Quat( pose );
        PointCloud::Ptr tmp(new PointCloud);
        PointCloud::Ptr tmpDyna(new PointCloud);
        // tmp为转换到世界坐标系下的点云
        pcl::transformPointCloud(*currentDyna, *tmpDyna, T.inverse().matrix());
        pcl::transformPointCloud(*current, *tmp, T.inverse().matrix());

        // depth filter and statistical removal，离群点剔除
        statistical_filter.setInputCloud(tmp);
        statistical_filterDyna.setInputCloud(tmpDyna);
        statistical_filter.filter(*current);
        statistical_filterDyna.filter(*currentDyna);
        (*mPointCloud) += *current;
        (*mPointCloudDyna) += *currentDyna;

        pcl::transformPointCloud(*mPointCloudDyna, *tmp, T.inverse().matrix());
        pcl::transformPointCloud(*mPointCloud, *tmpDyna, T.inverse().matrix());
        // 加入新的点云后，对整个点云进行体素滤波
        voxel.setInputCloud(mPointCloudDyna);
        voxelDyna.setInputCloud(mPointCloud);
        voxel.filter(*tmp);
        voxelDyna.filter(*tmpDyna);
        mPointCloudDyna->swap(*tmp);
        mPointCloud->swap(*tmpDyna);
        mPointCloudDyna->is_dense = false;
        mPointCloud->is_dense = false;

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        // std::cout << " Cost = " << t << std::endl;

        view_mPointCloud = mPointCloudDyna;


        ///剔除完全打开下面代码  可以自己调整距离阈值 constexpr float distance_threshold = 0.005; // 距离阈值    把显示的点云换成mPointCloudStaic  184行
    //    if(true) {
    //        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr kdtree(
    //                new pcl::search::KdTree<pcl::PointXYZRGBA>);
    //        if (mPointCloudDyna->size() > 100) {
    //            kdtree->setInputCloud(mPointCloudDyna->makeShared());
    //            for (const auto &point: mPointCloud->points) {
    //                std::vector<int> nearest_indices;
    //                std::vector<float> nearest_distances;
    //                constexpr int k = 1; // 搜索最近的一个点
    //                kdtree->nearestKSearch(point, k, nearest_indices, nearest_distances);

    //                // 如果最近邻点的距离小于某个阈值，表示该点在点云B中存在，可以忽略
    //                constexpr float distance_threshold = 0.01; // 距离阈值
    //                if (nearest_distances[0] > distance_threshold) {
    //                    mPointCloudStaic->points.push_back(point);
    //                }
    //            }
    //        }
    //    }

        return view_current;
    }

    void PointCloudMapping::updatecloud()
    {
        if(!cloudbusy)
        {
            {
                std::unique_lock<std::mutex> locker(mPointCloudMtx);
                loopbusy = true;
                chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
                cout <<GREEN<< "#--------------------------------------------#" << endl;
                cout<<GREEN<<"暂停稠密点云构建，检测到回环,正在更新点云ing"<<endl;
                Five_pointed_star();
                PointCloud::Ptr tmp1(new PointCloud);
                for (int i=0;i<currentvpKFs.size();i++)
                {
                    for (int j=0;j<pointcloud.size();j++)
                    {
                        if(pointcloud[j].pcID==currentvpKFs[i]->mnId)
                        {
                            Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose());
                            PointCloud::Ptr cloud(new PointCloud);
                            pcl::transformPointCloud( *pointcloud[j].pcE, *cloud, T.inverse().matrix());
                            ///
                            *tmp1 +=*cloud;

                            //cout <<GREEN<<"PointCloud 第"<<j<<"帧 与 vpKFs 第"<<i<<"帧匹配"<<endl;
                            continue;
                        }
                    }
                }
                chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
                chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
                cout <<GREEN<< "#--------------------------------------------#" << endl;
                cout << endl;
                cout <<GREEN<<"      检测到       " << loopcount+1<< "     次回环 "<<endl;
                cout << endl;
                cout <<GREEN<<"      调整了       " << currentvpKFs.size() << "    关键帧位姿  "<<endl;
                cout << endl;
                cout <<GREEN<<"      共计调整了   " << mPointCloud->points.size() << "   个点云位姿  "<<endl;
                cout << endl;
                cout <<GREEN<<"      耗时         " << time_used.count()*1000 << "   ms  "<<endl;
                cout << endl;
                cout <<GREEN<< "#--------------------------------------------#" << endl;
                PointCloud::Ptr tmp2(new PointCloud());
                voxel.setInputCloud( tmp1 );
                voxel.filter( *tmp2 );

                if(mPointCloud->points.size() < 10000)
                {
                    cout <<GREEN << "没有检测到回环！"  << endl;
                }
                else
                {
                    mPointCloud->points.clear();
                    mPointCloud->swap( *tmp2 );
                    view_mPointCloud = mPointCloud;
                    loopbusy = false;
                    Five_pointed_star();
                    cout << GREEN<<"地图调整完毕，恢复稠密点云构建！"<<endl;
                    cout <<GREEN<< "#--------------------------------------------#" << endl;
                    //cloudbusy = true;
                    loopcount++;
                    ///存储点云
                    string save_path = "./VSLAMRGBD.pcd";
                    pcl::io::savePCDFile(save_path, *mPointCloud);
                    cout << "save pcd files to :  " << save_path << endl;
                }
//            mPointCloud->points.clear();
//            mPointCloud->swap( *tmp2 );
//            view_mPointCloud = mPointCloud;
//            cout <<GREEN << "地图调整完毕！ "  << endl;
//            loopbusy = false;
//            //cloudbusy = true;
//            loopcount++;

            }

        }

    }

    void PointCloudMapping::getGlobalCloudMap(PointCloud::Ptr &outputMap)
    {
        std::unique_lock<std::mutex> locker(mPointCloudMtx);
        outputMap = mPointCloud;
    }
    void PointCloudMapping::Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out)
    {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered;
        Eigen::Matrix4f m;

        m<< 0,0,1,0,
            -1,0,0,0,
            0,-1,0,0;
        Eigen::Affine3f transform(m);
        pcl::transformPointCloud (source, out, transform);
    }
    void PointCloudMapping::Five_pointed_star()
    {
        int i,j;
        for(i=1;i<=6;i++)  //处理上层顶角
        {
            for(j=0;j<19-i;j++)  //输出空格
                printf(" ");
            for(j=0;j<2*i-1;j++)  //输出*号
                printf("*");
            printf("\n");
        }
        for(i=6;i>=3;i--)  //处理中层
        {
            for(j=0;j<6-i;j++)
                printf("   ");
            for(j=0;j<6*i;j++)
                printf("*");
            printf("\n");
        }
        for(i=1;i<=6;i++)  //处理下层
        {   for(int j=0 ; j<12-i; j++ )
                printf(" ");
            for(int j=0 ; j<12+2*i ; j++) {
                if(i<=2)
                    printf("*");
                else {
                    if(j>=14-2*i && j<=4*i-3)
                        printf(" ");
                    else
                        printf("*");
                }
            }
            printf("\n");
        }
    }
}
