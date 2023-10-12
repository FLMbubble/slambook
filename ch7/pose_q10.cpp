#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>

using namespace std;
using namespace cv;
// 代价函数的计算模型
void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

struct PnP_COST
{
    PnP_COST ( Point3f xyz, Point2f uv ) : _xyz ( xyz ), _uv ( uv ) {}
    // 残差的计算
    template <typename T>
    bool operator() (
        const T* const rot,     // 模型参数，旋转向量，3维
        const T* const trans,   // 模型参数，平移向量，3维
        T* residual ) const     // 残差，有2维
    {
        T p_origin[3],p_transformed[3];
        p_origin[0]=T(_xyz.x);
        p_origin[1]=T(_xyz.y);
        p_origin[2]=T(_xyz.z);
        ceres::AngleAxisRotatePoint(rot,p_origin,p_transformed);//先旋转
        p_transformed[0]=p_transformed[0]+trans[0];
        p_transformed[1]=p_transformed[1]+trans[1];
        p_transformed[2]=p_transformed[2]+trans[2];
        double fx=520.9,cx=325.1,fy=521.0,cy=249.7;
        T u_pred=fx*p_transformed[0]/p_transformed[2]+cx;
        T v_pred=fy*p_transformed[1]/p_transformed[2]+cy;

        residual[0] = T ( _uv.x ) -u_pred;
        residual[1] =T(_uv.y)-v_pred;
        return true;
    }
    Point3f _xyz;
    Point2f _uv;    // x,y数据
};

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    double rot[3] = {0,0,0};            // abc参数的估计值
    double trans[3]={0,0,0};
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        //Mat.ptr<type>(row)[col]，访问Mat第row+1行第col+1列元素
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;//乘上深度图和实际深度的一个比例系数
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );//第一张深度图的特征点的位置作为3D特征点位置
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    // 构建最小二乘问题
    ceres::Problem problem;
    for ( int i=0; i<pts_3d.size(); i++ )
    {
        problem.AddResidualBlock (     // 向问题中添加误差项
        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            new ceres::AutoDiffCostFunction<PnP_COST, 2, 3,3> ( // 模板参数依次为仿函数（functor）类型CostFunctor，残差维数residualDim和参数维数paramDim，接受参数类型为仿函数指针CostFunctor*
                new PnP_COST ( pts_3d[i], pts_2d[i] )
            ),
            nullptr,            // 核函数，这里不使用，为空
            rot,                 // 待估计参数
            trans
        );
    }

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    Mat R;
    Mat rotation=(Mat_<double>(3,1)<<rot[0],rot[1],rot[2]);
    cv::Rodrigues(rotation,R);
    Mat t=(Mat_<double>(3,1)<<trans[0],trans[1],trans[2]);
    cout<<"R = "<<endl<<R<<endl;
    cout<<"t = "<<endl<<t<<endl;

    return 0;
}





void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}
