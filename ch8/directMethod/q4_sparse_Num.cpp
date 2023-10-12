#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>


using namespace std;
using namespace cv;
struct Measurement
{
    Measurement ( Eigen::Vector3d p, float g ) : pos_world ( p ), grayscale ( g ) {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = fx*x/z+cx;
    float v = fy*y/z+cy;
    return Eigen::Vector2d ( u,v );
}



struct GRAY_COST
{
    GRAY_COST ( Measurement mea, float fx, float fy, float cx, float cy, cv::Mat* image ) : mea_ ( mea ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image ) {}
    // 残差的计算
    // template <typename T>
    bool operator() (
        const double* const rot,     // 模型参数，旋转向量，3维,是轴角表示
        const double* const trans,   // 模型参数，平移向量，3维
        double* residual ) const     // 残差，有1维
    {
        Eigen::Vector3d x_world_=mea_.pos_world;
        float measure=mea_.grayscale;
        double p_origin[3],p_transformed[3];
        // T p_origin[3],p_transformed[3];
        // p_origin[0]=T(x_world_[0]);
        // p_origin[1]=T(x_world_[1]);
        // p_origin[2]=T(x_world_[2]);
        p_origin[0]=(x_world_[0]);
        p_origin[1]=(x_world_[1]);
        p_origin[2]=(x_world_[2]);
        ceres::AngleAxisRotatePoint(rot,p_origin,p_transformed);//先旋转
        p_transformed[0]=p_transformed[0]+trans[0];
        p_transformed[1]=p_transformed[1]+trans[1];
        p_transformed[2]=p_transformed[2]+trans[2];
        float x=(fx_)*p_transformed[0]/p_transformed[2]+(cx_);
        float y=(fy_)*p_transformed[1]/p_transformed[2]+(cy_);

        // if ( x-T(4)<T(0) || ( x+T(4) ) >T((image_)->cols) || ( y-T(4) ) <T(0) || ( y+T(4) ) >T((image_)->rows) )
        if (( x-4)<0 || ( x+4 ) >(image_)->cols || ( y-4) <0 || ( y+4 ) >(image_)->rows )
        {
            residual[0]= 0.0;//重投影误差较大，下一次不优化
        }
        else
        {
            residual[0] = getPixelValue (x,y) - measure;
        }

        return true;
    }

    float getPixelValue (float x, float y )const
    {
        uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];//int默认向下取整，这里获得指向x,y左上角最近的像素块的指针
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                    ( 1-xx ) * ( 1-yy ) * data[0] +
                    xx* ( 1-yy ) * data[1] +
                    ( 1-xx ) *yy*data[ image_->step ] +
                    xx*yy*data[image_->step+1]
                );
    }

    Measurement mea_;
    float cx_, cy_, fx_, fy_; // Camera intrinsics
    cv::Mat* image_;    // reference image
};



void poseEstimationDirect(const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw);

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    srand ( ( unsigned int ) time ( 0 ) );//随机化初始种子，用系统时间设置
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";

    ifstream fin ( associate_file );

    string rgb_file, depth_file, time_rgb, time_depth;
    cv::Mat color, depth, gray,filt_gray;
    vector<Measurement> measurements;

    // 相机内参
    float cx = 325.5;
    float cy = 253.5;
    float fx = 518.0;
    float fy = 519.0;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    cv::Mat prev_color;
    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    for ( int index=0; index<10; index++ )
    {
        cout<<"*********** loop "<<index<<" ************"<<endl;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread ( path_to_dataset+"/"+rgb_file );
        depth = cv::imread ( path_to_dataset+"/"+depth_file, -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            continue; 
        cv::cvtColor ( color, gray, cv::COLOR_BGR2GRAY );//rgb转灰度图，后续在灰度图上计算梯度
        cv::GaussianBlur(gray,filt_gray,cv::Size(3,3),0,0);
        if ( index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> keypoints;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect ( color, keypoints );
            for ( auto kp:keypoints )
            {
                // 去掉邻近边缘处的点
                if ( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) >color.cols || ( kp.pt.y+20 ) >color.rows )
                    continue;
                ushort d = depth.ptr<ushort> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ];//四舍五入到整数像素坐标对应的深度
                if ( d==0 )
                    continue;
                Eigen::Vector3d p3d = project2Dto3D ( kp.pt.x, kp.pt.y, d, fx, fy, cx, cy, depth_scale );
                float grayscale;
                grayscale = float ( filt_gray.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );//四舍五入到整数像素坐标对应的灰度
                measurements.push_back ( Measurement ( p3d, grayscale ) );
            }
            prev_color = color.clone();
            continue;
        }
        // 使用直接法计算相机运动
        // cout<<"gray img:"<<endl<<gray<<endl;
        // cout<<"filter img:"<<endl<<filt_gray<<endl;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        poseEstimationDirect ( measurements, &gray, K, Tcw );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"direct method costs time: "<<time_used.count() <<" seconds."<<endl;
        cout<<"Tcw="<<Tcw.matrix() <<endl;

        // plot the feature points
        cv::Mat img_show ( color.rows*2, color.cols, CV_8UC3 );//height,width,type
        prev_color.copyTo ( img_show ( cv::Rect ( 0,0,color.cols, color.rows ) ) );//x,y,width,height
        color.copyTo ( img_show ( cv::Rect ( 0,color.rows,color.cols, color.rows ) ) );
        for ( Measurement m:measurements )
        {
            if ( rand() > RAND_MAX/5 )//随机绘制匹配到的像素（否则line太多，影响效果）
                continue;
            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            Eigen::Vector3d p2 = Tcw*m.pos_world;//p{frame i}=Tcw p{frame i-1}
            Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows )//剔除超出边界的像素坐标
                continue;

            float b = 255*float ( rand() ) /RAND_MAX;
            float g = 255*float ( rand() ) /RAND_MAX;
            float r = 255*float ( rand() ) /RAND_MAX;
            cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), 8, cv::Scalar ( b,g,r ), 2 );
            cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +color.rows ), cv::Scalar ( b,g,r ), 1 );
        }
        cv::imshow ( "result", img_show );
        cv::waitKey ( 0 );

    }
    return 0;
}

void poseEstimationDirect(const vector<Measurement>& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw)
{
    ceres::Problem problem;

    Eigen::Vector3d translation=Tcw.translation();
    Eigen::Matrix3d rotation=Tcw.rotation();
    Eigen::AngleAxisd rot_vec(rotation);
    Eigen::Vector3d rot_axis=rot_vec.axis();
    double angle=rot_vec.angle();
    double rot[3]={rot_axis(0)*angle,rot_axis(1)*angle,rot_axis(2)*angle};
    double trans[3]={translation(0),translation(1),translation(2)};
    for ( Measurement m: measurements )
    {
        ceres::CostFunction* cost_function=new ceres::NumericDiffCostFunction<GRAY_COST,ceres::CENTRAL,1, 3,3>(new GRAY_COST(m,K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), gray));
        problem.AddResidualBlock (     // 向问题中添加误差项
        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            cost_function,
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

    rot_axis<<rot[0],rot[1],rot[2];
    angle=sqrt(pow(rot[0],2)+pow(rot[1],2)+pow(rot[2],2));
    rot_axis=rot_axis/angle;
    rot_vec=Eigen::AngleAxisd(angle,rot_axis);
    translation<<trans[0],trans[1],trans[2];
    Tcw.rotate(rot_vec);
    Tcw.pretranslate(translation);
}