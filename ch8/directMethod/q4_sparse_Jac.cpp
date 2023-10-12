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

#include "sophus/so3.h"
#include "sophus/se3.h"


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



class GRAY_COST: public ceres::SizedCostFunction<1,6>
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GRAY_COST ( Measurement mea, float fx, float fy, float cx, float cy, cv::Mat* image ) : mea_ ( mea ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image ) {}
    virtual ~GRAY_COST(){}

    virtual bool Evaluate(double const* const* parameters,double *residual,double**jacobians)const
    {
        Eigen::Map<const Eigen::Matrix<double,6,1>> se3(*parameters);//Eigen::Map使用预分配的内存进行操作
        Sophus::SE3 T=Sophus::SE3::exp(se3);
        Eigen::Vector3d x_world_=mea_.pos_world;
        Eigen::Vector3d x_cam=T*x_world_;
        float measure=mea_.grayscale;
        double u=(fx_)*x_cam[0]/x_cam[2]+(cx_);
        double v=(fy_)*x_cam[1]/x_cam[2]+(cy_);
        if (( u-4)<0 || ( u+4 ) >(image_)->cols || ( v-4) <0 || ( v+4 ) >(image_)->rows )
        {
            residual[0]= 0.0;//重投影误差较大，下一次不优化
            if(jacobians!=NULL)
            {
                if(jacobians[0]!=NULL)
                {
                    Eigen::Map<Eigen::Matrix<double,6,1>> J(jacobians[0]);
                    J<<0,0,0,0,0,0;
                }
            }
        }
        else
        {
            residual[0] = getPixelValue (u,v) - measure;
            if(jacobians!=NULL)
            {
                if(jacobians[0]!=NULL)
                {
                    Eigen::Map<Eigen::Matrix<double,6,1>> J(jacobians[0]);
                    double x=x_cam[0],y=x_cam[1],z=x_cam[2];

                    double invz = 1.0/z;
                    double invz_2 = invz*invz;

                    // jacobian from se3 to u,v
                    // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
                    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

                    jacobian_uv_ksai ( 0,3 ) = - x*y*invz_2 *fx_;
                    jacobian_uv_ksai ( 0,4 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
                    jacobian_uv_ksai ( 0,5 ) = - y*invz *fx_;
                    jacobian_uv_ksai ( 0,0 ) = invz *fx_;
                    jacobian_uv_ksai ( 0,1 ) = 0;
                    jacobian_uv_ksai ( 0,2 ) = -x*invz_2 *fx_;

                    jacobian_uv_ksai ( 1,3 ) = - ( 1+y*y*invz_2 ) *fy_;
                    jacobian_uv_ksai ( 1,4 ) = x*y*invz_2 *fy_;
                    jacobian_uv_ksai ( 1,5 ) = x*invz *fy_;
                    jacobian_uv_ksai ( 1,0 ) = 0;
                    jacobian_uv_ksai ( 1,1 ) = invz *fy_;
                    jacobian_uv_ksai ( 1,2 ) = -y*invz_2 *fy_;

                    // jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
                    // jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
                    // jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
                    // jacobian_uv_ksai ( 0,3 ) = invz *fx_;
                    // jacobian_uv_ksai ( 0,4 ) = 0;
                    // jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

                    // jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
                    // jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
                    // jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
                    // jacobian_uv_ksai ( 1,3 ) = 0;
                    // jacobian_uv_ksai ( 1,4 ) = invz *fy_;
                    // jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;


                    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

                    jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
                    jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

                    J = (jacobian_pixel_uv*jacobian_uv_ksai).transpose();

                }
        }

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
    cv::Mat color, depth, gray;
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
                grayscale = float ( gray.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );//四舍五入到整数像素坐标对应的灰度
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
    Sophus::SE3 SE3_Rt(rotation,translation);
    Eigen::Matrix<double,6,1> se3(SE3_Rt.log());
    double se3d[6]={se3(0,0),se3(1,0),se3(2,0),se3(3,0),se3(4,0),se3(5,0)};
    for ( Measurement m: measurements )
    {
        ceres::CostFunction* cost_function=new GRAY_COST(m,K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), gray);
        problem.AddResidualBlock (     // 向问题中添加误差项
        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            cost_function,
            nullptr,            // 核函数，这里不使用，为空
            se3d
        );
    }

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    se3<<se3d[0],se3d[1],se3d[2],se3d[3],se3d[4],se3d[5];
    Sophus::SE3 T=Sophus::SE3::exp(se3);
    Tcw=T.matrix();
}