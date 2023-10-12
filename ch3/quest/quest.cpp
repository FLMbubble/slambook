#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>

/****************************
* 本程序演示了 Eigen 几何模块的使用方法
****************************/

int main ( int argc, char** argv )
{
    // 用数值对四元数声明并初始化
    Eigen::Quaterniond q1(0.35,0.2,0.3,0.1);
    cout<<"q1 = \n"<<q1.coeffs() <<endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部
    
    Eigen::Quaterniond q2(0.2,-0.5,0.4,-0.1);
    cout<<"q2 = \n"<<q2.coeffs() <<endl;

    Eigen::Vector3d t1 ( 0.3,0.1,0.1 );
    cout<<"t1 ="<<t1<<endl;

    Eigen::Vector3d t2 ( -0.1,0.5,0.3 );
    cout<<"t2 ="<<t2<<endl;

    Eigen::Isometry3d Tw21=Eigen::Isometry3d::Identity();
    Tw21.rotate ( Eigen::AngleAxisd(q1) );
    Tw21.pretranslate ( t1 );
    cout<<"T from world to 1:\n"<<Tw21.matrix()<<endl;

    Eigen::Isometry3d Tw22=Eigen::Isometry3d::Identity();
    Tw22.rotate ( Eigen::AngleAxisd(q2) );
    Tw22.pretranslate ( t2 );
    cout<<"T from world to 2:\n"<<Tw22.matrix()<<endl;

    Eigen::Vector3d p1(0.5,0,0.2);
    Eigen::Vector3d p2=(Tw22*Tw21.inverse())*p1;
    cout<<"p2:"<<p2<<endl;

    return 0;
}
