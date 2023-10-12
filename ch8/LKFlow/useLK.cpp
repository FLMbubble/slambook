#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
using namespace std; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: useLK path_to_dataset"<<endl;
        return 1;
    }
    string path_to_dataset = argv[1];
    string associate_file = path_to_dataset + "/associate.txt";
    cout<<"run:"<<path_to_dataset<<endl;

    ifstream fin( associate_file );//对象创建成功时返回非空值
    if ( !fin ) 
    {
        cerr<<"I cann't find associate.txt!"<<endl;
        return 1;
    }
    //fin.eof()，若返回false，说明没读到文件尾部，否则返回true
    cout<<"prepare reading!"<<endl;
    
    string rgb_file, depth_file, time_rgb, time_depth;
    // char rf[20],df[20],tr[20],td[20];
    list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    cv::Mat color, depth, last_color;
    
    for ( int index=0; index<100; index++ )
    {
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;//遇见空格时会自动分隔，将用空格隔开的内容依次赋值给后面跟的四个变量
        color = cv::imread( path_to_dataset+"/"+rgb_file );//flag默认值为1，读8bit3通道
        depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );//按图像原样返回加载的图像
        if (index ==0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );//接受图像输入color，将FAST特征点结果存储在kps中
            for ( auto kp:kps )
                keypoints.push_back( kp.pt );
            last_color = color;
            continue;
        }
        //忽略颜色或深度信息不全的帧
        if ( color.data==nullptr || depth.data==nullptr )
            continue;
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; 
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);
        vector<unsigned char> status;
        vector<float> error; 
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉
        int i=0; 
        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
        {
            if ( status[i] == 0 )//未追踪到关键点
            {
                iter = keypoints.erase(iter);//list_name.erase(iterator position)，删除position处的元素,同时返回下一个迭代器
                //iterator list_name.erase(iterator first, iterator last)，删除从first到last的元素
                continue;

                //advance(iter,num)，在原始迭代器的位置进行移动，num为负表示前移，正为后移
                //next(iter,num)，类似advance，但返回一个新的容器
                //inserter(container,iter),在iter位置前插入container，返回iter原先指向的位置
            }
            *iter = next_keypoints[i];//更新关键点的位置信息
            iter++;
        }
        cout<<"tracked keypoints: "<<keypoints.size()<<endl;
        if (keypoints.size() == 0)
        {
            cout<<"all keypoints are lost."<<endl;
            break; 
        }
        // 画出 keypoints
        cv::Mat img_show = color.clone();
        for ( auto kp:keypoints )
            cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
        cv::imshow("corners", img_show);
        cv::waitKey(0);
        last_color = color;
    }
    return 0;
}