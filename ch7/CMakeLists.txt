cmake_minimum_required( VERSION 2.8 )
project( vo1 )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )


include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
    "/usr/include/eigen3/"
)

# add_executable( feature_extraction feature_extraction.cpp  )
# target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# # add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
# target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

# # # add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2 
# add_executable( triangulation triangulation.cpp )
# target_link_libraries( triangulation ${OpenCV_LIBS} )

# add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
# target_link_libraries( pose_estimation_3d2d 
#    ${OpenCV_LIBS}
#    ${CSPARSE_LIBRARY}
#    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_solver_csparse g2o_types_slam3d
# )

# add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
# target_link_libraries( pose_estimation_3d3d 
#    ${OpenCV_LIBS}
#    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
#    ${CSPARSE_LIBRARY}
# )

# add_executable( pose_q6 pose_q6.cpp )
# target_link_libraries( pose_q6 
#    ${OpenCV_LIBS}
#    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_solver_csparse g2o_types_slam3d 
#    ${CSPARSE_LIBRARY}
# )

# add_executable( pose_q7 pose_q7.cpp )
# target_link_libraries( pose_q7 
#    ${OpenCV_LIBS}
#    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_types_slam3d
#    ${CSPARSE_LIBRARY}
# )

# add_executable( pose_q10 pose_q10.cpp )
# target_link_libraries( pose_q10 
#    ${OpenCV_LIBS}
#    ${CERES_LIBRARIES}
# )

add_executable( pose_q10ICP pose_q10ICP.cpp )
target_link_libraries( pose_q10ICP 
   ${OpenCV_LIBS}
   ${CERES_LIBRARIES}
)