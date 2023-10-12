cmake_minimum_required( VERSION 2.8 )
project( directMethod )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3" )

# 添加cmake模块路径
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( G2O )
include_directories( ${G2O_INCLUDE_DIRS} ) 

include_directories( "/usr/include/eigen3" )

find_package( Ceres REQUIRED )
include_directories(${CERES_INCLUDE_DIRS})

find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

set( G2O_LIBS 
    g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension 
)

# add_executable( direct_sparse direct_sparse.cpp )
# target_link_libraries( direct_sparse ${OpenCV_LIBS} ${G2O_LIBS} )

# add_executable( direct_semidense direct_semidense.cpp )
# target_link_libraries( direct_semidense ${OpenCV_LIBS} ${G2O_LIBS} )

# add_executable( q3 q3.cpp )
# target_link_libraries( q3 ${OpenCV_LIBS} ${G2O_LIBS} )

# add_executable( q4_sparse_Num q4_sparse_Num.cpp )
# target_link_libraries( q4_sparse_Num ${OpenCV_LIBS} ${CERES_LIBRARIES} )

add_executable( q4_sparse_Jac q4_sparse_Jac.cpp )
target_link_libraries( q4_sparse_Jac ${OpenCV_LIBS} ${CERES_LIBRARIES} ${Sophus_LIBRARIES} )