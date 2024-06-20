# CMake generated Testfile for 
# Source directory: /media/jetson/home/selfdriving/src/yolop
# Build directory: /media/jetson/home/selfdriving/build/yolop
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_yolop_roslaunch-check_launch "/media/jetson/home/selfdriving/build/yolop/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/media/jetson/home/selfdriving/build/yolop/test_results/yolop/roslaunch-check_launch.xml" "--return-code" "/usr/bin/cmake -E make_directory /media/jetson/home/selfdriving/build/yolop/test_results/yolop" "/opt/ros/noetic/share/roslaunch/cmake/../scripts/roslaunch-check -o \"/media/jetson/home/selfdriving/build/yolop/test_results/yolop/roslaunch-check_launch.xml\" \"/media/jetson/home/selfdriving/src/yolop/launch\" ")
set_tests_properties(_ctest_yolop_roslaunch-check_launch PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/roslaunch/cmake/roslaunch-extras.cmake;66;catkin_run_tests_target;/media/jetson/home/selfdriving/src/yolop/CMakeLists.txt;174;roslaunch_add_file_check;/media/jetson/home/selfdriving/src/yolop/CMakeLists.txt;0;")
subdirs("gtest")
