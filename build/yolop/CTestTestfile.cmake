# CMake generated Testfile for 
# Source directory: /home/fachri/selfdriving/src/yolop
# Build directory: /home/fachri/selfdriving/build/yolop
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_yolop_roslaunch-check_launch "/home/fachri/selfdriving/build/yolop/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/fachri/selfdriving/build/yolop/test_results/yolop/roslaunch-check_launch.xml" "--return-code" "/home/fachri/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E make_directory /home/fachri/selfdriving/build/yolop/test_results/yolop" "/opt/ros/noetic/share/roslaunch/cmake/../scripts/roslaunch-check -o \"/home/fachri/selfdriving/build/yolop/test_results/yolop/roslaunch-check_launch.xml\" \"/home/fachri/selfdriving/src/yolop/launch\" ")
set_tests_properties(_ctest_yolop_roslaunch-check_launch PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/roslaunch/cmake/roslaunch-extras.cmake;66;catkin_run_tests_target;/home/fachri/selfdriving/src/yolop/CMakeLists.txt;174;roslaunch_add_file_check;/home/fachri/selfdriving/src/yolop/CMakeLists.txt;0;")
subdirs("gtest")
