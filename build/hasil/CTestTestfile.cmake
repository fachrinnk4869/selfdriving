# CMake generated Testfile for 
# Source directory: /home/fachri/selfdriving/src/hasil
# Build directory: /home/fachri/selfdriving/build/hasil
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_hasil_roslaunch-check_launch "/home/fachri/selfdriving/build/hasil/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/fachri/selfdriving/build/hasil/test_results/hasil/roslaunch-check_launch.xml" "--return-code" "/home/fachri/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E make_directory /home/fachri/selfdriving/build/hasil/test_results/hasil" "/opt/ros/noetic/share/roslaunch/cmake/../scripts/roslaunch-check -o \"/home/fachri/selfdriving/build/hasil/test_results/hasil/roslaunch-check_launch.xml\" \"/home/fachri/selfdriving/src/hasil/launch\" ")
set_tests_properties(_ctest_hasil_roslaunch-check_launch PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/roslaunch/cmake/roslaunch-extras.cmake;66;catkin_run_tests_target;/home/fachri/selfdriving/src/hasil/CMakeLists.txt;167;roslaunch_add_file_check;/home/fachri/selfdriving/src/hasil/CMakeLists.txt;0;")
subdirs("gtest")
