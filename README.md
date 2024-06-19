install ros-realsense2-camera 
```
sudo apt-get install ros-noetic-realsense2-camera
```
build catkin 
```
catkin build
```

if you are use anaconda (deactivate anaconda) switch it to python native or use this code
```
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3
```

ros run cv_camera if use *brio*
```
rosrun cv_camera cv_camera_node _cv_cap_prop_frame_width:=640 _cv_cap_prop_frame_height:=480
```

run source
```
source devel/setup.bash
```

ros launch ros tracker
```
roslaunch ultralytics_ros tracker.launch debug:=true
```

