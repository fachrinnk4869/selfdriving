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

run this code if use *realsense2*
```
roslaunch realsense2_camera rs_camera.launch
```

install and run cv_camera if use *brio*
```
sudo apt install ros-noetic-cv-camera
rosrun cv_camera cv_camera_node _cv_cap_prop_frame_width:=640 _cv_cap_prop_frame_height:=480
```

ros this code if use zed2i camera 
```
roslaunch zed_wrapper zed2i.launch
```

to see image from camera topic
```
rosrun image_view image_view image:=/cv_camera/image_raw
```

run source
```
source devel/setup.bash
```

ros launch object detection yolo
```
roslaunch ultralytics_ros tracker.launch debug:=true
```

ros launch lane detection yolopv2 camera
```
roslaunch yolop lane_video.launch
```

ros launch lane detection yolopv2 camera
```
roslaunch yolop lane_camera.launch
```
print hasil fps to csv
```
roslaunch hasil hasil_fps.launch csv_filename_arg:=fps_yolop1.csv
```
