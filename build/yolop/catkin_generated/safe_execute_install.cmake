execute_process(COMMAND "/media/jetson/home/selfdriving/build/yolop/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/media/jetson/home/selfdriving/build/yolop/catkin_generated/python_distutils_install.sh) returned error code ")
endif()