#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/media/jetson/home/selfdriving/src/yolop"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/media/jetson/home/selfdriving/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/media/jetson/home/selfdriving/install/lib/python3/dist-packages:/media/jetson/home/selfdriving/build/yolop/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/media/jetson/home/selfdriving/build/yolop" \
    "/usr/bin/python3" \
    "/media/jetson/home/selfdriving/src/yolop/setup.py" \
     \
    build --build-base "/media/jetson/home/selfdriving/build/yolop" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/media/jetson/home/selfdriving/install" --install-scripts="/media/jetson/home/selfdriving/install/bin"
