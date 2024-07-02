import pyzed.sl as sl

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit(1)

    runtime_parameters = sl.RuntimeParameters()
    mat = sl.Mat()

    for i in range(100):
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            print(f"Frame {i} grabbed")

    zed.close()

if __name__ == "__main__":
    main()
