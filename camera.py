import cv2

def open_camera():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        return None
    return cam

def read_frame(cam):
        ret, frame = cam.read()

        if ret == False:
            print("Error: Could not read frame.")
            return False, None

        return True,frame

def close_camera(cam):
    if cam is not None:
        cam.release()
    cv2.destroyAllWindows()

