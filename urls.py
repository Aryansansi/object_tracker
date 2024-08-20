RTSP_URLS = "rtsp://admin:admin123@192.168.1.153:554/avstream/channel=1/stream=0.sdp"

def get_camera_url(index):
    return RTSP_URLS[index]
