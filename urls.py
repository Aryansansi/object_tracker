RTSP_URLS = "rtsp://username:password@IP-address/streamname/channel-number"

def get_camera_url(index):
    return RTSP_URLS[index]
