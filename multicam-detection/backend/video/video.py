class Video:
    def __init__(self, video_id, session_id, uri, streaming_type, camera):
        self.video_id = video_id
        self.session_id = session_id
        self.camera = camera
        self.uri = uri
        self.streaming_type = streaming_type
