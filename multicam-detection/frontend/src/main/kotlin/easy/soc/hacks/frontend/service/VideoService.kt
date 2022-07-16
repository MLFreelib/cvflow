package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.Video
import easy.soc.hacks.frontend.domain.VideoId
import easy.soc.hacks.frontend.repository.VideoRepository
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Service
import org.springframework.web.multipart.MultipartFile
import java.io.InputStream

@Service
class VideoService {
    companion object {
        private val videoIdInputStreamMap = mutableMapOf<VideoId, InputStream>()

        fun getVideoIdInputStream(videoId: VideoId) = videoIdInputStreamMap[videoId]

        fun deleteVideoIdInputStream(videoId: VideoId) {
            videoIdInputStreamMap.remove(videoId)
        }
    }

    @Autowired
    private lateinit var videoRepository: VideoRepository

    fun findVideoByIdAndSessionId(id: Long, sessionId: String) =
        videoRepository.findVideoByIdAndSessionId(id, sessionId)

    fun save(video: Video, multipartFile: MultipartFile? = null): Video {
        val futureId =
            videoRepository.save(
                video.session.id,
                video.name,
                video.streamingType.name,
                video.uri
            )

        val id = futureId.get()

        if (multipartFile != null) {
            videoIdInputStreamMap[VideoId(
                id = id,
                sessionId = video.session.id
            )] = multipartFile.inputStream
        }

        return Video(
            id = id,
            session = video.session,
            name = video.name,
            uri = video.uri,
            calibrationPointList = video.calibrationPointList,
            streamingType = video.streamingType
        )
    }

    fun setCalibration(video: Video) {
        for (calibrationPoint in video.calibrationPointList) {
            videoRepository.setCalibration(
                video.id,
                video.session.id,
                calibrationPoint.id
            )
        }
    }

    fun findVideosBySessionId(sessionId: String) = videoRepository.findVideosBySessionId(sessionId)
}