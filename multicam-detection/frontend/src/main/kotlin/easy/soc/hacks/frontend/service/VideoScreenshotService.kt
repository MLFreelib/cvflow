package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.Video
import easy.soc.hacks.frontend.domain.VideoScreenshot
import easy.soc.hacks.frontend.repository.VideoScreenshotRepository
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Service

@Service
class VideoScreenshotService {
    @Autowired
    private lateinit var videoScreenshotRepository: VideoScreenshotRepository

    fun save(videoScreenshot: VideoScreenshot): VideoScreenshot {
        videoScreenshotRepository.save(
            videoId = videoScreenshot.video.id,
            data = videoScreenshot.data,
            sessionId = videoScreenshot.video.session.id
        )

        return findVideoScreenshotByVideo(videoScreenshot.video).get()
    }

    fun findVideoScreenshotByVideo(video: Video) = videoScreenshotRepository.findVideoScreenshotByVideo(video)
}