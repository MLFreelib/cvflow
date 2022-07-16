package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.Video
import easy.soc.hacks.frontend.domain.VideoFragment
import easy.soc.hacks.frontend.repository.VideoFragmentRepository
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Service
import org.springframework.util.ResourceUtils
import java.util.*

@Service
class VideoFragmentService {
    @Autowired
    private lateinit var videoFragmentRepository: VideoFragmentRepository

    fun save(videoFragment: VideoFragment): VideoFragment {
        videoFragmentRepository.save(
            id = videoFragment.id,
            sessionId = videoFragment.video.session.id,
            videoId = videoFragment.video.id,
            data = videoFragment.data,
            duration = videoFragment.duration
        )

        return videoFragment
    }

    fun findVideoFragmentsByVideoIdAndSessionIdAndVideoFragmentIdRange(
        videoId: Long,
        sessionId: String,
        toId: Long
    ): Pair<Long, ByteArray> {
        val ids = mutableListOf<Long>()
        val durations = mutableListOf<Double>()

        videoFragmentRepository.findVideoFragmentsByVideoIdAndSessionIdAndVideoFragmentIdRange(
            videoId,
            sessionId,
            toId
        ).apply {
            ids.addAll(this.map { it.id })
            durations.addAll(this.map { it.duration })
        }

        if (ids.size < 2 && toId > 1) {
            return findVideoFragmentsByVideoIdAndSessionIdAndVideoFragmentIdRange(
                videoId,
                sessionId,
                toId - 1
            )
        }

        val manifestTextStringBuffer = StringBuilder()

        manifestTextStringBuffer.append("#EXTM3U\n")
        manifestTextStringBuffer.append("#EXT-X-VERSION:3\n")
        manifestTextStringBuffer.append("#EXT-X-MEDIA-SEQUENCE:${ids[0]}\n")
        manifestTextStringBuffer.append("#EXT-X-TARGETDURATION:${durations.maxOf { it }}\n")
        for (i in 0 until ids.size) {
            manifestTextStringBuffer.append(
                "#EXTINF:${durations[i]},\n/api/v1/video/fragment?id=$videoId&fragment=${ids[i]}\n"
            )
        }

        return Pair(ids[0], manifestTextStringBuffer.toString().toByteArray())
    }

    fun getManifestByVideoIdAndSessionIdAndNextBatchId(
        videoId: Long,
        sessionId: String,
        nextBatchId: Long
    ) = findVideoFragmentsByVideoIdAndSessionIdAndVideoFragmentIdRange(
        videoId,
        sessionId,
        nextBatchId
    )

    fun findVideoFragment(id: Long, video: Video): Optional<VideoFragment> {
        return when (id) {
            0L ->
                Optional.of(
                    VideoFragment(
                        id = 0,
                        video = video,
                        duration = 0.0,
                        data = ResourceUtils.getFile("classpath:media/video/zeroFragment.ts").readBytes()
                    )
                )

            else ->
                videoFragmentRepository.findVideoFragmentByIdAndVideo(id, video)
        }
    }


    fun getMaxVideoFragmentIdBySessionId(sessionId: String) =
        videoFragmentRepository.getMaxVideoFragmentIdBySessionId(sessionId)

    fun checkExistsVideoFragmentBySessionIdAndVideoId(
        sessionId: String,
        videoId: Long
    ) = videoFragmentRepository.countBySessionIdAndVideoId(sessionId, videoId) >= 2

    fun getDurationBySessionIdAndId(
        sessionId: String,
        id: Long
    ) = videoFragmentRepository.getDurationBySessionIdAndId(sessionId, id)
}

