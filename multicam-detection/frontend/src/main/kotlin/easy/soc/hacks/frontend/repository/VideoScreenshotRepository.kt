package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.Video
import easy.soc.hacks.frontend.domain.VideoScreenshot
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.util.*
import javax.transaction.Transactional

@Repository
interface VideoScreenshotRepository : JpaRepository<VideoScreenshot, Long> {
    @Query(
        """
            insert into video_screenshots
            (video_id, data, video_session_id)
                select :videoId, :data, :sessionId
                where not exists(
                    select *
                    from video_screenshots as vs
                    where vs.video_id = :videoId
                    and vs.video_session_id = :sessionId
                );
                    
            update video_screenshots set
            data = :data
            where 
                video_id = :videoId and
                video_session_id = :sessionId
        """,
        nativeQuery = true
    )
    @Modifying
    @Transactional
    fun save(
        @Param("videoId") videoId: Long,
        @Param("data") data: ByteArray,
        @Param("sessionId") sessionId: String
    )

    fun findVideoScreenshotByVideo(video: Video): Optional<VideoScreenshot>
}