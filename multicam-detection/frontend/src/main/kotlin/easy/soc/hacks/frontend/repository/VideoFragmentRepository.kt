package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.Video
import easy.soc.hacks.frontend.domain.VideoFragment
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.util.*
import javax.transaction.Transactional

@Repository
interface VideoFragmentRepository : JpaRepository<VideoFragment, Long> {

    @Query(
        """
            insert into video_fragments
            (id, session_id, video_id, data, duration)
            values (:id, :sessionId, :videoId, :data, :duration)
        """,
        nativeQuery = true
    )
    @Modifying
    @Transactional
    fun save(
        @Param("id") id: Long,
        @Param("sessionId") sessionId: String,
        @Param("videoId") videoId: Long,
        @Param("data") data: ByteArray,
        @Param("duration") duration: Double
    )

    fun findVideoFragmentByIdAndVideo(id: Long, video: Video): Optional<VideoFragment>

    @Query(
        """
            select max(id) from video_fragments
            where video_fragments.session_id = :sessionId
        """,
        nativeQuery = true
    )
    fun getMaxVideoFragmentIdBySessionId(
        @Param("sessionId") sessionId: String
    ): Optional<Long>

    @Query(
        """
            select * from video_fragments
            where video_fragments.video_id = :videoId
            and video_fragments.session_id = :sessionId
            and (
                video_fragments.id = :toId or 
                video_fragments.id = :toId - 1
            )
            order by video_fragments.id desc 
        """,
        nativeQuery = true
    )
    fun findVideoFragmentsByVideoIdAndSessionIdAndVideoFragmentIdRange(
        @Param("videoId") videoId: Long,
        @Param("sessionId") sessionId: String,
        @Param("toId") toId: Long
    ): List<VideoFragment>

    @Query(
        """
            select count(*) from video_fragments
            where session_id = :sessionId
            and video_id = :videoId 
        """,
        nativeQuery = true
    )
    fun countBySessionIdAndVideoId(sessionId: String, videoId: Long): Long

    @Query(
        """
            select duration from video_fragments
            where session_id = :sessionId
            and id = :id
            fetch first row only 
        """,
        nativeQuery = true
    )
    fun getDurationBySessionIdAndId(
        @Param("sessionId") sessionId: String,
        @Param("id") id: Long
    ): Double
}