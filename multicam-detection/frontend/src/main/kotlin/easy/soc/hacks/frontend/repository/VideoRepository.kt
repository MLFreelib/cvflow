package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.Video
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Repository
import java.util.*
import java.util.concurrent.Future
import javax.transaction.Transactional


@Repository
interface VideoRepository : JpaRepository<Video, Long> {

    @Query(
        """
            select * from  videos
            where id = :id and
            session_id = :sessionId
        """,
        nativeQuery = true
    )
    fun findVideoByIdAndSessionId(
        @Param("id") id: Long,
        @Param("sessionId") sessionId: String
    ): Optional<Video>

    fun findVideosBySessionId(sessionId: String): List<Video>

    @Async
    @Query(
        """
            insert into videos
            (session_id, name, streaming_type, uri)
            values (:sessionId, :name, :streamingType, :uri)
            returning id
        """,
        nativeQuery = true
    )
    @Transactional
    fun save(
        @Param("sessionId") sessionId: String,
        @Param("name") name: String,
        @Param("streamingType") streamingType: String,
        @Param("uri") uri: String?,
    ): Future<Long>

    @Query(
        """
            insert into videos_calibration_point_list
            (video_id, video_session_id, calibration_point_list_id)
            values (:videoId, :sessionId, :calibrationPointId)
        """,
        nativeQuery = true
    )
    @Modifying
    @Transactional
    fun setCalibration(
        @Param("videoId") videoId: Long,
        @Param("sessionId") sessionId: String,
        @Param("calibrationPointId") calibrationPointId: Long
    )
}