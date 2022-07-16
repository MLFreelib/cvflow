package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.Projection
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import javax.transaction.Transactional

@Repository
interface ProjectionRepository : JpaRepository<Projection, Long> {

    @Query(
        """
            insert into projections
            (point_id, batch_id, frame_id, session_id, radius, x, y, opacity, class_type)
             values (:pointId, :batchId, :frameId, :sessionId, :radius, :x, :y, :opacity, :classType)
             returning point_id
        """,
        nativeQuery = true
    )
    @Transactional
    fun save(
        @Param("pointId") pointId: Long,
        @Param("batchId") batchId: Long,
        @Param("frameId") frameId: Long,
        @Param("sessionId") sessionId: String,
        @Param("radius") radius: Double,
        @Param("x") x: Double,
        @Param("y") y: Double,
        @Param("opacity") opacity: Double,
        @Param("classType") classType: String
    ): Long

    fun findProjectionsByBatchIdAndSessionId(
        batchId: Long,
        sessionId: String
    ): List<Projection>
}