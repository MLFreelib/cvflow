package easy.soc.hacks.frontend.domain

import easy.soc.hacks.frontend.annotation.AllOpen
import lombok.Data
import javax.persistence.*

enum class ProjectionClassType {
    PERSON,
    CAR
}

@AllOpen
@Data
data class ProjectionId(
    val pointId: Long? = null,

    val frameId: Long? = null,

    val batchId: Long? = null,

    val sessionId: String? = null
) : java.io.Serializable

@Table(name = "projections")
@Entity
@IdClass(ProjectionId::class)
@Data
class Projection(
    @Id
    @Column(name = "point_id", nullable = false)
    val pointId: Long = 0,

    @Id
    @Column(name = "frame_id", nullable = false)
    val frameId: Long,

    @Id
    @Column(name = "batch_id", nullable = false)
    val batchId: Long,

    @ManyToOne
    @MapsId("session_id")
    val session: Session? = null,

    @Id
    @Column(name = "session_id", nullable = false)
    private val sessionId: String? = session?.id,

    @Column(name = "x", nullable = false)
    val x: Double,

    @Column(name = "y", nullable = false)
    val y: Double,

    @Column(name = "opacity", nullable = false)
    val opacity: Double,

    @Column(name = "radius", nullable = false)
    val radius: Double,

    @Column(name = "class_type", nullable = false)
    @Enumerated(EnumType.STRING)
    val classType: ProjectionClassType
)

data class ProjectionBatch(
    val batchId: Long,
    val projectionList: List<Projection>,
    val duration: Double
)