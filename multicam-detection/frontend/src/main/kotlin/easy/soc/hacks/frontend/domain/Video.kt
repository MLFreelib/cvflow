package easy.soc.hacks.frontend.domain

import easy.soc.hacks.frontend.annotation.AllOpen
import lombok.Data
import javax.persistence.*

enum class StreamingType(
    val value: String
) {
    CAMERA("camera"),
    FILE("file");
}

@AllOpen
@Data
data class VideoId(
    val id: Long? = null,

    val sessionId: String? = null


) : java.io.Serializable {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as VideoId

        if (id != other.id) return false
        if (sessionId != other.sessionId) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id?.hashCode() ?: 0
        result = 31 * result + (sessionId?.hashCode() ?: 0)
        return result
    }
}

@Table(name = "videos")
@Entity
@IdClass(VideoId::class)
@Data
class Video(
    @Id
    @Column(name = "id", nullable = false)
    @GeneratedValue(strategy = GenerationType.IDENTITY, generator = "video_id_seq")
    @SequenceGenerator(name = "videos_id_seq", initialValue = 1)
    val id: Long = 0,

    @ManyToOne
    @MapsId("session_id")
    val session: Session,

    @Id
    @Column(name = "session_id", nullable = false)
    private val sessionId: String = session.id,

    @Column(name = "name", nullable = false)
    val name: String,

    @Column(name = "uri", nullable = true)
    val uri: String? = null,

    @Column(name = "calibration_point_id", nullable = false)
    @OneToMany
    val calibrationPointList: List<CalibrationPoint> = emptyList(),

    @Enumerated(EnumType.STRING)
    @Column(name = "streaming_type", nullable = false)
    val streamingType: StreamingType,
)