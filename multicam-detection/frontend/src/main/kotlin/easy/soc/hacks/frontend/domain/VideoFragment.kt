package easy.soc.hacks.frontend.domain

import easy.soc.hacks.frontend.annotation.AllOpen
import lombok.Data
import javax.persistence.*


@AllOpen
@Data
data class VideoFragmentId(
    val id: Long? = null,

    val videoId: Long? = null,

    val sessionId: String? = null
) : java.io.Serializable, Comparable<VideoFragmentId> {
    override fun compareTo(other: VideoFragmentId): Int {
        if (sessionId == other.sessionId) {
            if (videoId == other.videoId) {
                return (id ?: -1).compareTo(other.id ?: -1)
            }

            return (videoId ?: -1).compareTo(other.videoId ?: -1)
        }

        return (sessionId ?: "").compareTo(other.sessionId ?: "")
    }
}

@Table(name = "video_fragments")
@Entity
@IdClass(VideoFragmentId::class)
@Data
class VideoFragment(
    @Id
    @Column(name = "id", nullable = false)
    val id: Long,

    @ManyToOne
    @JoinColumns(
        JoinColumn(name = "video_id", referencedColumnName = "id"),
        JoinColumn(name = "session_id", referencedColumnName = "session_id")
    )
    val video: Video,

    @Id
    @Column(name = "video_id", nullable = false, insertable = false, updatable = false)
    private val videoId: Long = video.id,

    @Id
    @Column(name = "session_id", nullable = false, insertable = false, updatable = false)
    private val sessionId: String = video.session.id,

    @Column(name = "duration", nullable = false)
    val duration: Double,

    @Column(name = "data", nullable = false)
    val data: ByteArray,
)