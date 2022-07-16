package easy.soc.hacks.frontend.domain

import lombok.Data
import javax.persistence.*

@Table(name = "video_screenshots")
@Entity
@Data
class VideoScreenshot(
    @OneToOne
    @MapsId("video_id")
    val video: Video,

    @Id
    @Column(name = "video_id", nullable = false)
    @GeneratedValue(strategy = GenerationType.IDENTITY, generator = "video_screenshots_video_id_seq")
    @SequenceGenerator(name = "video_screenshots_video_id_seq", initialValue = 1)
    private val videoId: Long = video.id,

    @Column(name = "data", nullable = false)
    val data: ByteArray
)