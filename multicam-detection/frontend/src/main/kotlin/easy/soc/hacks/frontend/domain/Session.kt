package easy.soc.hacks.frontend.domain

import lombok.Data
import java.util.*
import javax.persistence.*

enum class SessionStatusType {
    ACTIVE,
    STREAMING,
    DONE
}

@Table(name = "sessions")
@Entity
@Data
class Session(
    @Id
    @Column(name = "id", nullable = false)
    val id: String = UUID.randomUUID().toString(),

    @Column(name = "start_time", nullable = false)
    val startTime: Date = Date(),

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    val status: SessionStatusType,

    @Enumerated(EnumType.STRING)
    @Column(name = "streaming_type")
    val streamingType: StreamingType
)