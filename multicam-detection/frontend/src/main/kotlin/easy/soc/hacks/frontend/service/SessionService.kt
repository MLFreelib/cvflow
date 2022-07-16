package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.Session
import easy.soc.hacks.frontend.domain.SessionStatusType.*
import easy.soc.hacks.frontend.repository.SessionRepository
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.stereotype.Service

@Service
class SessionService {
    @Autowired
    private lateinit var sessionRepository: SessionRepository

    @EventListener(ApplicationReadyEvent::class)
    fun ensureCloseActiveSessions() {
        closeActiveSessions()
    }

    fun save(session: Session) = sessionRepository.save(session)

    private fun findNotDoneSession() = sessionRepository.findSessionByStatusNot(DONE)

    private fun closeActiveSessions() {
        findNotDoneSession().map {
            save(
                Session(
                    id = it.id,
                    startTime = it.startTime,
                    status = DONE,
                    streamingType = it.streamingType
                )
            )
        }
    }

    fun getActiveSession() = sessionRepository.findSessionByStatus(ACTIVE)

    fun getStreamingSession() = sessionRepository.findSessionByStatus(STREAMING)

    fun getActiveOrStreamingSession() =
        getActiveSession().or { getStreamingSession() }

    fun findSessionById(id: String) = sessionRepository.findSessionById(id)
}