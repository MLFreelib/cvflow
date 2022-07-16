package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.Session
import easy.soc.hacks.frontend.domain.SessionStatusType
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.*

@Repository
interface SessionRepository: JpaRepository<Session, String> {

    fun findSessionByStatusNot(status: SessionStatusType): List<Session>

    fun findSessionByStatus(status: SessionStatusType): Optional<Session>

    fun findSessionById(id: String): Optional<Session>
}