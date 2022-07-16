package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.User
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.*

@Repository
interface UserRepository : JpaRepository<User, Long> {
    fun findUserByLogin(login: String): Optional<User>
}