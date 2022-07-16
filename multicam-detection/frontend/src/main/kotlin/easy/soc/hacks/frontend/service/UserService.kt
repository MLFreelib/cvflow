package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.User
import easy.soc.hacks.frontend.domain.UserRole.ADMIN
import easy.soc.hacks.frontend.repository.UserRepository
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.stereotype.Service
import java.util.*

@Service
class UserService {
    @Autowired
    private lateinit var userRepository: UserRepository

    @EventListener(ApplicationReadyEvent::class)
    fun ensureAdminExists() {
        val adminUser = findAdmin()

        if (adminUser.isEmpty) {
            saveAdmin()
        }
    }

    fun save(user: User): Optional<User> {
        if (user.role == ADMIN && findAdmin().isPresent) {
            return Optional.empty()
        }

        return try {
            Optional.of(userRepository.save(user))
        } catch (e: IllegalArgumentException) {
            Optional.empty()
        }
    }

    fun login(login: String, password: String): Optional<User> {
        val foundUser = userRepository.findUserByLogin(login).orElseGet { null }

        return if (foundUser?.password == User.encryptPassword(password, foundUser.salt)) {
            Optional.of(foundUser)
        } else {
            Optional.empty()
        }
    }

    fun findAdmin() = userRepository.findUserByLogin("admin")

    fun saveAdmin() = userRepository.save(User(
        login = "admin",
        role = ADMIN
    ).apply {
        password = "admin"
    })
}