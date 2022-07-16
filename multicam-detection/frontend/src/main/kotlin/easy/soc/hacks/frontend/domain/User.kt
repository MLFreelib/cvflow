package easy.soc.hacks.frontend.domain

import easy.soc.hacks.frontend.domain.UserRole.VIEWER
import lombok.Data
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec
import javax.persistence.*
import kotlin.text.Charsets.UTF_8

@Table(name = "users")
@Entity
@Data
class User(
    @Id
    @Column(name = "id", nullable = false)
    @GeneratedValue(strategy = GenerationType.IDENTITY, generator = "users_id_seq")
    @SequenceGenerator(name = "users_id_seq", initialValue = 1)
    val id: Long = 0,

    @Column(name = "login", nullable = false, unique = true)
    val login: String,

    @Enumerated(EnumType.STRING)
    @Column(name = "role", nullable = false)
    val role: UserRole = VIEWER
) {
    companion object {
        fun encryptPassword(password: String, salt: ByteArray): String {
            val spec = PBEKeySpec(password.toCharArray(), salt, 256, 256)
            val secretKeyFactory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA1")

            return String(secretKeyFactory.generateSecret(spec).encoded, UTF_8)
        }
    }

    @Column(name = "salt", nullable = false)
    val salt: ByteArray = run {
        val saltBytes = ByteArray(128)
        val secureRandom = SecureRandom()
        secureRandom.nextBytes(saltBytes)
        saltBytes
    }

    @get:Column(name = "password", nullable = false)
    var password: String? = null
        set(password) {
            field = password?.let { encryptPassword(it, salt) }
        }
}

enum class UserRole {
    VIEWER,
    ADMIN
}