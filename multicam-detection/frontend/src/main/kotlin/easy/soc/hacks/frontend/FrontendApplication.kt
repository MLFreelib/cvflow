package easy.soc.hacks.frontend

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
open class FrontendApplication

fun main(args: Array<String>) {
    runApplication<FrontendApplication>(*args)
}
