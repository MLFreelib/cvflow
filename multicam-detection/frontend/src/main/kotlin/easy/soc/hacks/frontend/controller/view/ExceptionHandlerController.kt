package easy.soc.hacks.frontend.controller.view

import easy.soc.hacks.frontend.domain.MessageType.ERROR
import easy.soc.hacks.frontend.service.MessageService
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.http.HttpStatus.INTERNAL_SERVER_ERROR
import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.bind.annotation.ResponseStatus
import javax.servlet.http.HttpSession

@Controller
class ExceptionHandlerController {
    @Autowired
    private lateinit var messageService: MessageService

    @ResponseStatus(INTERNAL_SERVER_ERROR)
    @ExceptionHandler
    fun handleException(
        httpSession: HttpSession
    ): String {
        messageService.sendMessage(
            httpSession,
            "message.error.500.",
            ERROR
        )

        return "redirect:/500"
    }
}