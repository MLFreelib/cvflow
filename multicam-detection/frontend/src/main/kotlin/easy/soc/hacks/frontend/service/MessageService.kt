package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.*
import easy.soc.hacks.frontend.domain.MessageType.ERROR
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.i18n.LocaleContextHolder
import org.springframework.stereotype.Service
import java.util.*
import javax.servlet.http.HttpSession

@Service
class MessageService {
    @Autowired
    private lateinit var sessionService: SessionService
    fun sendMessage(httpSession: HttpSession, messageName: String, messageType: MessageType) {
        if (httpSession.getAttribute("messages") == null) {
            httpSession.setAttribute(
                "messages",
                mutableListOf<Message>()
            )
        }

        @Suppress("UNCHECKED_CAST")
        (httpSession.getAttribute("messages") as MutableList<Message>).add(
            Message(
                title = ResourceBundle.getBundle(
                    "messages",
                    LocaleContextHolder.getLocale()
                ).getString("$messageName.title"),
                message = ResourceBundle.getBundle(
                    "messages",
                    LocaleContextHolder.getLocale()
                ).getString("$messageName.message"),
                messageType = messageType
            )
        )
    }

    fun isAccessGranted(
        httpSession: HttpSession,
        streamingType: StreamingType? = null,
        sessionId: String? = null,
        video: Video? = null,
        checkSession: Optional<Session> = sessionService.getActiveSession()
    ): Boolean {
        val session = checkSession.orElseGet { null }

        when {
            session == null ->
                sendMessage(
                    httpSession,
                    "message.session.doesnt.exist",
                    ERROR
                )

            streamingType != null && session.streamingType != streamingType ->
                sendMessage(
                    httpSession,
                    "message.unmatched.streaming.type",
                    ERROR
                )

            sessionId != null && session.id != sessionId ->
                sendMessage(
                    httpSession,
                    "message.session.is.unmatched",
                    ERROR
                )

            video != null && session.id != video.session.id ->
                sendMessage(
                    httpSession,
                    "message.session.is.unmatched",
                    ERROR
                )

            video != null && session.streamingType != video.streamingType ->
                sendMessage(
                    httpSession,
                    "message.unmatched.streaming.type",
                    ERROR
                )


            else ->
                return true
        }

        return false
    }
}