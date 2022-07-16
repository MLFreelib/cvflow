package easy.soc.hacks.frontend.component

import org.springframework.stereotype.Component
import org.springframework.web.socket.CloseStatus
import org.springframework.web.socket.WebSocketHandler
import org.springframework.web.socket.WebSocketMessage
import org.springframework.web.socket.WebSocketSession

@Component
class BackendWebSocketHandlerComponent : WebSocketHandler {
    companion object {
        var activeBackendWebSocketSession: WebSocketSession? = null
    }

    override fun afterConnectionEstablished(session: WebSocketSession) {
        if (activeBackendWebSocketSession != null) {
            session.close(CloseStatus.NOT_ACCEPTABLE)
            return
        }
        activeBackendWebSocketSession = session
    }

    override fun handleMessage(session: WebSocketSession, message: WebSocketMessage<*>) {
        return
    }

    override fun handleTransportError(session: WebSocketSession, exception: Throwable) {
        return
    }

    override fun afterConnectionClosed(session: WebSocketSession, closeStatus: CloseStatus) {
        if (activeBackendWebSocketSession == session) {
            activeBackendWebSocketSession = null
        }
    }

    override fun supportsPartialMessages(): Boolean = true
}