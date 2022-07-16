package easy.soc.hacks.frontend.configuration

import easy.soc.hacks.frontend.component.BackendWebSocketHandlerComponent
import org.springframework.context.annotation.Configuration
import org.springframework.web.socket.config.annotation.EnableWebSocket
import org.springframework.web.socket.config.annotation.WebSocketConfigurer
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry

@Configuration
@EnableWebSocket
class EnableBackendWebSocketConfiguration : WebSocketConfigurer {
    override fun registerWebSocketHandlers(registry: WebSocketHandlerRegistry) {
        registry.addHandler(BackendWebSocketHandlerComponent(), "/backend/websocket")
    }
}