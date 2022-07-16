package easy.soc.hacks.frontend.service

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ArrayNode
import easy.soc.hacks.frontend.domain.Session
import easy.soc.hacks.frontend.domain.StreamingType.FILE
import easy.soc.hacks.frontend.domain.Video
import easy.soc.hacks.frontend.service.BackendBrokerService.Companion.Command.*
import org.springframework.stereotype.Service
import org.springframework.web.socket.TextMessage
import org.springframework.web.socket.WebSocketSession

@Service
class BackendBrokerService {
    companion object {
        enum class Command(
            val command: String
        ) {
            START_SESSION("START_SESSION"),
            APPEND_VIDEO("APPEND_VIDEO"),
            SET_CALIBRATION("SET_CALIBRATION"),
            START_STREAMING("START_STREAMING"),
            STOP_SESSION("STOP_SESSION");

            override fun toString(): String = command
        }
    }

    fun startSession(webSocketSession: WebSocketSession?, session: Session) {
        val objectMapper = ObjectMapper()
        val command = objectMapper.createObjectNode()

        command.put("command", START_SESSION.command)
        command.put("sessionId", session.id)

        webSocketSession?.sendMessage(
            TextMessage(
                objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(command)
            )
        )
    }

    fun stopSession(webSocketSession: WebSocketSession?, session: Session) {
        val objectMapper = ObjectMapper()
        val command = objectMapper.createObjectNode()

        command.put("command", STOP_SESSION.command)
        command.put("sessionId", session.id)

        webSocketSession?.sendMessage(
            TextMessage(
                objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(command)
            )
        )
    }

    fun appendVideo(webSocketSession: WebSocketSession?, video: Video) {
        val objectMapper = ObjectMapper()
        val command = objectMapper.createObjectNode()

        command.put("command", APPEND_VIDEO.command)
        command.put("videoId", video.id)
        command.put("sessionId", video.session.id)
        if (video.streamingType == FILE) {
            command.put(
                "uri",
                "video/download?id=${video.id}&session=${video.session.id}"
            )
        } else {
            command.put("uri", video.uri)
        }
        command.put("streamingType", video.streamingType.value)

        webSocketSession?.sendMessage(
            TextMessage(
                objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(command)
            )
        )
    }

    fun setCalibration(
        webSocketSession: WebSocketSession?,
        video: Video
    ) {
        val objectMapper = ObjectMapper()
        val command = objectMapper.createObjectNode()

        command.put("command", SET_CALIBRATION.command)
        command.put("videoId", video.id)
        command.put("sessionId", video.session.id)

        val calibrationPointJsonArray = objectMapper.createArrayNode()

        for (calibrationPoint in video.calibrationPointList) {
            val calibrationPointJson = objectMapper.createObjectNode()
            calibrationPointJson.put("xScreen", calibrationPoint.xScreen)
            calibrationPointJson.put("yScreen", calibrationPoint.yScreen)
            calibrationPointJson.put("xWorld", calibrationPoint.xWorld)
            calibrationPointJson.put("yWorld", calibrationPoint.yWorld)
            calibrationPointJson.put("zWorld", calibrationPoint.zWorld)

            calibrationPointJsonArray.add(calibrationPointJson)
        }

        command.set<ArrayNode>("calibrationPointList", calibrationPointJsonArray)

        webSocketSession?.sendMessage(
            TextMessage(
                objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(command)
            )
        )
    }

    fun startStreaming(
        webSocketSession: WebSocketSession?,
        session: Session
    ) {
        val objectMapper = ObjectMapper()
        val command = objectMapper.createObjectNode()

        command.put("command", START_STREAMING.command)
        command.put("sessionId", session.id)

        webSocketSession?.sendMessage(
            TextMessage(
                objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(command)
            )
        )
    }
}