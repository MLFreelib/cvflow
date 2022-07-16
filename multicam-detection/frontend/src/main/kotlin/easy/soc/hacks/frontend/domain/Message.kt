package easy.soc.hacks.frontend.domain

enum class MessageType {
    SUCCESS,
    WARNING,
    INFO,
    ERROR
}

data class Message(
    val title: String,

    val message: String,

    val messageType: MessageType
)