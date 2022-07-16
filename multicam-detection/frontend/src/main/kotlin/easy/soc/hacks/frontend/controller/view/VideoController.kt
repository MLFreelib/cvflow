package easy.soc.hacks.frontend.controller.view

import easy.soc.hacks.frontend.annotation.ModelHolder
import easy.soc.hacks.frontend.component.BackendWebSocketHandlerComponent.Companion.activeBackendWebSocketSession
import easy.soc.hacks.frontend.domain.*
import easy.soc.hacks.frontend.domain.MessageType.ERROR
import easy.soc.hacks.frontend.domain.MessageType.WARNING
import easy.soc.hacks.frontend.domain.SessionStatusType.*
import easy.soc.hacks.frontend.service.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.http.MediaType
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Controller
import org.springframework.ui.Model
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.ModelAttribute
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.multipart.MultipartFile
import java.util.concurrent.ConcurrentSkipListSet
import java.util.concurrent.atomic.AtomicLong
import javax.servlet.http.HttpSession

@Controller
class VideoController {
    @Autowired
    private lateinit var backendBrokerService: BackendBrokerService

    @Autowired
    private lateinit var videoService: VideoService

    @Autowired
    private lateinit var videoFragmentService: VideoFragmentService

    @Autowired
    private lateinit var calibrationPointService: CalibrationPointService

    @Autowired
    private lateinit var sessionService: SessionService

    @Autowired
    private lateinit var messageService: MessageService

    @GetMapping("")
    @ModelHolder
    fun index(): String {
        return "index"
    }

    @GetMapping("video/preview")
    @ModelHolder
    fun previewCameraVideo(
        @RequestParam("type") streamingType: StreamingType,
        @RequestParam("session") sessionId: String?,
        model: Model,
        httpSession: HttpSession
    ): String {
        val session = when (sessionId) {
            null ->
                sessionService.getActiveOrStreamingSession()
            else ->
                sessionService.findSessionById(sessionId)
        }.orElseGet { null }

        if (session != null) {
            if (session.streamingType == streamingType) {
                val maxVideoFragmentId =
                    videoFragmentService.getMaxVideoFragmentIdBySessionId(
                        sessionId = session.id
                    ).orElseGet { null }

                httpSession.setAttribute(
                    "nextBatchId",
                    when (maxVideoFragmentId) {
                        null -> AtomicLong(1)
                        else -> AtomicLong(maxVideoFragmentId)
                    }
                )

                httpSession.setAttribute(
                    "processedVideoFragmentIdSet",
                    ConcurrentSkipListSet<VideoFragmentId>()
                )
                httpSession.setAttribute(
                    "processedMapBatchIdSet",
                    ConcurrentSkipListSet<Long>()
                )

                model.addAttribute(
                    "videoList",
                    videoService.findVideosBySessionId(session.id)
                )
            } else {
                messageService.sendMessage(
                    httpSession,
                    "message.unmatched.streaming.type",
                    ERROR
                )
            }

            model.addAttribute(
                "currentSession",
                session
            )
        } else {
            if (sessionId != null) {
                messageService.sendMessage(
                    httpSession,
                    "message.session.doesnt.exist",
                    ERROR
                )
            }
        }

        return "preview"
    }

    @GetMapping("video/add")
    @ModelHolder
    fun addVideoGet(
        @RequestParam("type") streamingType: StreamingType,
        httpSession: HttpSession
    ): String {
        if (!messageService.isAccessGranted(
                httpSession = httpSession,
                streamingType = streamingType
            )
        ) {
            return "redirect:/"
        }

        return "addVideo"
    }

    @Async
    @PostMapping(
        path = ["video/add"],
        consumes = [MediaType.MULTIPART_FORM_DATA_VALUE]
    )
    fun addVideo(
        @RequestParam("type") streamingType: StreamingType,
        @RequestParam("name") name: String,
        @RequestParam("uri") uri: String?,
        @RequestParam("file") file: MultipartFile?,
        httpSession: HttpSession
    ): String {
        if (!messageService.isAccessGranted(
                httpSession = httpSession,
                streamingType = streamingType
            )
        ) {
            return "redirect:/"
        }

        if (activeBackendWebSocketSession == null) {
            messageService.sendMessage(
                httpSession,
                "message.backend.is.disable",
                ERROR
            )

            return "redirect:/video/preview?type=${streamingType.value}"
        }

        val session = sessionService.getActiveSession().orElseGet { null }

        if (session != null) {
            val savedVideo = videoService.save(
                Video(
                    session = session,
                    name = name,
                    uri = uri,
                    streamingType = streamingType
                ),
                file
            )

            backendBrokerService.appendVideo(
                activeBackendWebSocketSession,
                savedVideo
            )
        }

        return "redirect:/video/preview?type=${streamingType.value}"
    }

    @GetMapping("video/calibration")
    @ModelHolder
    fun calibrationVideo(
        @RequestParam("id") videoId: Long,
        model: Model,
        httpSession: HttpSession
    ): String {
        val session = sessionService.getActiveSession().get()
        val video = videoService.findVideoByIdAndSessionId(
            id = videoId,
            session.id
        ).orElseGet { null }

        if (video == null) {
            messageService.sendMessage(
                httpSession,
                "message.video.not.found",
                ERROR
            )

            return "redirect:/"
        }

        if (!messageService.isAccessGranted(
                httpSession = httpSession,
                video = video
            )
        ) {
            return "redirect:/"
        }

        if (video.session.status == DONE) {
            messageService.sendMessage(
                httpSession,
                "message.video.calibration.edit.done.session",
                ERROR
            )

            return "redirect:/video/preview?type=${video.streamingType.value}&session=${video.session.id}"
        }

        model.addAttribute("video", video)

        return "calibrationVideo"
    }

    @PostMapping("video/calibration/save")
    fun saveCalibration(
        @RequestParam("id") videoId: Long,
        @ModelAttribute calibrationPointListWrapper: CalibrationPointListWrapper,
        httpSession: HttpSession
    ): String {
        val session = sessionService.getActiveSession().get()
        val video = videoService.findVideoByIdAndSessionId(videoId, session.id).orElseGet { null }

        if (video == null) {
            messageService.sendMessage(
                httpSession,
                "message.video.not.found",
                ERROR
            )

            return "redirect:/"
        }

        if (!messageService.isAccessGranted(
                httpSession = httpSession,
                video = video
            )
        ) {
            return "redirect:/"
        }

        val calibrationPointList = calibrationPointListWrapper.toCalibrationPointList().map {
            calibrationPointService.save(it)
        }

        val updateVideo = Video(
            id = video.id,
            session = video.session,
            name = video.name,
            calibrationPointList = calibrationPointList,
            uri = video.uri,
            streamingType = video.streamingType
        )
        videoService.setCalibration(updateVideo)

        backendBrokerService.setCalibration(
            activeBackendWebSocketSession,
            updateVideo
        )

        return "redirect:/video/preview?type=${video.streamingType.value}"
    }

    @PostMapping("start")
    fun startVideoStream(
        httpSession: HttpSession
    ): String {
        if (!messageService.isAccessGranted(
                httpSession = httpSession
            )
        ) {
            return "redirect:/"
        }

        val session = sessionService.getActiveSession().orElseGet { null }
        val videoList = videoService.findVideosBySessionId(session.id)

        if (videoList.isEmpty()) {
            messageService.sendMessage(
                httpSession,
                "message.start.streaming.with.empty.video.list",
                WARNING
            )

            return "redirect:/video/preview?type=${session.streamingType.value}"
        }

        if (activeBackendWebSocketSession == null) {
            messageService.sendMessage(
                httpSession,
                "message.backend.is.disable",
                ERROR
            )

            return "redirect:/video/preview?type=${session.streamingType.value}"
        }

        backendBrokerService.startStreaming(
            activeBackendWebSocketSession,
            session
        )

        sessionService.save(
            Session(
                id = session.id,
                startTime = session.startTime,
                status = STREAMING,
                streamingType = session.streamingType
            )
        )

        return "redirect:/video/preview?type=${session.streamingType.value}"
    }

    private fun endSession(session: Session) {
        backendBrokerService.stopSession(activeBackendWebSocketSession, session)

        sessionService.save(
            Session(
                id = session.id,
                startTime = session.startTime,
                status = DONE,
                streamingType = session.streamingType
            )
        )
    }

    @PostMapping("stop")
    fun stop(
        httpSession: HttpSession
    ): String {
        if (!messageService.isAccessGranted(
                httpSession = httpSession,
                checkSession = sessionService.getStreamingSession()
            )
        ) {
            return "redirect:/"
        }

        val session = sessionService.getStreamingSession().orElseGet { null }

        if (activeBackendWebSocketSession == null) {
            messageService.sendMessage(
                httpSession,
                "message.backend.is.disable",
                ERROR
            )

            return "redirect:/video/preview?type=${session.streamingType.value}"
        }

        endSession(session)

        return "redirect:/video/preview?type=${session.streamingType.value}"
    }

    @PostMapping("delete")
    fun deleteSession(
        httpSession: HttpSession
    ): String {
        if (!messageService.isAccessGranted(
                httpSession = httpSession,
            )
        ) {
            return "redirect:/"
        }

        val session = sessionService.getActiveSession().orElseGet { null }

        if (activeBackendWebSocketSession == null) {
            messageService.sendMessage(
                httpSession,
                "message.backend.is.disable",
                ERROR
            )

            return "redirect:/video/preview?type=${session.streamingType.value}"
        }

        backendBrokerService.stopSession(activeBackendWebSocketSession, session)

        endSession(session)

        return "redirect:/video/preview?type=${session.streamingType.value}"
    }

    @PostMapping("session/create")
    fun createSession(
        @RequestParam("type") streamingType: StreamingType,
        httpSession: HttpSession
    ): String {
        if (activeBackendWebSocketSession == null) {
            messageService.sendMessage(
                httpSession,
                "message.backend.is.disable",
                ERROR
            )

            return "redirect:/"
        }

        return when (sessionService.getActiveSession().orElseGet { null }) {
            null -> {
                val session = sessionService.save(
                    Session(
                        status = ACTIVE,
                        streamingType = streamingType
                    )
                )

                backendBrokerService.startSession(
                    activeBackendWebSocketSession,
                    session
                )

                "redirect:/video/preview?type=${streamingType.value}&session=${
                    session.id
                }"
            }

            else -> {
                messageService.sendMessage(
                    httpSession,
                    "message.session.create",
                    ERROR
                )

                "redirect:/video/preview?type=${streamingType.value}"
            }
        }
    }
}