package easy.soc.hacks.frontend.interceptor

import easy.soc.hacks.frontend.annotation.ModelHolder
import easy.soc.hacks.frontend.domain.Message
import easy.soc.hacks.frontend.service.SessionService
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.http.HttpStatus.NOT_FOUND
import org.springframework.stereotype.Component
import org.springframework.web.method.HandlerMethod
import org.springframework.web.servlet.HandlerInterceptor
import org.springframework.web.servlet.ModelAndView
import javax.servlet.http.HttpServletRequest
import javax.servlet.http.HttpServletResponse

@Component
class WebInterceptor : HandlerInterceptor {
    @Autowired
    private lateinit var sessionService: SessionService

    override fun preHandle(request: HttpServletRequest, response: HttpServletResponse, handler: Any): Boolean {
        val user = request.session.getAttribute("user")

        if (response.status == NOT_FOUND.value()) {
            response.sendRedirect("/404")
            return false
        }

        return true
    }

    @Suppress("UNCHECKED_CAST")
    override fun postHandle(
        request: HttpServletRequest,
        response: HttpServletResponse,
        handler: Any,
        modelAndView: ModelAndView?
    ) {
        if (modelAndView != null && handler is HandlerMethod) {
            if (handler.hasMethodAnnotation(ModelHolder::class.java)) {
                val user = request.session.getAttribute("user")
                modelAndView.model["user"] = user

                val sessionMessages = request.session.getAttribute("messages") as MutableList<Message>?
                var modelMessages = modelAndView.model["messages"] as MutableList<Message>?

                if (modelMessages == null) {
                    modelAndView.model["messages"] = mutableListOf<Message>()
                    modelMessages = mutableListOf()
                }

                if (sessionMessages != null) {
                    modelMessages.addAll(sessionMessages)
                }

                modelAndView.model["messages"] = modelMessages
                request.session.setAttribute("messages", mutableListOf<Message>())

                modelAndView.model["activeSession"] = sessionService.getActiveSession().orElseGet { null }
            }
        }
    }
}