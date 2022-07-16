package easy.soc.hacks.frontend.configuration

import easy.soc.hacks.frontend.interceptor.WebInterceptor
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.web.servlet.config.annotation.InterceptorRegistry
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer

@Configuration
class InterceptorConfiguration : WebMvcConfigurer {
    @Bean
    fun webInterceptorBean(): WebInterceptor {
        return WebInterceptor()
    }

    override fun addInterceptors(registry: InterceptorRegistry) {
        registry.addInterceptor(webInterceptorBean())
    }
}