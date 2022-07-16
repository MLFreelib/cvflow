package easy.soc.hacks.frontend.configuration

import org.springframework.boot.web.servlet.MultipartConfigFactory
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import javax.servlet.MultipartConfigElement

@Configuration
class MultipartResolverConfiguration {
    @Bean
    fun multipartConfigElement(): MultipartConfigElement {
        return MultipartConfigFactory().createMultipartConfig()
    }
}