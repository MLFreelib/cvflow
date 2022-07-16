package easy.soc.hacks.frontend.configuration

import easy.soc.hacks.frontend.domain.StreamingType
import easy.soc.hacks.frontend.domain.StreamingType.CAMERA
import easy.soc.hacks.frontend.domain.StreamingType.FILE
import org.springframework.context.annotation.Configuration
import org.springframework.core.convert.converter.Converter
import org.springframework.format.FormatterRegistry
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer

@Configuration
class StreamingTypeConverterConfiguration : WebMvcConfigurer {
    override fun addFormatters(registry: FormatterRegistry) {
        registry.addConverter(StreamingTypeConverter())
    }
}
class StreamingTypeConverter : Converter<String, StreamingType> {
    override fun convert(source: String): StreamingType? {
        return when (source) {
            "camera" -> CAMERA
            "file" -> FILE
            else -> null
        }
    }
}