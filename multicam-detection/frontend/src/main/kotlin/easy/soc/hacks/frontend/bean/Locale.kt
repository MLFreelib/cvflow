package easy.soc.hacks.frontend.bean

import org.springframework.context.annotation.Bean
import org.springframework.web.servlet.LocaleResolver
import org.springframework.web.servlet.i18n.LocaleChangeInterceptor
import org.springframework.web.servlet.i18n.SessionLocaleResolver
import java.util.*


@Bean
fun localeResolver(): LocaleResolver {
    val sessionLocaleResolver = SessionLocaleResolver()
    sessionLocaleResolver.setDefaultLocale(Locale.US)
    return sessionLocaleResolver
}

@Bean
fun localeChangeInterceptor(): LocaleChangeInterceptor {
    val localeChangeInterceptor = LocaleChangeInterceptor()
    localeChangeInterceptor.paramName = "lang"
    return localeChangeInterceptor
}