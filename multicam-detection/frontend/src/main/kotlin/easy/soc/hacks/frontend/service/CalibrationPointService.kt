package easy.soc.hacks.frontend.service

import easy.soc.hacks.frontend.domain.CalibrationPoint
import easy.soc.hacks.frontend.repository.CalibrationPointRepository
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.stereotype.Service

@Service
class CalibrationPointService {
    @Autowired
    private lateinit var calibrationPointRepository: CalibrationPointRepository

    fun save(calibrationPoint: CalibrationPoint) = calibrationPointRepository.save(calibrationPoint)
}