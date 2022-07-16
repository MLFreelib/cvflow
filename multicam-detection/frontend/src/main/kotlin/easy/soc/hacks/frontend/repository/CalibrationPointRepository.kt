package easy.soc.hacks.frontend.repository

import easy.soc.hacks.frontend.domain.CalibrationPoint
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository

@Repository
interface CalibrationPointRepository : JpaRepository<CalibrationPoint, Long>