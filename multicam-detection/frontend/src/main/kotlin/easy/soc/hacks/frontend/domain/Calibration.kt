package easy.soc.hacks.frontend.domain

import lombok.Data
import javax.persistence.*
import javax.persistence.GenerationType.IDENTITY


@Table(name = "calibration_points")
@Entity
@Data
class CalibrationPoint(
    @Id
    @GeneratedValue(strategy = IDENTITY)
    @Column(name = "id", nullable = false)
    val id: Long = 0,

    @Column(name = "x_screen", nullable = false)
    val xScreen: Double,

    @Column(name = "y_screen", nullable = false)
    val yScreen: Double,

    @Column(name = "x_world", nullable = false)
    val xWorld: Double,

    @Column(name = "y_world", nullable = false)
    val yWorld: Double,

    @Column(name = "z_world", nullable = false)
    val zWorld: Double
)

data class CalibrationPointListWrapper(
    val xScreen0: Double,
    val yScreen0: Double,
    val xWorld0: Double,
    val yWorld0: Double,
    val zWorld0: Double,

    val xScreen1: Double,
    val yScreen1: Double,
    val xWorld1: Double,
    val yWorld1: Double,
    val zWorld1: Double,

    val xScreen2: Double,
    val yScreen2: Double,
    val xWorld2: Double,
    val yWorld2: Double,
    val zWorld2: Double,

    val xScreen3: Double,
    val yScreen3: Double,
    val xWorld3: Double,
    val yWorld3: Double,
    val zWorld3: Double,

    val xScreen4: Double,
    val yScreen4: Double,
    val xWorld4: Double,
    val yWorld4: Double,
    val zWorld4: Double,

    val xScreen5: Double,
    val yScreen5: Double,
    val xWorld5: Double,
    val yWorld5: Double,
    val zWorld5: Double,
) {
    fun toCalibrationPointList(): List<CalibrationPoint> {
        return listOf(
            CalibrationPoint(
                xScreen = xScreen0,
                yScreen = yScreen0,
                xWorld = xWorld0,
                yWorld = yWorld0,
                zWorld = zWorld0,
            ),

            CalibrationPoint(
                xScreen = xScreen1,
                yScreen = yScreen1,
                xWorld = xWorld1,
                yWorld = yWorld1,
                zWorld = zWorld1,
            ),

            CalibrationPoint(
                xScreen = xScreen2,
                yScreen = yScreen2,
                xWorld = xWorld2,
                yWorld = yWorld2,
                zWorld = zWorld2,
            ),

            CalibrationPoint(
                xScreen = xScreen3,
                yScreen = yScreen3,
                xWorld = xWorld3,
                yWorld = yWorld3,
                zWorld = zWorld3,
            ),

            CalibrationPoint(
                xScreen = xScreen4,
                yScreen = yScreen4,
                xWorld = xWorld4,
                yWorld = yWorld4,
                zWorld = zWorld4,
            ),

            CalibrationPoint(
                xScreen = xScreen5,
                yScreen = yScreen5,
                xWorld = xWorld5,
                yWorld = yWorld5,
                zWorld = zWorld5,
            )
        )
    }
}