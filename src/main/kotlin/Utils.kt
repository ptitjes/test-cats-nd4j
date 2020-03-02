import kotlin.time.DurationUnit
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource.Monotonic
import kotlin.time.measureTimedValue

@ExperimentalTime
inline fun <R> timed(prefix: String, block: () -> R): R {
    val timedResult = Monotonic.measureTimedValue(block)
    val elapsed = timedResult.duration.toInt(DurationUnit.SECONDS)
    println("$prefix took: ${elapsed}s")
    return timedResult.value
}

fun Double.asPercentage() = "%.1f%%".format(this * 100)
