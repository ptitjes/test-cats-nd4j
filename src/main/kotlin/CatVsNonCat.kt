import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import org.nd4j.linalg.dataset.DataSet
import java.io.File

@ExperimentalUnsignedTypes
fun loadCatVsNonCatDataSets(): Pair<DataSet, DataSet> {
    val trainSet = loadCatVsNonCatDataSet("train").apply { printDataSetStats("Train") }
    val testSet = loadCatVsNonCatDataSet("test").apply { printDataSetStats("Test") }
    println()

    checkDataSetAgainst(trainSet, "trainX.csv", "trainY.csv")
    return Pair(trainSet, testSet)
}

private fun DataSet.printDataSetStats(name: String) {
    println("$name data-set: #examples=${numExamples()}, #inputs=${numInputs()}, #outcomes=${numOutcomes()}")
}

@ExperimentalUnsignedTypes
private fun loadCatVsNonCatDataSet(dataSetName: String): DataSet {
    return HDF5.withH5File("datasets/${dataSetName}_catvnoncat.h5") {
        val xOrig = withDataSet("${dataSetName}_set_x") { readAsUByteNDArray() }
        val xShape = xOrig.shape()

        val m = xShape[0]
        val pixelCount = xShape.drop(1).reduce(Long::times)
        val xFlatten = xOrig.reshape(m, pixelCount)
        val x = xFlatten / 255

        val yOrig = withDataSet("${dataSetName}_set_y") { readAsLongNDArray() }
        val y = yOrig.reshape(yOrig.shape()[0], 1)

        DataSet(x, y)
    }
}

private fun checkDataSetAgainst(trainSet: DataSet, featuresCsv: String, labelsCsv: String) {
    val trainX = trainSet.features
    val trainXCsv = csvReader().readAll(File(featuresCsv))
    for (i in 0 until 209) {
        for (j in 0 until 12288) {
            try {
                val fromTrain = (trainX.getDouble(i, j) * 255).toInt()
                val fromCsv = trainXCsv[j][i].toInt()
                if (fromTrain != fromCsv) error("$i,$j -> $fromTrain != $fromCsv")
            } catch (e: IndexOutOfBoundsException) {
                println("$i,$j")
            }
        }
    }

    val trainY = trainSet.labels
    val trainYCsv = csvReader().readAll(File(labelsCsv))
    for (j in 0 until 209) {
        val fromTrain = trainY.getDouble(j, 0).toInt()
        val fromCsv = trainYCsv[0][j].toInt()
        if (fromTrain != fromCsv) error("0,$j -> $fromTrain != $fromCsv")
    }
}