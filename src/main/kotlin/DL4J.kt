import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.BatchNormalization
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.LossLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.ViewIterator
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import kotlin.math.min
import kotlin.random.Random
import kotlin.system.exitProcess

fun neuralNetConfiguration(block: NeuralNetConfiguration.Builder.() -> Unit): NeuralNetConfiguration.Builder {
    val builder = NeuralNetConfiguration.Builder()
    builder.block()
    return builder
}

infix fun NeuralNetConfiguration.Builder.list(block: NeuralNetConfiguration.ListBuilder.() -> Unit): MultiLayerConfiguration {
    val builder = this.list()
    builder.block()
    return builder.build()
}

fun NeuralNetConfiguration.ListBuilder.denseLayer(block: DenseLayer.Builder.() -> Unit): Unit {
    val builder = DenseLayer.Builder()
    builder.block()
    layer(builder.build())
}

fun NeuralNetConfiguration.ListBuilder.batchNormalization(block: BatchNormalization.Builder.() -> Unit): Unit {
    val builder = BatchNormalization.Builder()
    builder.block()
    layer(builder.build())
}

fun NeuralNetConfiguration.ListBuilder.lossLayer(block: LossLayer.Builder.() -> Unit): Unit {
    val builder = LossLayer.Builder()
    builder.block()
    layer(builder.build())
}

fun Evaluation.eval(network: MultiLayerNetwork, dataSet: DataSet): Evaluation {
    eval(dataSet.labels, network.output(dataSet.features, Layer.TrainingMode.TEST))
    return this
}

fun withUIServer(block: (statsStorage: StatsStorage) -> Unit) {
    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    try {
        block(statsStorage)
    } finally {
        uiServer.detach(statsStorage)
        uiServer.stop()
        exitProcess(0)
    }
}

open class MiniBatchIterator(
        private val data: DataSet,
        private val batchSize: Int,
        private val randomSeed: Long
) : DataSetIterator {

    private val random = Random(randomSeed)

    private var cursor = 0
    private var preProcessor: DataSetPreProcessor? = null

    private var currentData: DataSet = shuffleData()

    private fun shuffleData(): DataSet {
        val shuffledData = data.copy()
//        shuffledData.shuffle(random.nextLong())
        return shuffledData
    }

    override fun next(num: Int): DataSet {
        throw UnsupportedOperationException("Only allowed to retrieve dataset based on batch size")
    }

    override fun inputColumns(): Int {
        return currentData.numInputs()
    }

    override fun totalOutcomes(): Int {
        return currentData.numOutcomes()
    }

    override fun resetSupported(): Boolean {
        return true
    }

    override fun asyncSupported(): Boolean {
        // Already all in memory
        return false
    }

    override fun reset() {
        currentData = shuffleData()
        cursor = 0
    }

    override fun batch(): Int {
        return batchSize
    }

    override fun setPreProcessor(preProcessor: DataSetPreProcessor) {
        this.preProcessor = preProcessor
    }

    override fun getPreProcessor(): DataSetPreProcessor {
        return preProcessor!!
    }

    override fun getLabels(): List<String>? {
        return null
    }

    override fun hasNext(): Boolean {
        return cursor < currentData.numExamples()
    }

    override fun remove() {}

    override fun next(): DataSet {
        val last = min(currentData.numExamples(), cursor + batch())
        val next = currentData.getRange(cursor, last) as DataSet
        preProcessor?.preProcess(next)
        cursor += batch()
        return next
    }
}
