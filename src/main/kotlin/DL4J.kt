import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.BatchNormalization
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.LossLayer

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
