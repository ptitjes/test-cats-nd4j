import koach.Coach
import koach.DL4JModelStore
import koach.ModelStatistics
import kotlinx.serialization.ImplicitReflectionSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonConfiguration
import kotlinx.serialization.stringify
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import java.util.concurrent.TimeUnit

@ExperimentalUnsignedTypes
@ImplicitReflectionSerializer
fun main() {
    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    val (trainSet, testSet) = loadCatVsNonCatDataSets()

//    Seed: 13 - Accuracy: train=0.8995215311004785; test=0.86
//    Seed: 1485241205548954111 - Accuracy: train=0.9665071770334929; test=0.86

//    val random = Random.Default
//    val seeds = (1 until 100).map { random.nextLong() }
//    for (seed in seeds)
//        trainModel(seed, trainSet, testSet, statsStorage)

//    val seed = 13L
//    val (model, statistics) = trainModel(seed, trainSet, testSet, statsStorage)
//    with(statistics) {
//        println("Seed: $seed - Accuracy: train=${trainAccuracy}; test=${testAccuracy}")
//    }


    val json = Json(JsonConfiguration.Stable)
    println(json.stringify(Toto("Yo")))

//    val coach = Coach<CatVsNonCatConfiguration, MultiLayerNetwork>(DL4JModelStore()) {
//        val c = configure {
//            CatVsNonCatConfiguration(
//                    seed = 13,
//                    hiddenLayerDims = listOf(20, 7, 5)
//            )
//        }
//
//        trainModel(c, trainSet, testSet, statsStorage)
//    }
//
//    coach.searchMore(100)
}

@Serializable
data class Toto(val test: String)

@Serializable
data class CatVsNonCatConfiguration(
        val numEpochs: Int = 2500,
        val seed: Long,
        val hiddenLayerDims: List<Int>
)

private fun trainModel(
        configuration: CatVsNonCatConfiguration,
        trainSet: DataSet,
        testSet: DataSet,
        statsStorage: InMemoryStatsStorage
): Triple<CatVsNonCatConfiguration, MultiLayerNetwork, ModelStatistics> {

    val nx = trainSet.numInputs()

    val conf = neuralNetConfiguration {
        seed(seed)
        weightInit(WeightInit.XAVIER)
        activation(Activation.RELU)
        l2(0.00001)
        optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        updater(Adam())
    } list {
        for ((i, o) in (listOf(nx) + configuration.hiddenLayerDims).zipWithNext()) {
            denseLayer { nIn(i); nOut(o) }
            batchNormalization { }
        }
        denseLayer { nOut(1); activation(Activation.SIGMOID) }
        lossLayer { lossFunction(LossFunction.XENT) }
    }

    val model = trainModel(conf, trainSet, testSet, configuration.numEpochs, statsStorage)
    return Triple(configuration, model, statistics(model, trainSet, testSet))
}

private fun trainModel(
        conf: MultiLayerConfiguration,
        trainSet: DataSet,
        testSet: DataSet,
        numEpochs: Int,
        statsStorage: InMemoryStatsStorage
): MultiLayerNetwork {
    val network = MultiLayerNetwork(conf)
    network.init()
    network.addListeners(StatsListener(statsStorage))

    val train = SingletonDataSetIterator(trainSet)
    val test = SingletonDataSetIterator(testSet)
    val esConf = EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
            .epochTerminationConditions(MaxEpochsTerminationCondition(numEpochs))
            .iterationTerminationConditions(MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
            .scoreCalculator(DataSetLossCalculator(test, true))
            .evaluateEveryNEpochs(1)
            .modelSaver(LocalFileModelSaver("./temp"))
            .build()
    val trainer = EarlyStoppingTrainer(esConf, network, train)
    val result = trainer.fit()
    return result.bestModel
}

private fun statistics(network: MultiLayerNetwork, trainSet: DataSet, testSet: DataSet) =
        ModelStatistics(
                Evaluation().eval(network, trainSet).accuracy(),
                Evaluation().eval(network, testSet).accuracy()
        )

fun Evaluation.eval(network: MultiLayerNetwork, dataSet: DataSet): Evaluation {
    eval(dataSet.labels, network.output(dataSet.features, Layer.TrainingMode.TEST))
    return this
}

inline fun <R> timed(prefix: String, block: () -> R): R {
    val start = System.currentTimeMillis()
    val result = block()
    val stop = System.currentTimeMillis()
    val elapsed = (stop - start) / 1000
    println("$prefix took: ${elapsed}s")
    return result
}
