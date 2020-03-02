import koach.*
import kotlinx.serialization.ImplicitReflectionSerializer
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.stats.StatsListener
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
    withUIServer { statsStorage ->
        val (trainSet, testSet) = loadCatVsNonCatDataSets().also { (train, test) ->
            train.printDataSetStats("Train")
            test.printDataSetStats("Test")
            println()
        }

//    Seed: 13 - Accuracy: train=0.8995215311004785; test=0.86
//    Seed: 1485241205548954111 - Accuracy: train=0.9665071770334929; test=0.86

        val coach = Coach(DL4JModelStore())

        val simple = coach.recipe("simple") {
            val seed = 13L // long("seed")
            val numEpochs = 2500
            val hiddenLayerDims = intList("hiddenLayerDims", /*20..40, */20..50, 10..20, 5..15)

            trainer {
                val nx = trainSet.numInputs()

                val conf = neuralNetConfiguration {
                    seed(seed)
                    weightInit(WeightInit.XAVIER)
                    activation(Activation.RELU)
                    l2(0.00001)
                    optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    updater(Adam())
                } list {
                    for ((i, o) in (listOf(nx) + hiddenLayerDims).zipWithNext()) {
                        denseLayer { nIn(i); nOut(o) }
                        batchNormalization { }
                    }
                    denseLayer { nOut(1); activation(Activation.SIGMOID) }
                    lossLayer { lossFunction(LossFunction.XENT) }
                }

                val model = trainModel(conf, trainSet, testSet, numEpochs, statsStorage)
                model to statistics(model, trainSet, testSet)
            }
        }

        coach.searchMore(100, simple)

//        coach.reevaluateModels {
//            statistics(it, trainSet, testSet)
//        }

        coach.bestModels(10).forEach {
            println(it)
//            val m = coach.restoreModel(it)
//            Evaluation().eval(m, trainSet).apply { println(accuracy()) }
//            Evaluation().eval(m, testSet).apply { println(accuracy()) }
        }
    }
}

private fun trainModel(
        conf: MultiLayerConfiguration,
        trainSet: DataSet,
        testSet: DataSet,
        numEpochs: Int,
        statsStorage: StatsStorage? = null
): MultiLayerNetwork {
    val network = MultiLayerNetwork(conf)
    network.init()
    if (statsStorage != null) network.addListeners(StatsListener(statsStorage))

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
        ModelStatisticsV1(
                Evaluation().eval(network, trainSet).let { e -> DataSetStatisticsV1(e.accuracy(), e.precision(), e.recall(), e.f1()) },
                Evaluation().eval(network, testSet).let { e -> DataSetStatisticsV1(e.accuracy(), e.precision(), e.recall(), e.f1()) }
        )
