package koach

import asPercentage
import koach.HyperParameter.IntList
import koach.HyperParameter.Long
import kotlinx.serialization.*
import kotlinx.serialization.builtins.list
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonConfiguration
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.plus
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.time.format.DateTimeFormatter
import kotlin.properties.Delegates
import kotlin.random.Random
import kotlin.random.nextInt

@Serializable
data class TrainedModel(
        val modelFileName: String,
        val configuration: List<@Polymorphic HyperParameter<@ContextualSerialization Any>>,
        val statistics: @Polymorphic ModelStatistics
) {
    override fun toString(): String = "Model: $modelFileName\n  $configuration\n  $statistics\n"
}

sealed class ModelStatistics {
    abstract val defaultMetric: Double
}

@Serializable
@SerialName("v1")
data class ModelStatisticsV1(
        val train: DataSetStatisticsV1,
        val test: DataSetStatisticsV1
) : ModelStatistics() {
    override val defaultMetric: Double get() = test.f1

    override fun toString() = "Train: $train; Test: $test"
}

@Serializable
data class DataSetStatisticsV1(val accuracy: Double, val precision: Double, val recall: Double, val f1: Double) {
    override fun toString() = "accuracy=${accuracy.asPercentage()}, " +
            "precision=${precision.asPercentage()}, " +
            "recall=${recall.asPercentage()}, " +
            "f1=${f1.asPercentage()}"
}

val statisticsModule = SerializersModule {
    polymorphic(ModelStatistics::class) {
        ModelStatisticsV1::class with ModelStatisticsV1.serializer()
    }
}

interface ModelStore<M> {
    fun save(path: Path, model: M)
    fun restore(path: Path): M
}

data class Recipe<M>(val name: String, internal val block: RecipeBuilder<M>.() -> Unit)

typealias Trainer<M> = () -> Pair<M, ModelStatistics>

class RecipeBuilder<M> {
    private val _hyperParameters = mutableListOf<HyperParameter<*>>()
    private var _trainer: Trainer<M>? = null

    fun long(name: String): kotlin.Long = hyperParameter(Long(name, Random.nextLong()))

    fun intList(name: String, vararg ranges: IntRange): List<Int> =
            hyperParameter(IntList(name, ranges.map { Random.nextInt(it) }))

    private fun <T : Any> hyperParameter(hyperParameter: HyperParameter<T>): T {
        _hyperParameters.add(hyperParameter)
        return hyperParameter.value
    }

    fun trainer(trainer: Trainer<M>) {
        _trainer = trainer
    }

    internal val hyperParameters get() = _hyperParameters.toList()

    internal fun train(): Pair<M, ModelStatistics> {
        val trainer = _trainer ?: error("Recipe has no defined trainer")
        return trainer()
    }
}

@ImplicitReflectionSerializer
class Coach<M>(
        private val modelStore: ModelStore<M>,
        private val modelSaveLocation: Path = Path.of("./models")
) {
    private val json = Json(JsonConfiguration.Stable, HyperParameter.serializersModule + statisticsModule)
    private var trainedModels: List<TrainedModel> by Delegates.observable(loadModelDescriptions()) { _, _, value ->
        writeModelDescriptions(value)
    }

    fun recipe(name: String, block: RecipeBuilder<M>.() -> Unit): Recipe<M> {
        return Recipe(name, block)
    }

    fun searchMore(modelCount: Int, recipe: Recipe<M>) {
        var trainedModelCount = 0
        var triedModel = 0
        while (trainedModelCount < modelCount && triedModel < 10) {
            val recipeBuilder = RecipeBuilder<M>()
            val block = recipe.block
            recipeBuilder.block()

            val hyperParameters = recipeBuilder.hyperParameters
            if (trainedModels.any { it.configuration == hyperParameters }) {
                triedModel++
                continue
            }

            println("Trying configuration: $hyperParameters")
            val (model, statistics) = recipeBuilder.train()
            println(statistics)

            trainedModelCount++

            val timestamp = DateTimeFormatter.ISO_INSTANT.format(Instant.now())
            val trainedModel = TrainedModel("model-$timestamp", hyperParameters, statistics)
            trainedModels += trainedModel
            saveModel(trainedModel, model)
        }
    }

    fun bestModels(modelCount: Int): List<TrainedModel> {
        return bestModels(modelCount) { it.statistics.defaultMetric }
    }

    fun <R : Comparable<R>> bestModels(modelCount: Int, selector: (TrainedModel) -> R?): List<TrainedModel> {
        return trainedModels.sortedByDescending(selector).subList(0, modelCount)
    }

    private fun saveModel(description: TrainedModel, model: M) {
        Files.createDirectories(modelSaveLocation)
        modelStore.save(modelSaveLocation.resolve(description.modelFileName), model)
    }

    fun restoreModel(description: TrainedModel): M {
        return modelStore.restore(modelSaveLocation.resolve(description.modelFileName))
    }

    fun reevaluateModels(block: (M) -> ModelStatistics) {
        println("Reavaluating models")
        trainedModels = trainedModels.map { tm ->
            val m = restoreModel(tm)
            val statistics = block(m)
            tm.copy(statistics = statistics)
        }
    }

    private fun loadModelDescriptions(): List<TrainedModel> {
        val modelDescriptionFile = modelSaveLocation.resolve("models.json")
        if (!Files.exists(modelDescriptionFile)) return listOf()
        val modelsAsString = Files.readString(modelDescriptionFile)
        return json.parse(TrainedModel.serializer().list, modelsAsString)
    }

    private fun writeModelDescriptions(trainedModels: List<TrainedModel>) {
        val modelDescriptionFile = modelSaveLocation.resolve("models.json")
        val modelsAsString = json.stringify(TrainedModel.serializer().list, trainedModels)
        Files.createDirectories(modelSaveLocation)
        Files.writeString(modelDescriptionFile, modelsAsString)
    }
}

class DL4JModelStore : ModelStore<MultiLayerNetwork> {
    override fun save(path: Path, model: MultiLayerNetwork) =
            ModelSerializer.writeModel(model, path.toFile(), false)

    override fun restore(path: Path): MultiLayerNetwork =
            ModelSerializer.restoreMultiLayerNetwork(path.toFile())
}
