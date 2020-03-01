package koach

import kotlinx.serialization.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import java.nio.file.Files
import java.nio.file.Path
import kotlin.properties.Delegates

@Serializable
data class TrainedModel<C : @Serializable Any>(
        val modelFileName: String,
        val configuration: C,
        val statistics: ModelStatistics
)

@Serializable
data class ModelStatistics(val trainAccuracy: Double, val testAccuracy: Double)

interface ModelStore<M> {
    fun save(path: Path, model: M)
    fun restore(path: Path): M
}

class RandomParameterBuilder {

}

class ConfigurationBuilder<C> {
    fun configure(block: RandomParameterBuilder.() -> C): C {
        val builder = RandomParameterBuilder()
        return builder.block()
    }
}

typealias Recipe<C, M> = ConfigurationBuilder<C>.() -> Triple<C, M, ModelStatistics>

@ImplicitReflectionSerializer
class Coach<C : @Serializable Any, M>(
        private val modelStore: ModelStore<M>,
        private val modelSaveLocation: Path = Path.of("./models"),
        private val recipe: Recipe<C, M>
) {

    var trainedModels: List<TrainedModel<C>> by Delegates.observable(loadModelDescriptions()) { _, _, value ->
        writeModelDescriptions(value)
    }

    fun searchMore(modelCount: Int) {

    }

    fun bestModels(modelCount: Int): List<TrainedModel<C>> {
        return listOf()
    }

    private fun saveModel(description: TrainedModel<C>, model: M) {
        modelStore.save(modelSaveLocation.resolve(description.modelFileName), model)
    }

    fun restoreModel(description: TrainedModel<C>): M {
        return modelStore.restore(modelSaveLocation.resolve(description.modelFileName))
    }

    private fun loadModelDescriptions(): List<TrainedModel<C>> {
        val modelDescriptionFile = modelSaveLocation.resolve("models.json")
        if (!Files.exists(modelDescriptionFile)) return listOf()
        val modelsAsString = Files.readString(modelDescriptionFile)
        val json = Json(JsonConfiguration.Stable)
        return json.parse(modelsAsString)
    }

    private fun writeModelDescriptions(trainedModels: List<TrainedModel<C>>) {
        val json = Json(JsonConfiguration.Stable)
        val modelDescriptionFile = modelSaveLocation.resolve("models.json")
        val modelsAsString = json.stringify(trainedModels)
        Files.writeString(modelDescriptionFile, modelsAsString)
    }
}

class DL4JModelStore : ModelStore<MultiLayerNetwork> {
    override fun save(path: Path, model: MultiLayerNetwork) =
            ModelSerializer.writeModel(model, path.toFile(), false)

    override fun restore(path: Path): MultiLayerNetwork =
            ModelSerializer.restoreMultiLayerNetwork(path.toFile())
}
