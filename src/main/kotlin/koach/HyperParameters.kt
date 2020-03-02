package koach

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.modules.SerializersModule

sealed class HyperParameter<out T : @Serializable Any> {
    abstract val name: String
    abstract val value: T

    @Serializable
    @SerialName("long")
    data class Long(override val name: String, override val value: kotlin.Long) : HyperParameter<kotlin.Long>() {
        override fun toString() = "$name: $value"
    }

    @Serializable
    @SerialName("int-list")
    data class IntList(override val name: String, override val value: List<Int>) : HyperParameter<List<Int>>() {
        override fun toString() = "$name: $value"
    }

    companion object {
        val serializersModule = SerializersModule {
            polymorphic(HyperParameter::class) {
                Long::class with Long.serializer()
                IntList::class with IntList.serializer()
            }
        }
    }
}
