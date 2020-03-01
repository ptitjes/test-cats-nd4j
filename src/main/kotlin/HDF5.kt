import org.bytedeco.hdf5.*
import org.bytedeco.hdf5.global.hdf5
import org.bytedeco.javacpp.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class HDF5 {
    companion object {
        inline fun <R> withH5File(name: String, block: H5File.() -> R): R {
            val file = H5File(name, hdf5.H5F_ACC_RDONLY())
            try {
                return file.block()
            } finally {
                file.deallocate()
            }
        }

        init {
            try {
                Loader.load(hdf5::class.java)
            } catch (e: Throwable) {
                e.printStackTrace()
            }
        }
    }
}

/**
 * Get list of objects with a given type from this group.
 *
 * @receiver HDF5 file or group
 * @param objectType Type of object as integer
 * @return List of HDF5 object names
 */
fun Group.getObjectNames(objectType: Int): List<String> {
    val names = arrayListOf<String>()
    for (i in 0 until this.numObjs) {
        val objPtr = this.getObjnameByIdx(i)
        if (this.childObjType(objPtr) == objectType)
            names.add(this.getObjnameByIdx(i).string)
    }
    return names
}

/**
 * Get list of data sets from this group.
 *
 * @receiver HDF5 file or group.
 * @return List of HDF5 data set names
 */
fun Group.getDataSetNames(): List<String> {
    return this.getObjectNames(hdf5.H5O_TYPE_DATASET)
}

/**
 * Get list of groups from this group.
 *
 * @receiver HDF5 file or group.
 * @return List of HDF5 group names
 */
fun Group.getGroupNames(): List<String> {
    return this.getObjectNames(hdf5.H5O_TYPE_GROUP)
}

inline fun <R> H5Location.withGroup(name: String, block: Group.() -> R): R {
    val group = this.openGroup(name)
    try {
        return group.block()
    } finally {
        group.deallocate()
    }
}


inline fun <R> H5Location.withDataSet(name: String, block: DataSet.() -> R): R {
    val dataSet = this.openDataSet(name)
    try {
        return dataSet.block()
    } finally {
        dataSet.deallocate()
    }
}

fun DataSet.getDimensions(): LongArray {
    val space = this.space

    val nbDims = space.simpleExtentNdims
    val dims = LongArray(nbDims)
    space.getSimpleExtentDims(dims)

    return dims
}

private fun DataSet.readTo(array: ByteArray) {
    val bp = BytePointer(*array)
    val p = Pointer(bp)
    this.read(p, DataType(PredType.NATIVE_UINT8()))
    bp.get(array)
}

private fun DataSet.readTo(array: LongArray) {
    val fp = LongPointer(*array)
    this.read(fp, DataType(PredType.NATIVE_INT64()))
    fp.get(array)
}

@ExperimentalUnsignedTypes
fun DataSet.readAsUByteNDArray(): INDArray {
    val dimensions = getDimensions()
    val buffer = UByteArray(dimensions.reduce(Long::times).toInt())
    readTo(buffer.asByteArray())

    val array = buffer.map { it.toFloat() }.toTypedArray().toFloatArray()

    return Nd4j.create(array, dimensions, 'c')
}

fun DataSet.readAsLongNDArray(): INDArray {
    val dimensions = getDimensions()
    val buffer = LongArray(dimensions.reduce(Long::times).toInt())
    readTo(buffer)
    return Nd4j.createFromArray(buffer.toTypedArray())
}
