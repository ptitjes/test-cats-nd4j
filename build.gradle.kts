import org.jetbrains.kotlin.config.KotlinCompilerVersion

plugins {
    application
    kotlin("jvm") version "1.3.61"
    kotlin("plugin.serialization") version "1.3.61"
}

group = "org.example"
version = "1.0-SNAPSHOT"

application {
    mainClassName = "MainKt"
}

repositories {
    jcenter()
    mavenCentral()
}

val kotlinSerializationVersion = "0.14.0"
val dl4jVersion = "1.0.0-beta6"

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-runtime:$kotlinSerializationVersion")

    implementation("org.bytedeco:hdf5:1.10.5-1.5.2")
    implementation("org.bytedeco:hdf5-platform:1.10.5-1.5.2")
    implementation("org.nd4j:nd4j-native-platform:$dl4jVersion")
    implementation("org.deeplearning4j:deeplearning4j-core:$dl4jVersion")
    implementation("org.deeplearning4j:deeplearning4j-ui:$dl4jVersion")
//    implementation("ch.qos.logback:logback-classic:1.2.3")

    implementation("com.github.doyaaaaaken:kotlin-csv-jvm:0.7.3")

    testImplementation("org.junit.jupiter:junit-jupiter:5.6.0")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
}

tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "11"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "11"
    }
}