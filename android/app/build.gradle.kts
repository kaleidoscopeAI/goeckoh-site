plugins {
    id("com.android.application")
}

android {
    namespace = "com.goeckoh.voicecorrection"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.goeckoh.voicecorrection"
        minSdk = 21
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
    }

    val signingEnvVars = listOf(
        "GOECKOH_KEYSTORE_PATH", "GOECKOH_KEYSTORE_PASSWORD", "GOECKOH_KEY_ALIAS", "GOECKOH_KEY_PASSWORD"
    )
    val hasReleaseSigning = signingEnvVars.all { !System.getenv(it).isNullOrEmpty() }

    signingConfigs {
        create("release") {
            if (hasReleaseSigning) {
                storeFile = file(System.getenv("GOECKOH_KEYSTORE_PATH"))
                storePassword = System.getenv("GOECKOH_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("GOECKOH_KEY_ALIAS")
                keyPassword = System.getenv("GOECKOH_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            if (hasReleaseSigning) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation("com.google.androidbrowserhelper:androidbrowserhelper:2.7.2")
    implementation("androidx.appcompat:appcompat:1.7.0")
}
