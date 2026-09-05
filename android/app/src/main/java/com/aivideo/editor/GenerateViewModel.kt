package com.aivideo.editor

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

enum class GenerateMode(val label: String) {
    TEXT_TO_VIDEO("Text to video"),
    IMAGE_TO_VIDEO("Image to video"),
}

data class GeneratedResult(
    val path: String,
    val provider: String,
    val model: String,
    val elapsedSeconds: Double,
)

data class GenerateUiState(
    val mode: GenerateMode = GenerateMode.IMAGE_TO_VIDEO,
    val prompt: String = "",
    val seconds: Float = 4f,
    val image: Uri? = null,
    val imageName: String? = null,
    val provider: String? = null,
    val providers: List<VideoProviderDto> = emptyList(),

    val isGenerating: Boolean = false,
    val stage: String = "",
    val progress: Float = 0f,
    val elapsedSeconds: Int = 0,
    val downloadedBytes: Long = 0L,

    val result: GeneratedResult? = null,
    val isSaving: Boolean = false,
    val savedUri: Uri? = null,
    val message: String? = null,
) {
    /** Providers that can do the currently selected job. */
    val usableProviders: List<VideoProviderDto>
        get() = providers.filter {
            it.configured &&
                if (mode == GenerateMode.TEXT_TO_VIDEO) it.textToVideo else it.imageToVideo
        }

    val canGenerate: Boolean
        get() = !isGenerating && when (mode) {
            GenerateMode.TEXT_TO_VIDEO -> prompt.isNotBlank() && usableProviders.isNotEmpty()
            GenerateMode.IMAGE_TO_VIDEO -> image != null
        }

    val elapsedLabel: String
        get() = String.format(Locale.US, "%d:%02d", elapsedSeconds / 60, elapsedSeconds % 60)
}

class GenerateViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(GenerateUiState())
    val state: StateFlow<GenerateUiState> = _state.asStateFlow()

    private var generationJob: Job? = null
    private var timerJob: Job? = null

    private val context get() = getApplication<Application>().applicationContext

    fun setMode(mode: GenerateMode) = _state.update {
        // Keep a provider selection only if it can still do the new job.
        val stillValid = it.providers.firstOrNull { p -> p.key == it.provider }?.let { p ->
            if (mode == GenerateMode.TEXT_TO_VIDEO) p.textToVideo else p.imageToVideo
        } ?: false
        it.copy(mode = mode, provider = if (stillValid) it.provider else null)
    }

    fun setPrompt(value: String) = _state.update { it.copy(prompt = value) }

    fun setSeconds(value: Float) = _state.update { it.copy(seconds = value) }

    fun setProvider(key: String) = _state.update { it.copy(provider = key) }

    fun clearMessage() = _state.update { it.copy(message = null) }

    fun setImage(uri: Uri) {
        viewModelScope.launch {
            val name = withContext(Dispatchers.IO) {
                runCatching { ApiClient.describeUri(context, uri).displayName }.getOrNull()
            }
            _state.update { it.copy(image = uri, imageName = name ?: "selected image") }
        }
    }

    fun refreshProviders() {
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { ApiClient.listVideoProviders(context) }
            result.fold(
                onSuccess = { providers ->
                    _state.update { current ->
                        val usable = providers.filter { p ->
                            p.configured && if (current.mode == GenerateMode.TEXT_TO_VIDEO) {
                                p.textToVideo
                            } else p.imageToVideo
                        }
                        current.copy(
                            providers = providers,
                            provider = current.provider ?: usable.firstOrNull()?.key,
                        )
                    }
                },
                onFailure = { error ->
                    _state.update { it.copy(message = "Cannot reach the backend: ${error.message}") }
                },
            )
        }
    }

    fun generate() {
        val current = _state.value
        if (!current.canGenerate) return

        _state.update {
            it.copy(
                isGenerating = true, stage = "Submitting to ${it.provider ?: "the engine"}",
                progress = 0f, elapsedSeconds = 0, downloadedBytes = 0L,
                result = null, savedUri = null,
            )
        }
        startTimer()

        generationJob = viewModelScope.launch {
            val outcome = withContext(Dispatchers.IO) {
                when (current.mode) {
                    GenerateMode.TEXT_TO_VIDEO -> ApiClient.generateTextToVideo(
                        context = context,
                        prompt = current.prompt,
                        seconds = current.seconds,
                        provider = current.provider,
                        onProgress = ::onDownloadProgress,
                    )
                    GenerateMode.IMAGE_TO_VIDEO -> ApiClient.generateImageToVideo(
                        context = context,
                        image = requireNotNull(current.image),
                        prompt = current.prompt,
                        seconds = current.seconds,
                        provider = current.provider,
                        onProgress = ::onDownloadProgress,
                    )
                }
            }
            stopTimer()
            outcome.fold(
                onSuccess = { clip ->
                    _state.update {
                        it.copy(
                            isGenerating = false, progress = 1f, stage = "Done",
                            result = GeneratedResult(
                                path = clip.file.absolutePath,
                                provider = clip.provider,
                                model = clip.model,
                                elapsedSeconds = clip.elapsedSeconds,
                            ),
                            message = "Clip ready",
                        )
                    }
                },
                onFailure = { error ->
                    _state.update {
                        it.copy(
                            isGenerating = false, stage = "", progress = 0f,
                            message = error.message ?: "Generation failed",
                        )
                    }
                },
            )
        }
    }

    private fun onDownloadProgress(downloaded: Long, total: Long) {
        _state.update {
            it.copy(
                stage = "Receiving the clip",
                downloadedBytes = downloaded,
                progress = if (total > 0) (downloaded.toFloat() / total).coerceIn(0f, 1f) else 0f,
            )
        }
    }

    fun cancel() {
        generationJob?.cancel()
        stopTimer()
        _state.update {
            it.copy(isGenerating = false, stage = "", progress = 0f, message = "Cancelled")
        }
    }

    fun saveToGallery() {
        val result = _state.value.result ?: return
        _state.update { it.copy(isSaving = true) }
        viewModelScope.launch {
            val saved = withContext(Dispatchers.IO) {
                ApiClient.saveFileToGallery(context, File(result.path))
            }
            saved.fold(
                onSuccess = { uri ->
                    _state.update {
                        it.copy(isSaving = false, savedUri = uri,
                                message = "Saved to Movies/Moja AI")
                    }
                },
                onFailure = { error ->
                    _state.update {
                        it.copy(isSaving = false, message = "Save failed: ${error.message}")
                    }
                },
            )
        }
    }

    /** Generation can take minutes; the elapsed clock is the honest feedback. */
    private fun startTimer() {
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            var seconds = 0
            while (isActive) {
                delay(1000)
                seconds += 1
                _state.update { it.copy(elapsedSeconds = seconds) }
            }
        }
    }

    private fun stopTimer() {
        timerJob?.cancel()
        timerJob = null
    }

    override fun onCleared() {
        generationJob?.cancel()
        stopTimer()
        super.onCleared()
    }
}
