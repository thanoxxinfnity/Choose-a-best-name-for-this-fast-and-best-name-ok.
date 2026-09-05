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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Everything the editor screen renders. */
data class EditorUiState(
    val clips: List<SelectedClip> = emptyList(),
    val prompt: String = "",
    val youtubeUrl: String = "",
    val targetDurationSeconds: Float = 30f,
    val captionsEnabled: Boolean = true,
    val voiceAccent: String = "indian_accent",
    val maxStickers: Int = 4,
    val maxInpaints: Int = 2,
    val theme: String = "auto",
    val enableVoiceover: Boolean = true,
    val enableAnimation: Boolean = false,
    val maxAnimations: Int = 1,
    val enableIntro: Boolean = false,
    val enableOutro: Boolean = false,
    val autoSilenceCut: Boolean = false,
    val autoBeatSync: Boolean = false,
    val autoReframe: Boolean = false,
    val enableSfx: Boolean = true,
    val exportPreset: String = "1080p60",
    val exportPresets: List<ExportPresetDto> = emptyList(),
    val themes: List<ThemeDto> = emptyList(),
    val voices: List<VoiceDto> = emptyList(),

    val isUploading: Boolean = false,
    val uploadedBytes: Long = 0L,
    val totalBytes: Long = 0L,

    val job: JobStatusDto? = null,
    val isPolling: Boolean = false,

    val isDownloading: Boolean = false,
    val downloadedBytes: Long = 0L,
    val downloadTotalBytes: Long = 0L,
    val savedUri: Uri? = null,

    val health: HealthDto? = null,
    val statusMessage: String? = null,
    val errorMessage: String? = null,
) {
    val canSubmit: Boolean
        get() = clips.isNotEmpty() && prompt.isNotBlank() && !isUploading && !isPolling

    val uploadFraction: Float
        get() = if (totalBytes > 0) (uploadedBytes.toFloat() / totalBytes).coerceIn(0f, 1f) else 0f

    val downloadFraction: Float
        get() = if (downloadTotalBytes > 0) {
            (downloadedBytes.toFloat() / downloadTotalBytes).coerceIn(0f, 1f)
        } else 0f

    val isRenderComplete: Boolean get() = job?.isCompleted == true
}

class EditorViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(
        EditorUiState(
            captionsEnabled = SecureStore.captionsEnabled(application),
            voiceAccent = SecureStore.voiceAccent(application),
            enableVoiceover = SecureStore.voiceoverEnabled(application),
            exportPreset = SecureStore.exportPreset(application),
        )
    )
    val state: StateFlow<EditorUiState> = _state.asStateFlow()

    private var pollJob: Job? = null

    private val context get() = getApplication<Application>().applicationContext

    // ------------------------------------------------------------- editing
    fun addClips(uris: List<Uri>) {
        if (uris.isEmpty()) return
        viewModelScope.launch {
            val described = withContext(Dispatchers.IO) {
                uris.mapNotNull { uri -> runCatching { ApiClient.describeUri(context, uri) }.getOrNull() }
            }
            _state.update { current ->
                val existing = current.clips.map { it.uri }.toSet()
                current.copy(clips = current.clips + described.filterNot { it.uri in existing })
            }
        }
    }

    fun removeClip(clip: SelectedClip) {
        _state.update { it.copy(clips = it.clips.filterNot { candidate -> candidate.uri == clip.uri }) }
    }

    fun moveClip(from: Int, to: Int) {
        _state.update { current ->
            if (from !in current.clips.indices || to !in current.clips.indices) return@update current
            val reordered = current.clips.toMutableList()
            reordered.add(to, reordered.removeAt(from))
            current.copy(clips = reordered)
        }
    }

    fun setPrompt(value: String) = _state.update { it.copy(prompt = value) }

    fun setYoutubeUrl(value: String) = _state.update { it.copy(youtubeUrl = value) }

    fun setTargetDuration(value: Float) = _state.update { it.copy(targetDurationSeconds = value) }

    fun setCaptionsEnabled(value: Boolean) {
        SecureStore.writeBoolean(context, SecureStore.KEY_CAPTIONS, value)
        _state.update { it.copy(captionsEnabled = value) }
    }

    fun setVoiceAccent(value: String) {
        SecureStore.write(context, SecureStore.KEY_VOICE_ACCENT, value)
        _state.update { it.copy(voiceAccent = value) }
    }

    fun setTheme(value: String) = _state.update { it.copy(theme = value) }

    fun setEnableVoiceover(value: Boolean) {
        SecureStore.writeBoolean(context, SecureStore.KEY_VOICEOVER, value)
        _state.update { it.copy(enableVoiceover = value) }
    }

    fun setEnableAnimation(value: Boolean) = _state.update { it.copy(enableAnimation = value) }

    fun setMaxAnimations(value: Int) = _state.update { it.copy(maxAnimations = value) }

    fun setEnableIntro(value: Boolean) = _state.update { it.copy(enableIntro = value) }

    fun setEnableOutro(value: Boolean) = _state.update { it.copy(enableOutro = value) }

    fun setAutoSilenceCut(value: Boolean) = _state.update { it.copy(autoSilenceCut = value) }

    fun setAutoBeatSync(value: Boolean) = _state.update { it.copy(autoBeatSync = value) }

    fun setAutoReframe(value: Boolean) = _state.update { it.copy(autoReframe = value) }
    fun setEnableSfx(value: Boolean) = _state.update { it.copy(enableSfx = value) }

    fun setExportPreset(value: String) {
        SecureStore.write(context, SecureStore.KEY_EXPORT_PRESET, value)
        _state.update { it.copy(exportPreset = value) }
    }

    fun setMaxStickers(value: Int) = _state.update { it.copy(maxStickers = value) }

    fun setMaxInpaints(value: Int) = _state.update { it.copy(maxInpaints = value) }

    fun dismissMessages() = _state.update { it.copy(statusMessage = null, errorMessage = null) }

    // -------------------------------------------------------------- health
    fun refreshHealth() {
        viewModelScope.launch {
            val health = withContext(Dispatchers.IO) { ApiClient.checkHealth(context) }
            health.fold(
                onSuccess = { value -> _state.update { it.copy(health = value) } },
                onFailure = { _state.update { it.copy(health = null) } },
            )
            // The theme and voice lists come from the backend so the UI can
            // never offer a mode the renderer does not implement.
            val themes = withContext(Dispatchers.IO) { ApiClient.listThemes(context) }
            themes.onSuccess { value -> _state.update { it.copy(themes = value) } }
            val voices = withContext(Dispatchers.IO) { ApiClient.listVoices(context) }
            voices.onSuccess { value -> _state.update { it.copy(voices = value) } }
            val presets = withContext(Dispatchers.IO) { ApiClient.listExportPresets(context) }
            presets.onSuccess { value -> _state.update { it.copy(exportPresets = value) } }
        }
    }

    // -------------------------------------------------------------- render
    fun startRender() {
        val current = _state.value
        if (!current.canSubmit) return
        if (SecureStore.puterKey(context).isBlank()) {
            _state.update {
                it.copy(errorMessage = "Add your Puter.js API key in Settings - TTS, stickers and inpainting need it.")
            }
        }

        _state.update {
            it.copy(
                isUploading = true,
                uploadedBytes = 0L,
                totalBytes = it.clips.sumOf { clip -> clip.sizeBytes.coerceAtLeast(0L) },
                job = null,
                savedUri = null,
                statusMessage = "Uploading ${it.clips.size} clip(s)...",
            )
        }

        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                ApiClient.submitRender(
                    context = context,
                    clips = current.clips,
                    prompt = current.prompt,
                    youtubeUrl = current.youtubeUrl.ifBlank { null },
                    targetDurationSeconds = current.targetDurationSeconds,
                    captionsEnabled = current.captionsEnabled,
                    voiceAccent = current.voiceAccent,
                    maxStickers = current.maxStickers,
                    maxInpaints = current.maxInpaints,
                    theme = current.theme,
                    enableVoiceover = current.enableVoiceover,
                    enableAnimation = current.enableAnimation,
                    maxAnimations = current.maxAnimations,
                    enableIntro = current.enableIntro,
                    enableOutro = current.enableOutro,
                    autoSilenceCut = current.autoSilenceCut,
                    autoBeatSync = current.autoBeatSync,
                    autoReframe = current.autoReframe,
                    enableSfx = current.enableSfx,
                    exportPreset = current.exportPreset,
                ) { uploaded, total ->
                    _state.update { it.copy(uploadedBytes = uploaded, totalBytes = total) }
                }
            }

            result.fold(
                onSuccess = { created ->
                    _state.update {
                        it.copy(
                            isUploading = false,
                            statusMessage = created.message,
                            job = JobStatusDto(jobId = created.jobId, stage = created.stage),
                        )
                    }
                    pollJob(created.jobId)
                },
                onFailure = { error ->
                    _state.update {
                        it.copy(isUploading = false, errorMessage = error.message ?: "Upload failed")
                    }
                },
            )
        }
    }

    private fun pollJob(jobId: String) {
        pollJob?.cancel()
        _state.update { it.copy(isPolling = true) }
        pollJob = viewModelScope.launch {
            var failures = 0
            while (true) {
                val result = withContext(Dispatchers.IO) { ApiClient.getJob(context, jobId) }
                result.fold(
                    onSuccess = { status ->
                        failures = 0
                        _state.update { it.copy(job = status) }
                        if (status.isTerminal) {
                            _state.update {
                                it.copy(
                                    isPolling = false,
                                    statusMessage = if (status.isCompleted) "Render complete" else null,
                                    errorMessage = status.error.takeIf { _ -> status.isFailed },
                                )
                            }
                            return@launch
                        }
                    },
                    onFailure = { error ->
                        failures += 1
                        if (failures >= 5) {
                            _state.update {
                                it.copy(
                                    isPolling = false,
                                    errorMessage = "Lost contact with the backend: ${error.message}",
                                )
                            }
                            return@launch
                        }
                    },
                )
                delay(2000)
            }
        }
    }

    fun cancelRender() {
        val jobId = _state.value.job?.jobId ?: return
        pollJob?.cancel()
        viewModelScope.launch {
            withContext(Dispatchers.IO) { ApiClient.cancelJob(context, jobId) }
            _state.update { it.copy(isPolling = false, statusMessage = "Render cancelled") }
        }
    }

    fun resumeJob(jobId: String) {
        if (jobId.isBlank()) return
        pollJob(jobId)
    }

    // ------------------------------------------------------------ download
    fun downloadFinishedVideo() {
        val jobId = _state.value.job?.jobId ?: return
        if (_state.value.isDownloading) return
        _state.update { it.copy(isDownloading = true, downloadedBytes = 0L, downloadTotalBytes = 0L) }

        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                ApiClient.downloadToGallery(context, jobId) { downloaded, total ->
                    _state.update { it.copy(downloadedBytes = downloaded, downloadTotalBytes = total) }
                }
            }
            result.fold(
                onSuccess = { uri ->
                    _state.update {
                        it.copy(
                            isDownloading = false,
                            savedUri = uri,
                            statusMessage = "Saved to Movies/Moja AI",
                        )
                    }
                },
                onFailure = { error ->
                    _state.update {
                        it.copy(isDownloading = false, errorMessage = "Download failed: ${error.message}")
                    }
                },
            )
        }
    }

    fun previewUrl(): String? =
        _state.value.job?.takeIf { it.isCompleted }?.let { ApiClient.streamUrl(context, it.jobId) }

    override fun onCleared() {
        pollJob?.cancel()
        super.onCleared()
    }
}
