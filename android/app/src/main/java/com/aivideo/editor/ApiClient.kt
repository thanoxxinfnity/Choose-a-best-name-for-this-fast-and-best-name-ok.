package com.aivideo.editor

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import okhttp3.Headers
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.BufferedSink
import okio.source
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.concurrent.TimeUnit

// ---------------------------------------------------------------------------
// Wire models (mirrors backend/schemas.py)
// ---------------------------------------------------------------------------

@Serializable
data class HealthDto(
    val status: String = "unknown",
    val version: String = "",
    val ffmpeg: Boolean = false,
    val rembg: Boolean = false,
    val whisper: Boolean = false,
    @SerialName("puter_configured") val puterConfigured: Boolean = false,
    @SerialName("nim_configured") val nimConfigured: Boolean = false,
    @SerialName("active_jobs") val activeJobs: Int = 0,
)

@Serializable
data class ClipInfoDto(
    val filename: String = "",
    val duration: Double = 0.0,
    val width: Int = 0,
    val height: Int = 0,
    val fps: Double = 0.0,
    @SerialName("has_audio") val hasAudio: Boolean = false,
)

/** What the backend read off one clip, before anything is rendered. */
@Serializable
data class AnalysisDto(
    val duration: Double = 0.0,
    val width: Int = 0,
    val height: Int = 0,
    @SerialName("content_type") val contentType: String = "unknown",
    val subjects: List<String> = emptyList(),
    @SerialName("art_style") val artStyle: String = "",
    val mood: String = "",
    val energy: String = "",
    val summary: String = "",
    val recognisable: String = "",
    val bpm: Double = 0.0,
    @SerialName("scene_cuts") val sceneCuts: List<Double> = emptyList(),
    @SerialName("suggested_theme") val suggestedTheme: String = "",
    val theme: String = "",
    // False when the description came from a model whose reading is not
    // trusted. The user has to see this: a confident wrong reading is what
    // puts LEVEL UP on an anime edit.
    @SerialName("vision_trusted") val visionTrusted: Boolean = true,
    @SerialName("vision_model") val visionModel: String = "",
    @SerialName("vision_error") val visionError: String = "",
)

@Serializable
data class ThemeDto(
    val key: String,
    val name: String,
    val description: String = "",
    @SerialName("default_cut") val defaultCut: String = "",
    val shake: String = "",
)

@Serializable
data class GalleryItemDto(
    @SerialName("item_id") val itemId: String,
    val kind: String = "render",
    val filename: String = "",
    @SerialName("size_bytes") val sizeBytes: Long = 0,
    @SerialName("created_iso") val createdIso: String = "",
    val duration: Double = 0.0,
    val width: Int = 0,
    val height: Int = 0,
    val fps: Double = 0.0,
    val title: String = "",
    val theme: String = "",
    val provider: String = "",
    @SerialName("export_preset") val exportPreset: String = "",
    @SerialName("stream_url") val streamUrl: String = "",
    @SerialName("thumbnail_url") val thumbnailUrl: String = "",
)

@Serializable
data class GalleryStatsDto(
    val count: Int = 0,
    val renders: Int = 0,
    val generated: Int = 0,
    @SerialName("total_bytes") val totalBytes: Long = 0,
)

@Serializable
data class GalleryDto(
    val items: List<GalleryItemDto> = emptyList(),
    val stats: GalleryStatsDto = GalleryStatsDto(),
)

@Serializable
data class VideoProviderDto(
    val key: String,
    val label: String,
    @SerialName("text_to_video") val textToVideo: Boolean = false,
    @SerialName("image_to_video") val imageToVideo: Boolean = false,
    val models: List<String> = emptyList(),
    @SerialName("requires_key") val requiresKey: Boolean = true,
    val configured: Boolean = false,
    val notes: String = "",
)

/** A clip produced by the generation endpoints, already on disk. */
data class GeneratedClip(
    val file: File,
    val provider: String,
    val model: String,
    val elapsedSeconds: Double,
)

@Serializable
data class ExportPresetDto(
    val key: String,
    val label: String,
    val width: Int = 0,
    val height: Int = 0,
    val fps: Int = 60,
    val codec: String = "h264",
    val bitrate: String = "",
    val vertical: Boolean = true,
    @SerialName("relative_cost") val relativeCost: Double = 1.0,
)

@Serializable
data class VoiceDto(
    val key: String,
    val label: String,
    @SerialName("voice_id") val voiceId: String = "",
    val language: String = "",
    val deep: Boolean = false,
    val layered: Boolean = false,
    val custom: Boolean = false,
    /** Cloned from a recording rather than shaped out of a stock speaker. */
    val cloned: Boolean = false,
    val note: String = "",
    /**
     * What the reference recording turned out to be. A clone disappoints for
     * reasons that are all knowable before it is used, so they are shown at
     * the moment it is saved rather than guessed at afterwards.
     */
    @SerialName("reference_notes") val referenceNotes: List<String> = emptyList(),
)

@Serializable
data class JobCreatedDto(
    @SerialName("job_id") val jobId: String,
    val stage: String = "queued",
    @SerialName("status_url") val statusUrl: String = "",
    val message: String = "",
)

@Serializable
data class JobStatusDto(
    @SerialName("job_id") val jobId: String,
    val stage: String = "queued",
    val progress: Double = 0.0,
    val message: String = "",
    val error: String? = null,
    val prompt: String = "",
    @SerialName("youtube_url") val youtubeUrl: String? = null,
    val clips: List<ClipInfoDto> = emptyList(),
    val plan: JsonElement? = null,
    @SerialName("output_filename") val outputFilename: String? = null,
    @SerialName("output_size_bytes") val outputSizeBytes: Long? = null,
    @SerialName("export_preset") val exportPreset: String = "1080p60",
    @SerialName("output_width") val outputWidth: Int? = null,
    @SerialName("output_height") val outputHeight: Int? = null,
    @SerialName("output_fps") val outputFps: Double? = null,
    @SerialName("duration_seconds") val durationSeconds: Double? = null,
    val warnings: List<String> = emptyList(),
) {
    val isCompleted: Boolean get() = stage == "completed"
    val isFailed: Boolean get() = stage == "failed" || stage == "cancelled"
    val isTerminal: Boolean get() = isCompleted || isFailed
}

/** A clip chosen in the picker, ready to be uploaded. */
data class SelectedClip(
    val uri: Uri,
    val displayName: String,
    val sizeBytes: Long,
    val mimeType: String,
)

/**
 * All HTTP traffic between the app and the FastAPI backend.
 *
 * Every call reads the backend URL and the API keys from [SecureStore] at call
 * time, so changing them in Settings takes effect immediately. Keys are sent as
 * `X-Puter-Key` / `X-NIM-Key` / `X-YouTube-Token` headers and are never written
 * to disk by the backend.
 *
 * All methods block; call them from `Dispatchers.IO`.
 */
object ApiClient {

    private const val UPLOAD_CHUNK = 128 * 1024

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        explicitNulls = false
        coerceInputValues = true
    }

    private val client: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.MINUTES)
            .writeTimeout(30, TimeUnit.MINUTES) // large multi-clip uploads
            .callTimeout(0, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()
    }

    // ------------------------------------------------------------------ urls
    fun baseUrl(context: Context): String = SecureStore.backendUrl(context)

    fun streamUrl(context: Context, jobId: String): String =
        "${baseUrl(context)}/api/v1/jobs/$jobId/stream"

    fun downloadUrl(context: Context, jobId: String): String =
        "${baseUrl(context)}/api/v1/jobs/$jobId/download"

    private fun authHeaders(context: Context): Headers = Headers.Builder().apply {
        SecureStore.puterKey(context).takeIf { it.isNotBlank() }?.let { add("X-Puter-Key", it) }
        SecureStore.nvidiaNimKey(context).takeIf { it.isNotBlank() }?.let { add("X-NIM-Key", it) }
        SecureStore.youtubeToken(context).takeIf { it.isNotBlank() }?.let { add("X-YouTube-Token", it) }
        SecureStore.videoEndpoint(context).takeIf { it.isNotBlank() }
            ?.let { add("X-Video-Endpoint", it) }
        add("Accept", "application/json")
    }.build()

    // --------------------------------------------------------------- health
    fun checkHealth(context: Context): Result<HealthDto> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/health")
            .headers(authHeaders(context))
            .get()
            .build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(HealthDto.serializer(), body)
        }
    }

    // --------------------------------------------------------------- render
    fun submitRender(
        context: Context,
        clips: List<SelectedClip>,
        prompt: String,
        youtubeUrl: String?,
        targetDurationSeconds: Float?,
        captionsEnabled: Boolean,
        voiceAccent: String,
        maxStickers: Int = 4,
        maxInpaints: Int = 2,
        theme: String = "auto",
        enableVoiceover: Boolean = true,
        enableAnimation: Boolean = false,
        maxAnimations: Int = 1,
        enableIntro: Boolean = false,
        enableOutro: Boolean = false,
        autoSilenceCut: Boolean = false,
        autoBeatSync: Boolean = false,
        autoReframe: Boolean = false,
        matchGrade: Boolean = true,
        enableSfx: Boolean = true,
        enableTransitions: Boolean = true,
        autoHighlight: Boolean = true,
        reviewPlan: Boolean = true,
        exportPreset: String = "1080p60",
        onProgress: (uploadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): Result<JobCreatedDto> = runCatching {
        require(clips.isNotEmpty()) { "Select at least one video clip." }
        require(prompt.isNotBlank()) { "Describe the edit you want." }

        val totalBytes = clips.sumOf { it.sizeBytes.coerceAtLeast(0L) }
        var uploadedSoFar = 0L

        val multipart = MultipartBody.Builder().setType(MultipartBody.FORM).apply {
            addFormDataPart("prompt", prompt.trim())
            youtubeUrl?.takeIf { it.isNotBlank() }?.let { addFormDataPart("youtube_url", it.trim()) }
            targetDurationSeconds?.let { addFormDataPart("target_duration", it.toInt().toString()) }
            addFormDataPart("enable_captions", captionsEnabled.toString())
            addFormDataPart("voice_accent", voiceAccent)
            addFormDataPart("max_stickers", maxStickers.toString())
            addFormDataPart("max_inpaints", maxInpaints.toString())
            addFormDataPart("theme", theme)
            addFormDataPart("enable_voiceover", enableVoiceover.toString())
            addFormDataPart("enable_animation", enableAnimation.toString())
            addFormDataPart("max_animations", maxAnimations.toString())
            addFormDataPart("enable_intro", enableIntro.toString())
            addFormDataPart("enable_outro", enableOutro.toString())
            addFormDataPart("auto_silence_cut", autoSilenceCut.toString())
            addFormDataPart("auto_beat_sync", autoBeatSync.toString())
            addFormDataPart("auto_reframe", autoReframe.toString())
            addFormDataPart("match_grade", matchGrade.toString())
            addFormDataPart("enable_sfx", enableSfx.toString())
            addFormDataPart("enable_transitions", enableTransitions.toString())
            addFormDataPart("auto_highlight", autoHighlight.toString())
            addFormDataPart("review_plan", reviewPlan.toString())
            addFormDataPart("export_preset", exportPreset)

            clips.forEach { clip ->
                val body = UriRequestBody(
                    context = context,
                    uri = clip.uri,
                    mediaType = clip.mimeType.toMediaTypeOrNull() ?: "video/mp4".toMediaType(),
                    declaredLength = clip.sizeBytes,
                ) { deltaBytes ->
                    uploadedSoFar += deltaBytes
                    onProgress(uploadedSoFar, totalBytes)
                }
                addFormDataPart("videos", clip.displayName, body)
            }
        }.build()

        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/render")
            .headers(authHeaders(context))
            .post(multipart)
            .build()

        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(JobCreatedDto.serializer(), body)
        }
    }

    /** Read one clip before committing to a render.
     *
     * The app used to send a prompt written blind: nothing showed what the
     * backend thought the footage was until the finished video came back
     * minutes later, wrong.
     */
    fun analyzeClip(context: Context, clip: SelectedClip): Result<AnalysisDto> = runCatching {
        val body = UriRequestBody(
            context = context,
            uri = clip.uri,
            mediaType = clip.mimeType.toMediaTypeOrNull() ?: "video/mp4".toMediaType(),
            declaredLength = clip.sizeBytes,
            onBytesWritten = {},
        )
        val multipart = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("video", clip.displayName, body)
            .build()
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/analyze")
            .headers(authHeaders(context)).post(multipart).build()
        client.newCall(request).execute().use { response ->
            val text = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, text))
            json.decodeFromString(AnalysisDto.serializer(), text)
        }
    }

    /** Editing modes the backend actually implements. */
    fun listThemes(context: Context): Result<List<ThemeDto>> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/themes")
            .headers(authHeaders(context)).get().build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(kotlinx.serialization.builtins.ListSerializer(ThemeDto.serializer()), body)
        }
    }

    // --------------------------------------------------------------- gallery
    fun listGallery(context: Context, limit: Int = 100): Result<GalleryDto> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/gallery?limit=$limit")
            .headers(authHeaders(context)).get().build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(GalleryDto.serializer(), body)
        }
    }

    fun deleteGalleryItem(context: Context, itemId: String): Result<Unit> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/gallery/$itemId")
            .headers(authHeaders(context)).delete().build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException(errorMessage(response, response.body?.string().orEmpty()))
            }
        }
    }

    /** Download a gallery entry straight into Movies/Moja AI. */
    fun saveGalleryItem(
        context: Context,
        item: GalleryItemDto,
        onProgress: (Long, Long) -> Unit = { _, _ -> },
    ): Result<Uri> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/gallery/${item.itemId}/download")
            .headers(authHeaders(context)).get().build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException(errorMessage(response, response.body?.string().orEmpty()))
            }
            val body = response.body ?: throw IOException("Empty response body")
            writeToGallery(context, item.filename.ifBlank { "${item.itemId}.mp4" },
                           body.contentLength()) { output ->
                copyWithProgress(body.byteStream(), output, body.contentLength(), onProgress)
            }
        }
    }

    /** Absolute URL for a gallery path the DTO returned as relative. */
    fun absolute(context: Context, path: String): String =
        if (path.startsWith("http")) path else "${baseUrl(context)}$path"

    /** Generation backends and which of them are usable right now. */
    fun listVideoProviders(context: Context): Result<List<VideoProviderDto>> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/video/providers")
            .headers(authHeaders(context)).get().build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(
                kotlinx.serialization.builtins.ListSerializer(VideoProviderDto.serializer()), body,
            )
        }
    }

    /** Text-to-video. Streams the finished MP4 into the app cache. */
    fun generateTextToVideo(
        context: Context,
        prompt: String,
        seconds: Float,
        resolution: String = "720x1280",
        provider: String? = null,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): Result<GeneratedClip> = runCatching {
        require(prompt.isNotBlank()) { "Describe the clip you want." }
        val payload = buildJsonObject {
            put("prompt", prompt.trim())
            put("seconds", seconds.toDouble())
            put("resolution", resolution)
            provider?.let { put("provider", it) }
        }
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/video/text-to-video")
            .headers(authHeaders(context))
            .post(json.encodeToString(JsonObject.serializer(), payload)
                .toRequestBody("application/json".toMediaType()))
            .build()
        executeGeneration(context, request, "moja_t2v", onProgress)
    }

    /** Image-to-video. Works without any key via the offline motion provider. */
    fun generateImageToVideo(
        context: Context,
        image: Uri,
        prompt: String,
        seconds: Float,
        motionStrength: Float = 0.7f,
        provider: String? = null,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): Result<GeneratedClip> = runCatching {
        val described = describeUri(context, image)
        val body = MultipartBody.Builder().setType(MultipartBody.FORM).apply {
            addFormDataPart(
                "image", described.displayName,
                UriRequestBody(
                    context, image,
                    described.mimeType.toMediaTypeOrNull() ?: "image/png".toMediaType(),
                    described.sizeBytes,
                ) {},
            )
            addFormDataPart("prompt", prompt.trim())
            addFormDataPart("seconds", seconds.toInt().toString())
            addFormDataPart("motion_strength", motionStrength.toString())
            provider?.let { addFormDataPart("provider_key", it) }
        }.build()

        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/video/image-to-video")
            .headers(authHeaders(context))
            .post(body)
            .build()
        executeGeneration(context, request, "moja_i2v", onProgress)
    }

    private fun executeGeneration(
        context: Context,
        request: Request,
        prefix: String,
        onProgress: (Long, Long) -> Unit,
    ): GeneratedClip {
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException(errorMessage(response, response.body?.string().orEmpty()))
            }
            val body = response.body ?: throw IOException("Empty response body")
            val target = File(context.cacheDir, "$prefix-${System.currentTimeMillis()}.mp4")
            target.outputStream().use { output ->
                copyWithProgress(body.byteStream(), output, body.contentLength(), onProgress)
            }
            if (target.length() == 0L) throw IOException("The provider returned an empty clip.")
            return GeneratedClip(
                file = target,
                provider = response.header("X-Moja-Provider").orEmpty(),
                model = response.header("X-Moja-Model").orEmpty(),
                elapsedSeconds = response.header("X-Moja-Elapsed")?.toDoubleOrNull() ?: 0.0,
            )
        }
    }

    /** Export presets the encoder supports, up to 4K 60fps. */
    fun listExportPresets(context: Context): Result<List<ExportPresetDto>> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/export-presets")
            .headers(authHeaders(context)).get().build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(
                kotlinx.serialization.builtins.ListSerializer(ExportPresetDto.serializer()), body,
            )
        }
    }

    /** Voice profiles, including the deep/dark mysterious narrator. */
    fun listVoices(context: Context): Result<List<VoiceDto>> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/voices")
            .headers(authHeaders(context)).get().build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(kotlinx.serialization.builtins.ListSerializer(VoiceDto.serializer()), body)
        }
    }

    /**
     * Clones a voice from a recording on the device.
     *
     * The recording is the voice: unlike a shaping pack this does not pick a
     * stock speaker, so the sample needs to be a few clean seconds of the
     * person actually talking. The backend reports back what it found in it.
     */
    fun cloneVoice(
        context: Context,
        key: String,
        label: String,
        sample: Uri,
        displayName: String,
        language: String = "en-US",
        quality: Int = 20,
        pitchSemitones: Float? = null,
        note: String = "",
    ): Result<VoiceDto> = runCatching {
        val multipart = MultipartBody.Builder().setType(MultipartBody.FORM).apply {
            addFormDataPart("key", key.trim())
            addFormDataPart("label", label.trim())
            addFormDataPart("language", language)
            addFormDataPart("quality", quality.toString())
            if (note.isNotBlank()) addFormDataPart("note", note.trim())
            pitchSemitones?.let { addFormDataPart("pitch_semitones", it.toString()) }
            addFormDataPart(
                "sample", displayName,
                UriRequestBody(
                    context = context,
                    uri = sample,
                    mediaType = "audio/mpeg".toMediaType(),
                    declaredLength = -1L,
                ) { },
            )
        }.build()

        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/voices/clone")
            .headers(authHeaders(context))
            .post(multipart)
            .build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(VoiceDto.serializer(), body)
        }
    }

    /** Removes a saved voice pack, and the recording behind a cloned one. */
    fun deleteVoice(context: Context, key: String): Result<Unit> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/voices/$key")
            .headers(authHeaders(context))
            .delete()
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException(errorMessage(response, response.body?.string().orEmpty()))
            }
        }
    }

    fun getJob(context: Context, jobId: String): Result<JobStatusDto> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/jobs/$jobId")
            .headers(authHeaders(context))
            .get()
            .build()
        client.newCall(request).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) throw IOException(errorMessage(response, body))
            json.decodeFromString(JobStatusDto.serializer(), body)
        }
    }

    fun cancelJob(context: Context, jobId: String): Result<Unit> = runCatching {
        val request = Request.Builder()
            .url("${baseUrl(context)}/api/v1/jobs/$jobId/cancel")
            .headers(authHeaders(context))
            .post(ByteArray(0).toRequestBody(null))
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful && response.code != 409) {
                throw IOException(errorMessage(response, response.body?.string().orEmpty()))
            }
        }
    }

    // ------------------------------------------------------------- download
    /**
     * Streams the finished high quality MP4 into the device gallery
     * (`Movies/Moja AI/`) and returns its content [Uri].
     */
    fun downloadToGallery(
        context: Context,
        jobId: String,
        fileName: String = "moja_ai_$jobId.mp4",
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ): Result<Uri> = runCatching {
        val request = Request.Builder()
            .url(downloadUrl(context, jobId))
            .headers(authHeaders(context))
            .get()
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException(errorMessage(response, response.body?.string().orEmpty()))
            }
            val body = response.body ?: throw IOException("Empty response body")
            val total = body.contentLength()

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val collection = MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
                    put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                    put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/Moja AI")
                    put(MediaStore.Video.Media.IS_PENDING, 1)
                }
                val resolver = context.contentResolver
                val uri = resolver.insert(collection, values)
                    ?: throw IOException("MediaStore refused to create the file")
                try {
                    resolver.openOutputStream(uri)?.use { output ->
                        copyWithProgress(body.byteStream(), output, total, onProgress)
                    } ?: throw IOException("Could not open the output stream")
                    values.clear()
                    values.put(MediaStore.Video.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                    uri
                } catch (error: Throwable) {
                    resolver.delete(uri, null, null)
                    throw error
                }
            } else {
                @Suppress("DEPRECATION")
                val directory = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
                    "Moja AI",
                )
                if (!directory.exists() && !directory.mkdirs()) {
                    throw IOException("Could not create ${directory.absolutePath}")
                }
                val file = File(directory, fileName)
                FileOutputStream(file).use { output ->
                    copyWithProgress(body.byteStream(), output, total, onProgress)
                }
                android.media.MediaScannerConnection.scanFile(
                    context, arrayOf(file.absolutePath), arrayOf("video/mp4"), null,
                )
                Uri.fromFile(file)
            }
        }
    }

    /** Copy a local file the app produced into Movies/Moja AI. */
    fun saveFileToGallery(
        context: Context,
        file: File,
        fileName: String = file.name,
    ): Result<Uri> = runCatching {
        file.inputStream().use { input ->
            writeToGallery(context, fileName, file.length()) { output ->
                copyWithProgress(input, output, file.length()) { _, _ -> }
            }
        }
    }

    private inline fun writeToGallery(
        context: Context,
        fileName: String,
        @Suppress("UNUSED_PARAMETER") sizeHint: Long,
        write: (OutputStream) -> Unit,
    ): Uri {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val collection = MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
            val values = ContentValues().apply {
                put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
                put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/Moja AI")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            }
            val resolver = context.contentResolver
            val uri = resolver.insert(collection, values)
                ?: throw IOException("MediaStore refused to create the file")
            try {
                resolver.openOutputStream(uri)?.use(write)
                    ?: throw IOException("Could not open the output stream")
                values.clear()
                values.put(MediaStore.Video.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
                return uri
            } catch (error: Throwable) {
                resolver.delete(uri, null, null)
                throw error
            }
        }
        @Suppress("DEPRECATION")
        val directory = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES),
            "Moja AI",
        )
        if (!directory.exists() && !directory.mkdirs()) {
            throw IOException("Could not create ${directory.absolutePath}")
        }
        val target = File(directory, fileName)
        FileOutputStream(target).use(write)
        android.media.MediaScannerConnection.scanFile(
            context, arrayOf(target.absolutePath), arrayOf("video/mp4"), null,
        )
        return Uri.fromFile(target)
    }

    private fun copyWithProgress(
        input: java.io.InputStream,
        output: OutputStream,
        total: Long,
        onProgress: (Long, Long) -> Unit,
    ) {
        val buffer = ByteArray(UPLOAD_CHUNK)
        var copied = 0L
        var lastReport = 0L
        input.use { stream ->
            while (true) {
                val read = stream.read(buffer)
                if (read <= 0) break
                output.write(buffer, 0, read)
                copied += read
                if (copied - lastReport > 512 * 1024) {
                    lastReport = copied
                    onProgress(copied, total)
                }
            }
        }
        output.flush()
        onProgress(copied, if (total > 0) total else copied)
    }

    // ---------------------------------------------------------------- utils
    private fun errorMessage(response: Response, body: String): String {
        val detail = runCatching {
            val element = json.parseToJsonElement(body)
            (element as? kotlinx.serialization.json.JsonObject)
                ?.get("detail")?.toString()?.trim('"')
        }.getOrNull()
        return "HTTP ${response.code}: ${detail ?: body.take(300).ifBlank { response.message }}"
    }

    /** Reads display name, size and MIME type for a picked content:// video. */
    fun describeUri(context: Context, uri: Uri): SelectedClip {
        var name = "clip.mp4"
        var size = -1L
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (cursor.moveToFirst()) {
                if (nameIndex >= 0) name = cursor.getString(nameIndex) ?: name
                if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) size = cursor.getLong(sizeIndex)
            }
        }
        val mime = context.contentResolver.getType(uri)
            ?: MimeTypeMap.getSingleton()
                .getMimeTypeFromExtension(name.substringAfterLast('.', "").lowercase())
            ?: "video/mp4"
        if (!name.contains('.')) {
            val extension = MimeTypeMap.getSingleton().getExtensionFromMimeType(mime) ?: "mp4"
            name = "$name.$extension"
        }
        return SelectedClip(uri = uri, displayName = name, sizeBytes = size, mimeType = mime)
    }
}

/** Streams a `content://` video into the multipart request, reporting progress. */
private class UriRequestBody(
    private val context: Context,
    private val uri: Uri,
    private val mediaType: MediaType,
    private val declaredLength: Long,
    private val onBytesWritten: (Long) -> Unit,
) : RequestBody() {

    override fun contentType(): MediaType = mediaType

    override fun contentLength(): Long = if (declaredLength > 0) declaredLength else -1L

    override fun writeTo(sink: BufferedSink) {
        val stream = context.contentResolver.openInputStream(uri)
            ?: throw IOException("Cannot open $uri")
        stream.source().use { source ->
            val buffer = okio.Buffer()
            while (true) {
                val read = source.read(buffer, 128 * 1024L)
                if (read == -1L) break
                sink.write(buffer, read)
                onBytesWritten(read)
            }
        }
    }
}
