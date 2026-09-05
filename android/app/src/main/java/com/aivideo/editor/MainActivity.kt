package com.aivideo.editor

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.MovieCreation
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.VideoLibrary
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.annotation.OptIn as AndroidXOptIn
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.AspectRatioFrameLayout
import androidx.media3.ui.PlayerView
import com.aivideo.editor.ui.MojaAITheme
import java.util.Locale

/**
 * The editor: pick clips, describe the edit, optionally point at a YouTube
 * reference, render on the backend, watch the result in ExoPlayer and save the
 * high quality 1080x1920 60fps MP4 to the gallery.
 */
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MojaAITheme {
                EditorScreen(
                    onOpenSettings = { startActivity(SettingsActivity.intent(this)) },
                    onOpenStudio = { startActivity(GenerateActivity.intent(this)) },
                    onOpenGallery = { startActivity(GalleryActivity.intent(this)) },
                    onShare = { uri -> shareVideo(uri) },
                )
            }
        }
    }

    private fun shareVideo(uri: Uri) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "video/mp4"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Share your edit"))
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun EditorScreen(
    onOpenSettings: () -> Unit,
    onOpenStudio: () -> Unit,
    onOpenGallery: () -> Unit,
    onShare: (Uri) -> Unit,
    viewModel: EditorViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    val context = LocalContext.current

    val pickVideos = rememberLauncherForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(maxItems = 10)
    ) { uris -> viewModel.addClips(uris) }

    // Fallback picker for devices without the system photo picker.
    val openDocuments = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments()
    ) { uris ->
        uris.forEach { uri ->
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
        }
        viewModel.addClips(uris)
    }

    LaunchedEffect(Unit) { viewModel.refreshHealth() }

    LaunchedEffect(state.statusMessage, state.errorMessage) {
        val message = state.errorMessage ?: state.statusMessage
        if (message != null) {
            snackbar.showSnackbar(message)
            viewModel.dismissMessages()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Moja AI")
                        Text(
                            state.health?.let { "backend v${it.version} - 1080x1920 @ 60fps" }
                                ?: "backend unreachable - check Settings",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onOpenGallery) {
                        Icon(Icons.Filled.PhotoLibrary, contentDescription = "Gallery")
                    }
                    IconButton(onClick = onOpenStudio) {
                        Icon(Icons.Filled.AutoAwesome, contentDescription = "AI Studio")
                    }
                    IconButton(onClick = onOpenSettings) {
                        Icon(Icons.Filled.Settings, contentDescription = "Settings")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            // ------------------------------------------------------- clips --
            SectionCard(title = "1. Source clips") {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            pickVideos.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                            )
                        },
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Filled.VideoLibrary, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Pick videos")
                    }
                    OutlinedButton(
                        onClick = { openDocuments.launch(arrayOf("video/*")) },
                        modifier = Modifier.weight(1f),
                    ) { Text("Browse files") }
                }

                if (state.clips.isEmpty()) {
                    Text(
                        "No clips yet. Pick one or more videos - they are cut into a fast paced vertical edit.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    state.clips.forEachIndexed { index, clip ->
                        ClipRow(
                            index = index,
                            clip = clip,
                            onRemove = { viewModel.removeClip(clip) },
                            onMoveUp = { if (index > 0) viewModel.moveClip(index, index - 1) },
                            onMoveDown = {
                                if (index < state.clips.lastIndex) viewModel.moveClip(index, index + 1)
                            },
                        )
                    }
                }
            }

            // ------------------------------------------------------ prompt --
            SectionCard(title = "2. Tell the AI what to make") {
                OutlinedTextField(
                    value = state.prompt,
                    onValueChange = viewModel::setPrompt,
                    label = { Text("Edit prompt") },
                    placeholder = {
                        Text("e.g. Fast paced Hinglish reel with an Indian voiceover, glowing subscribe sticker and a stormy sky")
                    },
                    minLines = 3,
                    maxLines = 6,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = state.youtubeUrl,
                    onValueChange = viewModel::setYoutubeUrl,
                    label = { Text("YouTube reference URL (optional)") },
                    placeholder = { Text("https://youtube.com/watch?v=...") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Uri, imeAction = ImeAction.Done,
                    ),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // ----------------------------------------------------- options --
            SectionCard(title = "3. Options") {
                ToggleRow(
                    title = "Word level captions",
                    subtitle = "Faster-Whisper animated subtitles",
                    checked = state.captionsEnabled,
                    onCheckedChange = viewModel::setCaptionsEnabled,
                )

                Text("Editing mode", style = MaterialTheme.typography.bodyMedium)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(
                        selected = state.theme == "auto",
                        onClick = { viewModel.setTheme("auto") },
                        label = { Text("Auto") },
                    )
                    state.themes.forEach { theme ->
                        FilterChip(
                            selected = state.theme == theme.key,
                            onClick = { viewModel.setTheme(theme.key) },
                            label = { Text(theme.name) },
                        )
                    }
                }
                state.themes.firstOrNull { it.key == state.theme }?.let { theme ->
                    Text(
                        theme.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } ?: if (state.theme == "auto") {
                    Text(
                        "Moja AI picks the mode from what it sees in your footage.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else Unit

                ToggleRow(
                    title = "AI voiceover",
                    subtitle = if (state.enableVoiceover) {
                        "Puter TTS narration, timed to the cut"
                    } else {
                        "Off - the edit keeps only its original audio"
                    },
                    checked = state.enableVoiceover,
                    onCheckedChange = viewModel::setEnableVoiceover,
                )

                if (state.enableVoiceover) {
                    Text("Voiceover voice", style = MaterialTheme.typography.bodyMedium)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        val voices = state.voices.ifEmpty {
                            VOICE_ACCENTS.map { (key, label) -> VoiceDto(key = key, label = label) }
                        }
                        voices.forEach { voice ->
                            FilterChip(
                                selected = state.voiceAccent == voice.key,
                                onClick = { viewModel.setVoiceAccent(voice.key) },
                                label = { Text(voice.label) },
                                leadingIcon = if (voice.deep) {
                                    { Text("\uD83C\uDF11") }
                                } else null,
                            )
                        }
                    }
                    state.voices.firstOrNull { it.key == state.voiceAccent }?.let { voice ->
                        Text(
                            "${voice.label} - ${voice.voiceId} (${voice.language})",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                ToggleRow(
                    title = "AI intro",
                    subtitle = "Animates the opening frame with wan2.2 image-to-video",
                    checked = state.enableIntro,
                    onCheckedChange = viewModel::setEnableIntro,
                )
                ToggleRow(
                    title = "AI outro",
                    subtitle = "Generated closing sequence with a follow card",
                    checked = state.enableOutro,
                    onCheckedChange = viewModel::setEnableOutro,
                )
                ToggleRow(
                    title = "Keyframe animation",
                    subtitle = "Animates a key moment and splices it back into the cut",
                    checked = state.enableAnimation,
                    onCheckedChange = viewModel::setEnableAnimation,
                )
                if (state.enableAnimation) {
                    Text(
                        "Animated moments: ${state.maxAnimations}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Slider(
                        value = state.maxAnimations.toFloat(),
                        onValueChange = { viewModel.setMaxAnimations(it.toInt()) },
                        valueRange = 1f..4f,
                        steps = 2,
                    )
                }

                HorizontalDivider()
                Text("Export quality", style = MaterialTheme.typography.titleSmall)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    val presets = state.exportPresets.ifEmpty {
                        listOf(ExportPresetDto(key = "1080p60", label = "1080p 60fps"))
                    }
                    presets.forEach { preset ->
                        FilterChip(
                            selected = state.exportPreset == preset.key,
                            onClick = { viewModel.setExportPreset(preset.key) },
                            label = { Text("${preset.width}x${preset.height} ${preset.fps}") },
                        )
                    }
                }
                state.exportPresets.firstOrNull { it.key == state.exportPreset }?.let { preset ->
                    Text(
                        "${preset.label} - ${preset.codec.uppercase()} @ ${preset.bitrate}" +
                            if (preset.relativeCost > 1.5) {
                                "  (~${preset.relativeCost}x render time)"
                            } else "",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (preset.relativeCost > 1.5) {
                            MaterialTheme.colorScheme.secondary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                    )
                }

                HorizontalDivider()
                Text("Auto edit", style = MaterialTheme.typography.titleSmall)
                ToggleRow(
                    title = "Auto silence-cut",
                    subtitle = "Drops dead air detected in the audio",
                    checked = state.autoSilenceCut,
                    onCheckedChange = viewModel::setAutoSilenceCut,
                )
                ToggleRow(
                    title = "Auto beat-sync",
                    subtitle = "Snaps every cut onto the nearest musical onset",
                    checked = state.autoBeatSync,
                    onCheckedChange = viewModel::setAutoBeatSync,
                )
                ToggleRow(
                    title = "Auto-reframe",
                    subtitle = "Tracks the subject so the 9:16 crop follows it",
                    checked = state.autoReframe,
                    onCheckedChange = viewModel::setAutoReframe,
                )
                ToggleRow(
                    title = "Find the moment",
                    subtitle = "Searches a long video or film and edits only its best window",
                    checked = state.autoHighlight,
                    onCheckedChange = viewModel::setAutoHighlight,
                )
                ToggleRow(
                    title = "Sound effects",
                    subtitle = "Whooshes on the cuts, impacts on the hits, a riser into the payoff",
                    checked = state.enableSfx,
                    onCheckedChange = viewModel::setEnableSfx,
                )
                HorizontalDivider()

                Text(
                    "Target length: ${state.targetDurationSeconds.toInt()}s",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Slider(
                    value = state.targetDurationSeconds,
                    onValueChange = viewModel::setTargetDuration,
                    valueRange = 10f..120f,
                    steps = 21,
                )
            }

            // ------------------------------------------------------ render --
            Button(
                onClick = viewModel::startRender,
                enabled = state.canSubmit,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
            ) {
                Icon(Icons.Filled.MovieCreation, contentDescription = null)
                Spacer(Modifier.width(10.dp))
                Text("Generate AI edit")
            }

            AnimatedVisibility(visible = state.isUploading) {
                ProgressCard(
                    title = "Uploading clips",
                    subtitle = "${formatBytes(state.uploadedBytes)} / ${formatBytes(state.totalBytes)}",
                    fraction = state.uploadFraction,
                )
            }

            state.job?.let { job ->
                AnimatedVisibility(visible = !state.isUploading) {
                    JobCard(
                        job = job,
                        onCancel = viewModel::cancelRender,
                        showCancel = state.isPolling,
                    )
                }
            }

            // ----------------------------------------------------- preview --
            if (state.isRenderComplete) {
                val previewUrl = viewModel.previewUrl()
                if (previewUrl != null) {
                    SectionCard(title = "4. Preview") {
                        VideoPreview(url = previewUrl)
                        state.job?.durationSeconds?.let { duration ->
                            val job = state.job
                            val geometry = if ((job?.outputWidth ?: 0) > 0) {
                                "${job?.outputWidth}x${job?.outputHeight} @ " +
                                    "${(job?.outputFps ?: 0.0).toInt()}fps"
                            } else "rendering"
                            Text(
                                "%.1fs - %s - %s".format(
                                    Locale.US, duration,
                                    formatBytes(job?.outputSizeBytes ?: 0L), geometry,
                                ),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }

                    SectionCard(title = "5. Export") {
                        Button(
                            onClick = viewModel::downloadFinishedVideo,
                            enabled = !state.isDownloading,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(52.dp),
                        ) {
                            Icon(Icons.Filled.Download, contentDescription = null)
                            Spacer(Modifier.width(10.dp))
                            Text(
                                if (state.isDownloading) {
                                    "Downloading ${(state.downloadFraction * 100).toInt()}%"
                                } else {
                                    "Download high quality MP4"
                                }
                            )
                        }
                        if (state.isDownloading) {
                            LinearProgressIndicator(
                                progress = { state.downloadFraction },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                        state.savedUri?.let { uri ->
                            OutlinedButton(
                                onClick = { onShare(uri) },
                                modifier = Modifier.fillMaxWidth(),
                            ) { Text("Share saved video") }
                        }
                    }
                }
            }

            if (state.job?.warnings?.isNotEmpty() == true) {
                SectionCard(title = "Notes from the pipeline") {
                    state.job?.warnings?.forEach { warning ->
                        Text("- $warning", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            Spacer(Modifier.height(28.dp))
        }
    }
}

/** Offline fallback list; the live one comes from GET /api/v1/voices. */
private val VOICE_ACCENTS = listOf(
    "indian_accent" to "Indian (F)",
    "indian_accent_male" to "Indian (M)",
    "hinglish" to "Hinglish",
    "deep_dark" to "Deep dark",
)

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

@Composable
private fun ToggleRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyMedium)
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun ClipRow(
    index: Int,
    clip: SelectedClip,
    onRemove: () -> Unit,
    onMoveUp: () -> Unit,
    onMoveDown: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            "${index + 1}",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.width(24.dp),
        )
        Column(Modifier.weight(1f)) {
            Text(
                clip.displayName,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                formatBytes(clip.sizeBytes),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text("^", modifier = Modifier
            .clickable(onClick = onMoveUp)
            .padding(horizontal = 8.dp, vertical = 4.dp))
        Text("v", modifier = Modifier
            .clickable(onClick = onMoveDown)
            .padding(horizontal = 8.dp, vertical = 4.dp))
        IconButton(onClick = onRemove) {
            Icon(Icons.Filled.Close, contentDescription = "Remove clip")
        }
    }
}

@Composable
private fun ProgressCard(title: String, subtitle: String, fraction: Float) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            LinearProgressIndicator(progress = { fraction }, modifier = Modifier.fillMaxWidth())
            Text(
                subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun JobCard(job: JobStatusDto, onCancel: () -> Unit, showCancel: Boolean) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(stageLabel(job.stage), style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.weight(1f))
                if (!job.isTerminal) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                }
            }
            LinearProgressIndicator(
                progress = { job.progress.toFloat().coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
            Text(
                job.message.ifBlank { "Working..." },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            job.error?.let { error ->
                Text(error, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
            }
            AssistChip(
                onClick = {},
                label = { Text("job ${job.jobId}", fontFamily = FontFamily.Monospace) },
            )
            if (showCancel) {
                OutlinedButton(onClick = onCancel, modifier = Modifier.fillMaxWidth()) {
                    Text("Cancel render")
                }
            }
        }
    }
}

private fun stageLabel(stage: String): String = when (stage) {
    "queued" -> "Queued"
    "analyzing" -> "Analysing what is in the footage"
    "animating" -> "Generating AI animation"
    "planning" -> "Kimi K3 is planning the edit"
    "tts" -> "Puter.js TTS (Indian accent)"
    "stickers" -> "Generating AI stickers"
    "inpainting" -> "Puter.js inpainting frames"
    "rendering" -> "Rendering timeline"
    "captions" -> "Word level captions"
    "encoding" -> "Encoding 1080x1920 60fps"
    "completed" -> "Completed"
    "failed" -> "Failed"
    "cancelled" -> "Cancelled"
    else -> stage.replaceFirstChar { it.uppercase() }
}

/** Native ExoPlayer surface streaming the rendered MP4 (Range enabled). */
@AndroidXOptIn(UnstableApi::class)
@Composable
private fun VideoPreview(url: String) {
    val context = LocalContext.current
    val player = remember(url) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(url))
            repeatMode = Player.REPEAT_MODE_ONE
            playWhenReady = false
            prepare()
        }
    }

    DisposableEffect(url) {
        onDispose { player.release() }
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(9f / 16f)
            .clip(RoundedCornerShape(14.dp))
            .background(Color.Black),
    ) {
        AndroidView(
            factory = { viewContext ->
                PlayerView(viewContext).apply {
                    this.player = player
                    useController = true
                    resizeMode = AspectRatioFrameLayout.RESIZE_MODE_FIT
                    setShowNextButton(false)
                    setShowPreviousButton(false)
                }
            },
            modifier = Modifier.fillMaxSize(),
        )
    }

    OutlinedButton(
        onClick = { player.playWhenReady = !player.playWhenReady },
        modifier = Modifier.fillMaxWidth(),
    ) {
        Icon(Icons.Filled.PlayArrow, contentDescription = null)
        Spacer(Modifier.width(8.dp))
        Text("Play / pause")
    }
}

private fun formatBytes(bytes: Long): String {
    if (bytes <= 0) return "-"
    val units = arrayOf("B", "KB", "MB", "GB")
    var value = bytes.toDouble()
    var unit = 0
    while (value >= 1024 && unit < units.lastIndex) {
        value /= 1024
        unit += 1
    }
    return String.format(Locale.US, if (unit == 0) "%.0f %s" else "%.1f %s", value, units[unit])
}
