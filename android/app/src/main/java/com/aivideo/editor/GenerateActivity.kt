package com.aivideo.editor

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn as AndroidXOptIn
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Button
import androidx.compose.material3.Card
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
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.AspectRatioFrameLayout
import androidx.media3.ui.PlayerView
import com.aivideo.editor.ui.MojaAITheme
import java.util.Locale

/**
 * AI Studio: generate a clip from a prompt (text-to-video) or from a still
 * (image-to-video), watch it, and save it to the gallery.
 *
 * Every control is wired to a real backend call. Text-to-video needs a
 * generation key and says so plainly when one is missing; image-to-video always
 * has at least the offline motion provider behind it, so it is never a dead
 * button.
 */
class GenerateActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MojaAITheme {
                GenerateScreen(onBack = { finish() }, onShare = ::shareVideo)
            }
        }
    }

    private fun shareVideo(uri: Uri) {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "video/mp4"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Share your clip"))
    }

    companion object {
        fun intent(context: Context): Intent = Intent(context, GenerateActivity::class.java)
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun GenerateScreen(
    onBack: () -> Unit,
    onShare: (Uri) -> Unit,
    viewModel: GenerateViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }

    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let(viewModel::setImage) }

    LaunchedEffect(Unit) { viewModel.refreshProviders() }
    LaunchedEffect(state.message) {
        state.message?.let {
            snackbar.showSnackbar(it)
            viewModel.clearMessage()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = { Text("AI Studio") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
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
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
                GenerateMode.entries.forEachIndexed { index, mode ->
                    SegmentedButton(
                        selected = state.mode == mode,
                        onClick = { viewModel.setMode(mode) },
                        shape = SegmentedButtonDefaults.itemShape(index, GenerateMode.entries.size),
                        enabled = !state.isGenerating,
                    ) { Text(mode.label) }
                }
            }

            if (state.mode == GenerateMode.IMAGE_TO_VIDEO) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text("Source image", style = MaterialTheme.typography.titleSmall)
                        Button(
                            onClick = {
                                pickImage.launch(
                                    PickVisualMediaRequest(
                                        ActivityResultContracts.PickVisualMedia.ImageOnly
                                    )
                                )
                            },
                            enabled = !state.isGenerating,
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(Icons.Filled.Image, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text(if (state.imageName == null) "Pick an image" else "Change image")
                        }
                        state.imageName?.let {
                            Text(
                                it,
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }

            OutlinedTextField(
                value = state.prompt,
                onValueChange = viewModel::setPrompt,
                label = { Text(if (state.mode == GenerateMode.TEXT_TO_VIDEO) "Prompt" else "Motion prompt") },
                placeholder = {
                    Text(
                        if (state.mode == GenerateMode.TEXT_TO_VIDEO) {
                            "e.g. cursed energy swirling around a lone figure, cinematic"
                        } else {
                            "e.g. slow push in, hair moving in the wind"
                        }
                    )
                },
                minLines = 3,
                maxLines = 6,
                enabled = !state.isGenerating,
                modifier = Modifier.fillMaxWidth(),
            )

            Text("Length: ${state.seconds.toInt()}s", style = MaterialTheme.typography.bodyMedium)
            Slider(
                value = state.seconds,
                onValueChange = viewModel::setSeconds,
                valueRange = 2f..10f,
                steps = 7,
                enabled = !state.isGenerating,
            )

            if (state.providers.isNotEmpty()) {
                Text("Engine", style = MaterialTheme.typography.bodyMedium)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    state.usableProviders.forEach { provider ->
                        FilterChip(
                            selected = state.provider == provider.key,
                            onClick = { viewModel.setProvider(provider.key) },
                            label = { Text(provider.label) },
                            enabled = !state.isGenerating,
                        )
                    }
                }
                state.providers.firstOrNull { it.key == state.provider }?.notes
                    ?.takeIf { it.isNotBlank() }?.let {
                        Text(
                            it,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                if (state.usableProviders.none { it.textToVideo } &&
                    state.mode == GenerateMode.TEXT_TO_VIDEO
                ) {
                    Text(
                        "No text-to-video engine is configured. Add a Puter.js key in " +
                            "Settings, or switch to Image to video - that works offline.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }

            Button(
                onClick = viewModel::generate,
                enabled = state.canGenerate,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(54.dp),
            ) {
                if (state.isGenerating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text("Generating - ${state.elapsedLabel}")
                } else {
                    Icon(Icons.Filled.AutoAwesome, contentDescription = null)
                    Spacer(Modifier.width(10.dp))
                    Text("Generate")
                }
            }

            AnimatedVisibility(visible = state.isGenerating) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Text(state.stage, style = MaterialTheme.typography.titleSmall)
                        if (state.progress > 0f) {
                            LinearProgressIndicator(
                                progress = { state.progress },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        } else {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        }
                        Text(
                            "Elapsed ${state.elapsedLabel}" +
                                if (state.downloadedBytes > 0) {
                                    " - ${state.downloadedBytes / 1024} KB received"
                                } else "",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        OutlinedButton(
                            onClick = viewModel::cancel,
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text("Cancel") }
                    }
                }
            }

            state.result?.let { result ->
                HorizontalDivider()
                Text("Result", style = MaterialTheme.typography.titleMedium)
                GeneratedPreview(path = result.path)
                Text(
                    "%s - %s - %.1fs to generate".format(
                        Locale.US, result.provider.ifBlank { "engine" },
                        result.model.ifBlank { "model" }, result.elapsedSeconds,
                    ),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = viewModel::saveToGallery,
                        enabled = !state.isSaving,
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Filled.Download, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(if (state.isSaving) "Saving..." else "Save")
                    }
                    state.savedUri?.let { uri ->
                        OutlinedButton(
                            onClick = { onShare(uri) },
                            modifier = Modifier.weight(1f),
                        ) { Text("Share") }
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

@AndroidXOptIn(UnstableApi::class)
@Composable
private fun GeneratedPreview(path: String) {
    val context = LocalContext.current
    val player = remember(path) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(Uri.fromFile(java.io.File(path))))
            repeatMode = Player.REPEAT_MODE_ONE
            playWhenReady = true
            prepare()
        }
    }
    DisposableEffect(path) { onDispose { player.release() } }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(9f / 16f)
            .clip(RoundedCornerShape(14.dp))
            .background(Color.Black),
        contentAlignment = Alignment.Center,
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
}
