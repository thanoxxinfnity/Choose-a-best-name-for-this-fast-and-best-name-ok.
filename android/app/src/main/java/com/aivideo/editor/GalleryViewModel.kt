package com.aivideo.editor

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.LruCache
import androidx.compose.foundation.Image
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

enum class GalleryFilter(val label: String, val kind: String) {
    ALL("All", ""),
    RENDERS("Edits", "render"),
    GENERATED("Generated", "generated"),
}

data class GalleryUiState(
    val items: List<GalleryItemDto> = emptyList(),
    val stats: GalleryStatsDto = GalleryStatsDto(),
    val filter: GalleryFilter = GalleryFilter.ALL,
    val isLoading: Boolean = false,
    val playing: GalleryItemDto? = null,
    val pendingDelete: GalleryItemDto? = null,
    val isSaving: Boolean = false,
    val savedUri: Uri? = null,
    val message: String? = null,
)

class GalleryViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(GalleryUiState())
    val state: StateFlow<GalleryUiState> = _state.asStateFlow()

    private val context get() = getApplication<Application>().applicationContext

    fun refresh() {
        _state.update { it.copy(isLoading = true) }
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { ApiClient.listGallery(context) }
            result.fold(
                onSuccess = { gallery ->
                    _state.update { current ->
                        current.copy(
                            isLoading = false,
                            stats = gallery.stats,
                            items = gallery.items.filter { item ->
                                current.filter.kind.isEmpty() || item.kind == current.filter.kind
                            },
                        )
                    }
                    allItems = gallery.items
                },
                onFailure = { error ->
                    _state.update {
                        it.copy(isLoading = false, message = "Cannot load the gallery: ${error.message}")
                    }
                },
            )
        }
    }

    private var allItems: List<GalleryItemDto> = emptyList()

    fun setFilter(filter: GalleryFilter) {
        _state.update { current ->
            current.copy(
                filter = filter,
                items = allItems.filter { filter.kind.isEmpty() || it.kind == filter.kind },
            )
        }
    }

    fun open(item: GalleryItemDto) = _state.update { it.copy(playing = item, savedUri = null) }

    fun close() = _state.update { it.copy(playing = null) }

    fun clearMessage() = _state.update { it.copy(message = null) }

    fun save(item: GalleryItemDto) {
        _state.update { it.copy(isSaving = true) }
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) { ApiClient.saveGalleryItem(context, item) }
            result.fold(
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

    fun requestDelete(item: GalleryItemDto) =
        _state.update { it.copy(pendingDelete = item, playing = null) }

    fun cancelDelete() = _state.update { it.copy(pendingDelete = null) }

    fun confirmDelete(item: GalleryItemDto) {
        _state.update { it.copy(pendingDelete = null) }
        viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                ApiClient.deleteGalleryItem(context, item.itemId)
            }
            result.fold(
                onSuccess = {
                    _state.update { it.copy(message = "Deleted") }
                    refresh()
                },
                onFailure = { error ->
                    _state.update { it.copy(message = "Delete failed: ${error.message}") }
                },
            )
        }
    }
}

/**
 * Minimal network image loader for poster frames.
 *
 * The gallery needs exactly one thing - fetch a small JPEG and cache it - so
 * this is a bounded in-memory LruCache over the OkHttp client the app already
 * has, rather than a whole image-loading dependency for one screen.
 */
object ThumbnailLoader {

    private const val CACHE_BYTES = 12 * 1024 * 1024

    private val cache = object : LruCache<String, Bitmap>(CACHE_BYTES) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.byteCount
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    fun cached(url: String): Bitmap? = cache.get(url)

    fun load(url: String): Bitmap? {
        cache.get(url)?.let { return it }
        return runCatching {
            client.newCall(Request.Builder().url(url).get().build()).execute().use { response ->
                if (!response.isSuccessful) return null
                val bytes = response.body?.bytes() ?: return null
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.also { cache.put(url, it) }
            }
        }.getOrNull()
    }
}

@Composable
fun NetworkThumbnail(url: String, modifier: Modifier = Modifier) {
    var bitmap by remember(url) { mutableStateOf(ThumbnailLoader.cached(url)) }
    LaunchedEffect(url) {
        if (bitmap == null) {
            bitmap = withContext(Dispatchers.IO) { ThumbnailLoader.load(url) }
        }
    }
    bitmap?.let {
        Image(
            bitmap = it.asImageBitmap(),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = modifier,
        )
    }
}
