package com.aivideo.editor

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.aivideo.editor.ui.AIVideoEditorTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Encrypted key/value store for the three API credentials required by the
 * backend plus the backend URL itself.
 *
 * Values are held in an [EncryptedSharedPreferences] file (AES256-SIV for the
 * keys, AES256-GCM for the values) whose master key lives in the Android
 * Keystore. If the keystore entry is ever invalidated - a common failure mode
 * after a restore onto a different device - the corrupted file is dropped and
 * recreated once; only if that also fails do we fall back to a plain
 * SharedPreferences file so the app stays usable.
 */
object SecureStore {

    private const val SECURE_FILE = "aivideo_secure_prefs"
    private const val PLAIN_FILE = "aivideo_prefs"

    const val KEY_NVIDIA_NIM = "nvidia_nim_api_key"
    const val KEY_YOUTUBE_TOKEN = "youtube_data_token"
    const val KEY_PUTER = "puter_api_key"
    const val KEY_BACKEND_URL = "backend_url"
    const val KEY_VOICE_ACCENT = "voice_accent"
    const val KEY_CAPTIONS = "captions_enabled"

    @Volatile
    private var cached: SharedPreferences? = null

    @Volatile
    var isEncrypted: Boolean = true
        private set

    fun prefs(context: Context): SharedPreferences {
        cached?.let { return it }
        synchronized(this) {
            cached?.let { return it }
            val appContext = context.applicationContext
            val prefs = createEncrypted(appContext) ?: run {
                isEncrypted = false
                appContext.getSharedPreferences(PLAIN_FILE, Context.MODE_PRIVATE)
            }
            cached = prefs
            return prefs
        }
    }

    private fun createEncrypted(context: Context): SharedPreferences? {
        repeat(2) { attempt ->
            try {
                val masterKey = MasterKey.Builder(context, MasterKey.DEFAULT_MASTER_KEY_ALIAS)
                    .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                    .build()
                return EncryptedSharedPreferences.create(
                    context,
                    SECURE_FILE,
                    masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
                )
            } catch (error: Throwable) {
                if (attempt == 0) {
                    // Drop the unreadable file (and its keyset) and try once more.
                    context.deleteSharedPreferences(SECURE_FILE)
                    context.deleteSharedPreferences("__androidx_security_crypto_encrypted_prefs_key_keyset__")
                    context.deleteSharedPreferences("__androidx_security_crypto_encrypted_prefs_value_keyset__")
                }
            }
        }
        return null
    }

    // ---------------------------------------------------------------- getters
    fun nvidiaNimKey(context: Context): String = read(context, KEY_NVIDIA_NIM)

    fun youtubeToken(context: Context): String = read(context, KEY_YOUTUBE_TOKEN)

    fun puterKey(context: Context): String = read(context, KEY_PUTER)

    fun backendUrl(context: Context): String =
        read(context, KEY_BACKEND_URL).ifBlank { BuildConfig.DEFAULT_BACKEND_URL }.trimEnd('/')

    fun voiceAccent(context: Context): String =
        read(context, KEY_VOICE_ACCENT).ifBlank { "indian_accent" }

    fun captionsEnabled(context: Context): Boolean =
        prefs(context).getBoolean(KEY_CAPTIONS, true)

    private fun read(context: Context, key: String): String =
        prefs(context).getString(key, "").orEmpty().trim()

    // ---------------------------------------------------------------- setters
    fun write(context: Context, key: String, value: String) {
        prefs(context).edit().putString(key, value.trim()).apply()
    }

    fun writeBoolean(context: Context, key: String, value: Boolean) {
        prefs(context).edit().putBoolean(key, value).apply()
    }

    fun clearCredentials(context: Context) {
        prefs(context).edit()
            .remove(KEY_NVIDIA_NIM)
            .remove(KEY_YOUTUBE_TOKEN)
            .remove(KEY_PUTER)
            .apply()
    }

    fun hasRequiredKeys(context: Context): Boolean =
        puterKey(context).isNotBlank() && nvidiaNimKey(context).isNotBlank()
}

/** Settings screen: the only place where API keys are entered or changed. */
class SettingsActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AIVideoEditorTheme {
                SettingsScreen(onBack = { finish() })
            }
        }
    }

    companion object {
        fun intent(context: Context): Intent = Intent(context, SettingsActivity::class.java)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }

    var backendUrl by rememberSaveable { mutableStateOf(SecureStore.backendUrl(context)) }
    var nimKey by rememberSaveable { mutableStateOf(SecureStore.nvidiaNimKey(context)) }
    var youtubeToken by rememberSaveable { mutableStateOf(SecureStore.youtubeToken(context)) }
    var puterKey by rememberSaveable { mutableStateOf(SecureStore.puterKey(context)) }
    var testing by rememberSaveable { mutableStateOf(false) }

    fun persist() {
        SecureStore.write(context, SecureStore.KEY_BACKEND_URL, backendUrl.trimEnd('/'))
        SecureStore.write(context, SecureStore.KEY_NVIDIA_NIM, nimKey)
        SecureStore.write(context, SecureStore.KEY_YOUTUBE_TOKEN, youtubeToken)
        SecureStore.write(context, SecureStore.KEY_PUTER, puterKey)
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = { persist(); onBack() }) {
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
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Credentials", style = MaterialTheme.typography.titleMedium)
                    Text(
                        if (SecureStore.isEncrypted) {
                            "Keys are stored with EncryptedSharedPreferences (AES-256, Android Keystore) " +
                                "and are sent only to the backend URL you configure below."
                        } else {
                            "Warning: the Android Keystore is unavailable on this device, so keys are " +
                                "stored in plain app-private preferences."
                        },
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            OutlinedTextField(
                value = backendUrl,
                onValueChange = { backendUrl = it },
                label = { Text("Backend URL") },
                supportingText = { Text("e.g. http://10.0.2.2:8000 (emulator) or https://api.example.com") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Next),
                modifier = Modifier.fillMaxWidth(),
            )

            SecretField(
                label = "NVIDIA NIM API Key",
                helper = "Used for the Kimi K3 editing orchestrator (nvapi-...)",
                value = nimKey,
                onValueChange = { nimKey = it },
            )

            SecretField(
                label = "Puter.js API Key",
                helper = "Used for TTS (Indian accent), AI stickers and inpainting",
                value = puterKey,
                onValueChange = { puterKey = it },
            )

            SecretField(
                label = "YouTube Data Token",
                helper = "Optional: richer metadata for the reference video",
                value = youtubeToken,
                onValueChange = { youtubeToken = it },
            )

            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Button(
                    onClick = {
                        persist()
                        Toast.makeText(context, "Saved securely", Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier.weight(1f),
                ) { Text("Save") }

                OutlinedButton(
                    enabled = !testing,
                    onClick = {
                        persist()
                        testing = true
                        scope.launch {
                            val result = withContext(Dispatchers.IO) {
                                ApiClient.checkHealth(context)
                            }
                            testing = false
                            snackbar.showSnackbar(
                                result.fold(
                                    onSuccess = { health ->
                                        "Backend OK - v${health.version}, ffmpeg=${health.ffmpeg}, " +
                                            "rembg=${health.rembg}, whisper=${health.whisper}"
                                    },
                                    onFailure = { "Cannot reach backend: ${it.message}" },
                                )
                            )
                        }
                    },
                    modifier = Modifier.weight(1f),
                ) {
                    if (testing) {
                        CircularProgressIndicator(modifier = Modifier.height(18.dp), strokeWidth = 2.dp)
                    } else {
                        Text("Test connection")
                    }
                }
            }

            OutlinedButton(
                onClick = {
                    SecureStore.clearCredentials(context)
                    nimKey = ""; youtubeToken = ""; puterKey = ""
                    Toast.makeText(context, "Credentials cleared", Toast.LENGTH_SHORT).show()
                },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Clear stored credentials") }

            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun SecretField(
    label: String,
    helper: String,
    value: String,
    onValueChange: (String) -> Unit,
) {
    var visible by rememberSaveable { mutableStateOf(false) }
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = { Text(label) },
        supportingText = { Text(helper) },
        singleLine = true,
        visualTransformation = if (visible) VisualTransformation.None else PasswordVisualTransformation(),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Next),
        trailingIcon = {
            IconButton(onClick = { visible = !visible }) {
                Icon(
                    imageVector = if (visible) Icons.Filled.VisibilityOff else Icons.Filled.Visibility,
                    contentDescription = if (visible) "Hide" else "Show",
                )
            }
        },
        modifier = Modifier.fillMaxWidth(),
    )
}
