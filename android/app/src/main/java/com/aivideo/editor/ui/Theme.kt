package com.aivideo.editor.ui

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.core.view.WindowCompat

private val Violet = Color(0xFF8B5CF6)
private val VioletDark = Color(0xFF6D28D9)
private val Amber = Color(0xFFFDE047)
private val Ink = Color(0xFF0B0B12)
private val Surface = Color(0xFF15151F)
private val SurfaceVariant = Color(0xFF1F1F2E)

private val DarkColors = darkColorScheme(
    primary = Violet,
    onPrimary = Color.White,
    primaryContainer = VioletDark,
    onPrimaryContainer = Color.White,
    secondary = Amber,
    onSecondary = Ink,
    background = Ink,
    onBackground = Color(0xFFEDEDF5),
    surface = Surface,
    onSurface = Color(0xFFEDEDF5),
    surfaceVariant = SurfaceVariant,
    onSurfaceVariant = Color(0xFFBFBFD4),
    error = Color(0xFFFF6B6B),
)

private val LightColors = lightColorScheme(
    primary = VioletDark,
    onPrimary = Color.White,
    secondary = Color(0xFFB45309),
    background = Color(0xFFF7F7FB),
    surface = Color.White,
    surfaceVariant = Color(0xFFEDEDF5),
)

private val AppTypography = Typography(
    titleLarge = Typography().titleLarge.copy(fontWeight = FontWeight.Bold, fontSize = 22.sp),
    titleMedium = Typography().titleMedium.copy(fontWeight = FontWeight.SemiBold),
    labelLarge = Typography().labelLarge.copy(fontWeight = FontWeight.SemiBold),
)

@Composable
fun AIVideoEditorTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) DarkColors else LightColors
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            // enableEdgeToEdge() owns the system bar backgrounds; we only
            // need to keep the icon contrast in step with the theme.
            val window = (view.context as? Activity)?.window ?: return@SideEffect
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = !darkTheme
                isAppearanceLightNavigationBars = !darkTheme
            }
        }
    }
    MaterialTheme(colorScheme = colors, typography = AppTypography, content = content)
}
