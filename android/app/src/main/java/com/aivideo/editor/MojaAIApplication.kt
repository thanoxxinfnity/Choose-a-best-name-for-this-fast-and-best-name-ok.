package com.aivideo.editor

import android.app.Application

/**
 * Warms up [SecureStore] off the first frame so the encrypted preferences file
 * (and its Keystore master key) is ready before the editor screen reads it.
 */
class MojaAIApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        SecureStore.prefs(this)
    }
}
