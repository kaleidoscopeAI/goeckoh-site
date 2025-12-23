npm init -y
npm i @capacitor/core @capacitor/android @capacitor/ios
npx cap init goeckoh com.goeckoh.mirror
cp -r voice-bubble/www/ # Your exact demo
npx cap add android
npx cap sync
# Edit AndroidManifest.xml: <uses-permission android:name="android.permission.RECORD_AUDIO"/>
npx cap open android  # Build APK in Android Studio
