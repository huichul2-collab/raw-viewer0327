[app]
title = RawViewer
package.name = rawviewer
package.domain = org.rawviewer

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

version = 1.0

# 의존성: python-for-android 레시피 이름 사용
# opencv → 'opencv' 레시피, numpy/pillow/matplotlib 포함
requirements = python3,kivy==2.3.0,numpy,opencv,pillow,matplotlib

orientation = landscape
fullscreen = 0

# 아이콘/프리스플래시 (없으면 기본값)
# icon.filename = %(source.dir)s/data/icon.png
# presplash.filename = %(source.dir)s/data/presplash.png

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,MANAGE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33
android.archs = arm64-v8a

# Gradle 빌드 캐시
android.gradle_dependencies =

# 추가 Java/Kotlin 소스 없음
android.add_src =

# Release 서명은 별도 설정 (debug 빌드에서는 불필요)
# android.keystore = myapp.keystore
# android.keystore_alias = myapp

[buildozer]
log_level = 2
warn_on_root = 1
