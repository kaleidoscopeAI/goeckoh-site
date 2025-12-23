@rem
@rem Gradle startup script for Windows
@rem Based on Gradle 8.3 distribution
@rem

@if "%DEBUG%"=="" @echo off
set DIR=%~dp0
set APP_BASE_NAME=%~n0
set APP_HOME=%DIR%

set CLASSPATH=%APP_HOME%\\gradle\\wrapper\\gradle-wrapper.jar

@rem Locate Java
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if "%ERRORLEVEL%"=="0" goto init

echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH. 1>&2
exit /b 1

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%\\bin\\java.exe

if exist "%JAVA_EXE%" goto init

echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME% 1>&2
exit /b 1

:init
set DEFAULT_JVM_OPTS=
set JAVA_OPTS=%JAVA_OPTS%
set GRADLE_OPTS=%GRADLE_OPTS%

"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% "-classpath" "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
