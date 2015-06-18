@echo off
cls

rem .paket\paket.bootstrapper.exe
rem if errorlevel 1 (
rem   exit /b %errorlevel%
rem )

.paket\paket.exe restore
if errorlevel 1 (
  exit /b %errorlevel%
)

"packages\FAKE\tools\Fake.exe" build.fsx Default %*