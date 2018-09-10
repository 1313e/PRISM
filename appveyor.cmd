@ECHO OFF

SET COMMAND_TO_RUN=%*
SET WIN_SDK_ROOT=C:\Program Files\Microsoft SDKs\Windows
SET WIN_WDK=c:\Program Files (x86)\Windows Kits\10\Include\wdf

:: Extract the major and minor versions, and allow for the minor version to be
:: more than 9.  This requires the version number to have two dots in it.
SET MAJOR_PYTHON_VERSION=%PYTHON_VERSION:~0,1%
IF "%PYTHON_VERSION:~3,1%" == "." (
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,1%
) ELSE (
    SET MINOR_PYTHON_VERSION=%PYTHON_VERSION:~2,2%
)

:: Based on the Python version, determine what SDK version to use, and whether
:: to set the SDK for 64-bit.
IF %MAJOR_PYTHON_VERSION% == 2 (
    SET WINDOWS_SDK_VERSION="v7.0"
    SET SET_SDK_64=Y
) ELSE (
    IF %MAJOR_PYTHON_VERSION% == 3 (
        SET WINDOWS_SDK_VERSION="v7.1"
        IF %MINOR_PYTHON_VERSION% LEQ 4 (
            SET SET_SDK_64=Y
        ) ELSE (
            SET SET_SDK_64=N
            IF EXIST "%WIN_WDK%" (
                :: See: https://connect.microsoft.com/VisualStudio/feedback/details/1610302/
                REN "%WIN_WDK%" 0wdf
            )
        )
    ) ELSE (
        ECHO Unsupported Python version: "%MAJOR_PYTHON_VERSION%"
        EXIT 1
    )
)

IF %PYTHON_ARCH% == 64 (
    IF %SET_SDK_64% == Y (
        ECHO Configuring Windows SDK %WINDOWS_SDK_VERSION% for Python %MAJOR_PYTHON_VERSION% on a 64 bit architecture
        SET DISTUTILS_USE_SDK=1
        SET MSSdk=1
        "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Setup\WindowsSdkVer.exe" -q -version:%WINDOWS_SDK_VERSION%
        "%WIN_SDK_ROOT%\%WINDOWS_SDK_VERSION%\Bin\SetEnv.cmd" /x64 /release
        ECHO Executing: %COMMAND_TO_RUN%
        call %COMMAND_TO_RUN% || EXIT 1
    ) ELSE (
        ECHO Using default MSVC build environment for 64 bit architecture
        ECHO Executing: %COMMAND_TO_RUN%
        call %COMMAND_TO_RUN% || EXIT 1
    )
) ELSE (
    ECHO Using default MSVC build environment for 32 bit architecture
    ECHO Executing: %COMMAND_TO_RUN%
    call %COMMAND_TO_RUN% || EXIT 1
)
