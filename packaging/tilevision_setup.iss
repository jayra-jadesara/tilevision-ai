; TileVision AI — Windows installer (Inno Setup 6)
;
; Prerequisites:
;   1. PyInstaller one-folder build at dist\TileVisionAI\
;   2. Inno Setup 6 — https://jrsoftware.org/isinfo.php
;
; Build from project root:
;   powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
;
; Or manually:
;   iscc packaging\tilevision_setup.iss

#define MyAppName "TileVision AI"
#define MyAppVersion "1.2.26"
#define MyAppPublisher "JD Software"
#define MyAppExeName "TileVisionAI.exe"
#define BuildSource "..\dist\TileVisionAI"

[Setup]
AppId={{A7B3C4D5-E6F7-4890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=
OutputDir=..\dist\installer
OutputBaseFilename=TileVisionAI-Setup-{#MyAppVersion}
SetupIconFile=..\src\resources\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; Allow in-app silent upgrades to close the running app and replace files.
CloseApplications=yes
CloseApplicationsFilter=TileVisionAI.exe
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#BuildSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Dirs]
; Encrypted license/trial store (see src/licensing/crypto_store.py)
Name: "{commonappdata}\TileVisionAI"; Permissions: users-modify
Name: "{commonappdata}\TileVisionAI\.lic"; Permissions: users-modify

[Run]
; Interactive installs only. Silent CI verify and in-app upgrades must NOT
; launch the GUI here — without skipifsilent, Windows Build hung ~2h on
; Start-Process -Wait (v1.2.23–1.2.26). In-app updates relaunch via
; src/utils/update_installer.py after the elevated setup exits.
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nTileVision AI is an offline visual tile search application for showrooms and distributors.
