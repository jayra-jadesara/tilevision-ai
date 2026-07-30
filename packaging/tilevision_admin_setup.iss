; TileVision AI — Vendor Admin tool (Windows only)
; DO NOT ship this installer to customers — vendor use only.

#define MyAppName "TileVision AI Admin"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "JD Software"
#define MyAppExeName "TileVisionAI-Admin.exe"
#define BuildSource "..\dist\TileVisionAI-Admin"

[Setup]
AppId={{B8C4D5E6-F7A8-4901-BCDE-F12345678901}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=TileVisionAI-Admin-VENDOR-ONLY-{#MyAppVersion}
SetupIconFile=..\src\resources\app_icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#BuildSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "VENDOR_ADMIN_README.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Messages]
WelcomeLabel2=Vendor-only license manager for TileVision AI.%n%nDO NOT install this on customer PCs or send this file to clients.%n%nYour private signing key stays in %USERPROFILE%\.tilevision_ai_vendor\ — it is never bundled in this installer.
