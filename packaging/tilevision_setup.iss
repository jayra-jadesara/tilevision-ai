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
#define MyAppVersion "1.0.1"
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
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nTileVision AI is an offline visual tile search application for showrooms and distributors.

[Code]
function InitializeSetup(): Boolean;
begin
  if not DirExists(ExpandConstant('{#BuildSource}')) then
  begin
    MsgBox('PyInstaller build not found.' + #13#10 +
      'Run first: powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1' + #13#10 +
      'Expected folder: dist\TileVisionAI\',
      mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
