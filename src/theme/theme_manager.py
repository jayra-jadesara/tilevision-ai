"""
Theme management for TileVision AI.

Provides dark and light QSS variants applied at the QApplication level.

SCOPE NOTE: Several views (SearchView, DuplicatesView, CropDialog,
IndexingView) apply their own hardcoded dark-palette QSS directly via
setStyleSheet() on themselves — Qt gives explicit widget-level
stylesheets precedence over inherited/application-level ones for any
property they set. This means the app-level theme switch here reliably
re-skins the MainWindow chrome (sidebar, status bar, page background),
but a full re-skin of every already-styled child view for light mode is
a larger follow-up task (each view would need its own theme-aware
_apply_styles(theme) variant). This is a known, deliberate scope
boundary rather than an oversight — documented here so it's easy to
finish later without re-deriving why the chrome and content don't fully
match in light mode yet.
"""

DARK_APP_QSS = """
QMainWindow, QDialog { background-color: #1A1D26; }
QWidget { color: #E8EAF6; }
#Sidebar { background-color: #14161F; border-right: 1px solid #232634; }
#AppStatusBar { background-color: #14161F; color: #8A8FA3; border-top: 1px solid #232634; }
#ContentStack { background-color: #1A1D26; }
"""

LIGHT_APP_QSS = """
QMainWindow, QDialog { background-color: #F5F6FA; }
QWidget { color: #1E212C; }
#Sidebar { background-color: #FFFFFF; border-right: 1px solid #E0E2EC; }
#AppStatusBar { background-color: #FFFFFF; color: #5A5F73; border-top: 1px solid #E0E2EC; }
#ContentStack { background-color: #F5F6FA; }
"""


def get_app_stylesheet(theme: str) -> str:
    """
    Return the application-level QSS for the given theme.

    Args:
        theme: "dark" or "light". Any other value falls back to "dark".

    Returns:
        A QSS string, applied via QApplication.setStyleSheet().
    """
    return LIGHT_APP_QSS if theme == "light" else DARK_APP_QSS
