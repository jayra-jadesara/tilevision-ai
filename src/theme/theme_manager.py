"""
Theme management for TileVision AI.

Provides dark and light QSS variants applied at the QApplication level.

Covers, app-wide (Tasks 5 & 10 — dark theme readability fixes):
  - QComboBox popup lists (QAbstractItemView) — previously unstyled
    anywhere in the app. Qt's default combo-box popup does NOT
    automatically inherit a widget's own stylesheet colors; it's a
    separate top-level view. Every view sets `color: #E8EAF6` (near-
    white) on QWidget app-wide, which the popup's item text DOES pick up,
    but the popup's BACKGROUND stays whatever the native/default style
    provides (frequently white or light gray) — producing light text on
    a light background, i.e. unreadable. This was the actual root cause
    of "dropdown text is unreadable".
  - QMenu (right-click context menus, e.g. Duplicates/Search actions).
  - QTableWidget selection/hover colors (Search results table).
  - QScrollBar (dark-styled scroll handles instead of the native default,
    which looks out of place against a dark background).

SCOPE NOTE: Several views (SearchView, DuplicatesView, CropDialog,
IndexingView) apply their own hardcoded dark-palette QSS directly via
setStyleSheet() on themselves — Qt gives explicit widget-level
stylesheets precedence over inherited/application-level ones for any
property they set. This means the app-level theme switch here reliably
re-skins the MainWindow chrome (sidebar, status bar, page background)
plus the combo/menu/table/scrollbar fixes above (those were never styled
locally anywhere, so there's no per-widget override to conflict with),
but a full re-skin of the *rest* of every already-styled child view for
light mode is a larger follow-up task (each view would need its own
theme-aware _apply_styles(theme) variant). This is a known, deliberate
scope boundary rather than an oversight — documented here so it's easy
to finish later without re-deriving why the chrome/dropdowns/menus match
the selected theme but some view backgrounds don't yet.
"""

_SHARED_COMPONENT_QSS_TEMPLATE = """
/* ── QComboBox popup list (the actual bug fix for Task 5/10) ── */
QComboBox QAbstractItemView {{
    background-color: {popup_bg};
    color: {popup_text};
    border: 1px solid {border};
    selection-background-color: {selection_bg};
    selection-color: {selection_text};
    outline: none;
    padding: 2px;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

/* ── Context menus ── */
QMenu {{
    background-color: {popup_bg};
    color: {popup_text};
    border: 1px solid {border};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 20px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {selection_bg};
    color: {selection_text};
}}
QMenu::separator {{
    height: 1px;
    background-color: {border};
    margin: 4px 8px;
}}

/* ── Tables (Search results, etc.) ── */
QTableWidget {{
    gridline-color: {border};
    selection-background-color: {selection_bg};
    selection-color: {selection_text};
}}
QTableWidget::item:hover {{
    background-color: {hover_bg};
}}
QHeaderView::section {{
    background-color: {header_bg};
    color: {popup_text};
    border: none;
    border-bottom: 1px solid {border};
    padding: 6px;
}}

/* ── Scrollbars ── */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {scrollbar_handle};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {scrollbar_handle_hover};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {scrollbar_handle};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {scrollbar_handle_hover};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
"""

DARK_APP_QSS = """
QMainWindow, QDialog { background-color: #1A1D26; }
QWidget { color: #E8EAF6; }
#Sidebar { background-color: #14161F; border-right: 1px solid #232634; }
#AppStatusBar { background-color: #14161F; color: #8A8FA3; border-top: 1px solid #232634; }
#ContentStack { background-color: #1A1D26; }
""" + _SHARED_COMPONENT_QSS_TEMPLATE.format(
    popup_bg="#232634",
    popup_text="#E8EAF6",
    border="#3A3F52",
    selection_bg="#3B4270",
    selection_text="#FFFFFF",
    hover_bg="#2A2E3D",
    header_bg="#262B3D",
    scrollbar_handle="#3A3F52",
    scrollbar_handle_hover="#5C6BC0",
)

LIGHT_APP_QSS = """
QMainWindow, QDialog { background-color: #F5F6FA; }
QWidget { color: #1E212C; }
#Sidebar { background-color: #FFFFFF; border-right: 1px solid #E0E2EC; }
#AppStatusBar { background-color: #FFFFFF; color: #5A5F73; border-top: 1px solid #E0E2EC; }
#ContentStack { background-color: #F5F6FA; }
""" + _SHARED_COMPONENT_QSS_TEMPLATE.format(
    popup_bg="#FFFFFF",
    popup_text="#1E212C",
    border="#D5D8E3",
    selection_bg="#5C6BC0",
    selection_text="#FFFFFF",
    hover_bg="#EEF0F7",
    header_bg="#F0F1F7",
    scrollbar_handle="#C7CAD9",
    scrollbar_handle_hover="#5C6BC0",
)


def get_app_stylesheet(theme: str) -> str:
    """
    Return the application-level QSS for the given theme.

    Args:
        theme: "dark" or "light". Any other value falls back to "dark".

    Returns:
        A QSS string, applied via QApplication.setStyleSheet().
    """
    return LIGHT_APP_QSS if theme == "light" else DARK_APP_QSS
