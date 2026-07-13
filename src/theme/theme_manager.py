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

SCOPE NOTE (historical — now fixed): earlier versions of this app had
every view hardcoding its own dark-only colors directly, so switching
themes only ever re-skinned the MainWindow chrome plus the shared
combo/menu/table/scrollbar fixes above, not each view's own panels/cards.
See get_palette() below — every view now builds its stylesheet from the
shared palette and exposes set_theme() so MainWindow can propagate a
theme change to every open view immediately.
"""

import re

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


# ─────────────────────────────────────────────────────────────────────────
# Shared color palette (fixes "theme not worked properly")
#
# Every view previously hardcoded its own dark-only hex colors directly in
# a local setStyleSheet() call. Switching themes only ever re-skinned the
# MainWindow chrome + combo/menu/table/scrollbar (see SCOPE NOTE above) —
# every view's own panels, cards, and text stayed dark regardless of the
# selected theme, which is what made the light theme look broken/half-
# applied rather than genuinely not working at all.
#
# Fix: every view now pulls its colors from get_palette(theme) instead of
# hardcoding hex values, and exposes a set_theme(theme) method that rebuilds
# its stylesheet from the new palette. MainWindow calls set_theme() on every
# currently-open view when the theme changes, and passes the current theme
# to any dialog constructed afterward (Duplicates, Crop, Help), so newly
# opened dialogs also match immediately.
# ─────────────────────────────────────────────────────────────────────────

_DARK_PALETTE = {
    "bg_app": "#1A1D26",
    "bg_panel": "#232634",
    "bg_panel_alt": "#1E212C",
    "bg_sidebar": "#14161F",
    "bg_input": "#232634",
    "text_primary": "#E8EAF6",
    "text_secondary": "#ACB0C4",
    "text_muted": "#8A8FA3",
    "text_faint": "#55596B",
    "border": "#2E3243",
    "border_strong": "#3A3F52",
    "accent": "#3949AB",
    "accent_hover": "#5C6BC0",
    "accent_text": "#7C83D3",
    "button_bg": "#2A2E3D",
    "button_hover": "#333852",
    "success_bg": "#1B4332",
    "success_text": "#6EE7B7",
    "warning_bg": "#453410",
    "warning_text": "#FBBF24",
    "danger_bg": "#4A1A1A",
    "danger_hover": "#7A2828",
    "row_alt": "#262B3D",
}

_LIGHT_PALETTE = {
    "bg_app": "#F5F6FA",
    "bg_panel": "#FFFFFF",
    "bg_panel_alt": "#F7F8FC",
    "bg_sidebar": "#FFFFFF",
    "bg_input": "#FFFFFF",
    "text_primary": "#1E212C",
    "text_secondary": "#3F4358",
    "text_muted": "#5A5F73",
    "text_faint": "#9297A8",
    "border": "#E0E2EC",
    "border_strong": "#C7CAD9",
    "accent": "#3949AB",
    "accent_hover": "#5C6BC0",
    "accent_text": "#3949AB",
    "button_bg": "#EEF0F7",
    "button_hover": "#E0E4F2",
    "success_bg": "#DCFCE7",
    "success_text": "#15803D",
    "warning_bg": "#FEF3C7",
    "warning_text": "#B45309",
    "danger_bg": "#FEE2E2",
    "danger_hover": "#FCA5A5",
    "row_alt": "#F5F6FB",
}


def get_palette(theme: str) -> dict:
    """
    Return the shared named-color palette for a theme, for views to build
    their own stylesheets from.

    Args:
        theme: "dark" or "light". Any other value falls back to "dark".

    Returns:
        A dict of semantic color name -> hex string. Keys are stable
        across both themes so view code never branches on theme itself,
        only on which palette dict it was given.
    """
    return dict(_LIGHT_PALETTE if theme == "light" else _DARK_PALETTE)


def adapt_legacy_stylesheet(stylesheet: str, theme: str) -> str:
    """Translate legacy dark-only QSS into the selected shared palette.

    MainWindow predates the palette-based views and still owns a local QSS
    stylesheet. Local widget QSS has higher precedence than QApplication QSS,
    so it must be translated too or it will keep the app chrome dark in light
    mode.
    """
    if theme != "light":
        return stylesheet

    palette = get_palette(theme)
    replacements = {
        "#1A1D26": palette["bg_app"],
        "#13151F": palette["bg_sidebar"],
        "#2D3250": palette["border"],
        "#5C6BC0": palette["accent_hover"],
        "#37474F": palette["text_faint"],
        "#546E7A": palette["text_muted"],
        "#1E2130": palette["bg_panel_alt"],
        "#1E2847": palette["button_hover"],
        "#7986CB": palette["accent_text"],
        "#69F0AE": palette["success_text"],
        "#E8EAF6": palette["text_primary"],
        "#252837": palette["bg_input"],
        "#3D4166": palette["border_strong"],
    }
    pattern = re.compile("|".join(re.escape(color) for color in replacements), re.IGNORECASE)
    return pattern.sub(lambda match: replacements[match.group(0).upper()], stylesheet)
