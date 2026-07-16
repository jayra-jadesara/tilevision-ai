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
QMainWindow, QDialog { background-color: #0F172A; }
QWidget { color: #F1F5F9; }
#Sidebar { background-color: #0B1120; border-right: 1px solid #1E293B; }
#AppStatusBar { background-color: #0B1120; color: #94A3B8; border-top: 1px solid #1E293B; }
#ContentStack { background-color: #0F172A; }
""" + _SHARED_COMPONENT_QSS_TEMPLATE.format(
    popup_bg="#1E293B",
    popup_text="#F1F5F9",
    border="#334155",
    selection_bg="#0369A1",
    selection_text="#FFFFFF",
    hover_bg="#273449",
    header_bg="#172033",
    scrollbar_handle="#475569",
    scrollbar_handle_hover="#0EA5E9",
)

LIGHT_APP_QSS = """
QMainWindow, QDialog { background-color: #F1F5F9; }
QWidget { color: #0F172A; }
#Sidebar { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
#AppStatusBar { background-color: #FFFFFF; color: #64748B; border-top: 1px solid #E2E8F0; }
#ContentStack { background-color: #F1F5F9; }
""" + _SHARED_COMPONENT_QSS_TEMPLATE.format(
    popup_bg="#FFFFFF",
    popup_text="#0F172A",
    border="#CBD5E1",
    selection_bg="#0284C7",
    selection_text="#FFFFFF",
    hover_bg="#E0F2FE",
    header_bg="#F8FAFC",
    scrollbar_handle="#CBD5E1",
    scrollbar_handle_hover="#0EA5E9",
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
    "bg_app": "#0F172A",
    "bg_panel": "#1E293B",
    "bg_panel_alt": "#172033",
    "bg_sidebar": "#0B1120",
    "bg_input": "#1E293B",
    "text_primary": "#F1F5F9",
    "text_secondary": "#CBD5E1",
    "text_muted": "#94A3B8",
    "text_faint": "#64748B",
    "border": "#334155",
    "border_strong": "#475569",
    "accent": "#0EA5E9",
    "accent_hover": "#38BDF8",
    "accent_text": "#7DD3FC",
    "accent_deep": "#0284C7",
    "accent_soft": "#22D3EE",
    "highlight_bg": "#164E63",
    "highlight_border": "#0EA5E9",
    "button_bg": "#1E293B",
    "button_hover": "#334155",
    "success_bg": "#064E3B",
    "success_text": "#6EE7B7",
    "warning_bg": "#451A03",
    "warning_text": "#FBBF24",
    "danger_bg": "#450A0A",
    "danger_hover": "#991B1B",
    "danger_text": "#FCA5A5",
    "row_alt": "#172033",
    "button_text": "#FFFFFF",
    "button_secondary_text": "#E2E8F0",
}

_LIGHT_PALETTE = {
    "bg_app": "#F1F5F9",
    "bg_panel": "#FFFFFF",
    "bg_panel_alt": "#F8FAFC",
    "bg_sidebar": "#FFFFFF",
    "bg_input": "#FFFFFF",
    "text_primary": "#0F172A",
    "text_secondary": "#334155",
    "text_muted": "#64748B",
    "text_faint": "#94A3B8",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "accent": "#0284C7",
    "accent_hover": "#0EA5E9",
    "accent_text": "#0369A1",
    "accent_deep": "#0369A1",
    "accent_soft": "#38BDF8",
    "highlight_bg": "#E0F2FE",
    "highlight_border": "#7DD3FC",
    "button_bg": "#F8FAFC",
    "button_hover": "#E2E8F0",
    "success_bg": "#DCFCE7",
    "success_text": "#15803D",
    "warning_bg": "#FEF3C7",
    "warning_text": "#B45309",
    "danger_bg": "#FEE2E2",
    "danger_hover": "#FCA5A5",
    "danger_text": "#DC2626",
    "row_alt": "#F8FAFC",
    "button_text": "#FFFFFF",
    "button_secondary_text": "#334155",
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


def get_shared_view_qss(theme: str) -> str:
    """Shared typography and button styles used on every main screen."""
    p = get_palette(theme)
    return f"""
    #PageTitle, #Title, #DialogTitle {{
        font-size: 20px;
        font-weight: 700;
        color: {p['text_primary']};
    }}
    #PageSubtitle, #Subtitle, #DialogSubtitle {{
        font-size: 12px;
        color: {p['text_muted']};
    }}
    #SectionLabel {{
        font-size: 13px;
        font-weight: 600;
        color: {p['text_secondary']};
    }}
    #SectionNote {{
        font-size: 11px;
        color: {p['text_muted']};
    }}
    #PrimaryButton, #StartButton, #ScanButton, #ActionButton, #CloseButton, #ActivateButton {{
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {p['accent_deep']},
            stop: 1 {p['accent']}
        );
        color: {p['button_text']};
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
    }}
    #PrimaryButton:hover:enabled, #StartButton:hover:enabled, #ScanButton:hover:enabled,
    #ActionButton:hover:enabled, #CloseButton:hover:enabled, #ActivateButton:hover:enabled {{
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {p['accent']},
            stop: 1 {p['accent_soft']}
        );
    }}
    #PrimaryButton:pressed, #StartButton:pressed, #ScanButton:pressed,
    #ActionButton:pressed, #CloseButton:pressed, #ActivateButton:pressed {{
        background-color: {p['accent_deep']};
    }}
    #PrimaryButton:disabled, #StartButton:disabled, #ScanButton:disabled,
    #ActionButton:disabled, #CloseButton:disabled, #ActivateButton:disabled {{
        background-color: {p['button_bg']};
        color: {p['text_faint']};
    }}
    #SecondaryButton, #BrowseButton, #ClearLogButton {{
        background-color: {p['button_bg']};
        color: {p['text_secondary']};
        border: 1px solid {p['border_strong']};
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 600;
    }}
    #SecondaryButton:hover:enabled, #BrowseButton:hover:enabled, #ClearLogButton:hover:enabled {{
        background-color: {p['button_hover']};
        color: {p['text_primary']};
        border-color: {p['accent_soft']};
    }}
    #SecondaryButton:disabled, #BrowseButton:disabled, #ClearLogButton:disabled {{
        color: {p['text_faint']};
        border-color: {p['border']};
    }}
    """


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
