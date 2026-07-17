"""Theme styles for the vendor Admin License Manager."""

from __future__ import annotations

from src.theme.theme_manager import get_palette


def get_admin_qss(theme: str) -> str:
    """Return a complete stylesheet for the vendor admin tool."""
    p = get_palette(theme)
    return f"""
    QMainWindow, QWidget {{
        background-color: {p['bg_app']};
        color: {p['text_primary']};
        font-size: 12px;
    }}
    QLabel {{
        background: transparent;
        color: {p['text_secondary']};
    }}
    #Title {{
        font-size: 20px;
        font-weight: 700;
        color: {p['text_primary']};
        background: transparent;
    }}
    #Subtitle {{
        font-size: 12px;
        color: {p['text_muted']};
        background: transparent;
    }}
    #BrandLabel {{
        background: transparent;
    }}
    #StatValue {{
        font-size: 24px;
        font-weight: 700;
        color: {p['accent']};
        background: transparent;
    }}
    #Warning {{
        color: {p['warning_text']};
        background-color: {p['warning_bg']};
        border: 1px solid {p['warning_text']};
        border-radius: 8px;
        padding: 10px 12px;
    }}
    #Hint {{
        color: {p['text_muted']};
        font-size: 11px;
        background: transparent;
    }}
    #KeyStatus {{
        font-weight: 600;
        background: transparent;
    }}
    #KeyStatus[loaded="true"] {{
        color: {p['accent']};
    }}
    #KeyStatus[loaded="false"] {{
        color: {p['warning_text']};
    }}
    #ThemeLabel {{
        color: {p['text_muted']};
        font-size: 11px;
        font-weight: 600;
        background: transparent;
    }}
    QGroupBox {{
        background-color: {p['bg_panel']};
        border: 1px solid {p['border']};
        border-radius: 10px;
        margin-top: 14px;
        padding: 18px 14px 14px 14px;
        font-weight: 600;
        color: {p['text_secondary']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {p['accent_text']};
        background-color: {p['bg_panel']};
    }}
    QTabWidget::pane {{
        border: 1px solid {p['border']};
        border-radius: 8px;
        background-color: {p['bg_panel']};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {p['bg_panel_alt']};
        color: {p['text_muted']};
        border: 1px solid {p['border']};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 8px 16px;
        margin-right: 4px;
    }}
    QTabBar::tab:selected {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
        font-weight: 600;
    }}
    QTabBar::tab:hover {{
        color: {p['text_primary']};
        background-color: {p['button_hover']};
    }}
    QLineEdit, QPlainTextEdit, QTextEdit {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        border-radius: 8px;
        padding: 4px 10px;
        min-height: 24px;
        selection-background-color: {p['highlight_bg']};
    }}
    #WebDatePicker {{
        background-color: {p['bg_input']};
        border: 1px solid {p['border_strong']};
        border-radius: 8px;
        min-height: 36px;
        max-height: 36px;
        min-width: 210px;
        max-width: 210px;
        padding: 0;
        margin: 0;
    }}
    #ExpiryDateRow {{
        background: transparent;
        min-height: 40px;
        max-height: 40px;
    }}
    #WebDatePicker:disabled {{
        background-color: {p['bg_panel_alt']};
        color: {p['text_muted']};
    }}
    #WebDateEdit {{
        background: transparent;
        border: none;
        color: {p['text_primary']};
        padding: 0 10px;
        margin: 0;
        min-height: 0px;
    }}
    #WebDateEdit:disabled {{
        color: {p['text_muted']};
    }}
    #WebDateButton {{
        background-color: {p['button_bg']};
        color: {p['text_secondary']};
        border: none;
        border-left: 1px solid {p['border_strong']};
        border-top-right-radius: 7px;
        border-bottom-right-radius: 7px;
        min-width: 40px;
        max-width: 40px;
        min-height: 0px;
        padding: 0;
        margin: 0;
        font-size: 15px;
    }}
    #WebDateButton:hover {{
        background-color: {p['button_hover']};
        color: {p['text_primary']};
    }}
    #WebDateButton:disabled {{
        background-color: {p['bg_panel_alt']};
        color: {p['text_muted']};
    }}
    #WebDatePopup {{
        background-color: {p['bg_panel']};
        border: 1px solid {p['border_strong']};
        border-radius: 10px;
    }}
    #WebDateNav {{
        background: transparent;
    }}
    #WebDateNavButton {{
        background-color: {p['button_bg']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        border-radius: 6px;
        padding: 0;
        min-width: 28px;
        max-width: 28px;
        min-height: 28px;
        max-height: 28px;
        font-size: 16px;
        font-weight: 600;
    }}
    #WebDateNavButton:hover {{
        background-color: {p['button_hover']};
    }}
    #WebDateMonth {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        border-radius: 6px;
        padding: 2px 8px;
        min-height: 28px;
        max-height: 28px;
        min-width: 96px;
    }}
    #WebDateMonth::drop-down {{
        border: none;
        width: 18px;
    }}
    #WebDateMonth QAbstractItemView {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        selection-background-color: {p['highlight_bg']};
        selection-color: {p['text_primary']};
        outline: none;
        padding: 2px;
    }}
    #WebDateYear {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        border-radius: 6px;
        padding: 2px 4px;
        min-height: 28px;
        max-height: 28px;
        min-width: 72px;
        max-width: 76px;
    }}
    #WebDateYear::up-button, #WebDateYear::down-button {{
        width: 16px;
        border: none;
        background-color: {p['button_bg']};
    }}
    #WebDateYear::up-button:hover, #WebDateYear::down-button:hover {{
        background-color: {p['button_hover']};
    }}
    #WebDateYear::up-arrow, #WebDateYear::down-arrow {{
        width: 8px;
        height: 8px;
    }}
    #WebDateCalendar {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
    }}
    #WebDateCalendar QWidget {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
    }}
    #WebDateCalendar QAbstractItemView:enabled {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        selection-background-color: {p['accent']};
        selection-color: {p['button_text']};
        font-size: 12px;
        outline: none;
    }}
    QCalendarWidget {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
    }}
    QCalendarWidget QWidget {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
    }}
    QCalendarWidget QAbstractItemView:enabled {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        selection-background-color: {p['highlight_bg']};
        selection-color: {p['text_primary']};
    }}
    QCalendarWidget QToolButton {{
        background-color: {p['button_bg']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QCalendarWidget QSpinBox {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
    }}
    QComboBox {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        border-radius: 8px;
        padding: 4px 10px;
        min-height: 24px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p['bg_panel']};
        color: {p['text_primary']};
        border: 1px solid {p['border_strong']};
        selection-background-color: {p['highlight_bg']};
        selection-color: {p['text_primary']};
        outline: none;
    }}
    QTableWidget {{
        background-color: {p['bg_input']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        gridline-color: {p['border']};
        selection-background-color: {p['highlight_bg']};
        selection-color: {p['text_primary']};
    }}
    QHeaderView::section {{
        background-color: {p['bg_panel_alt']};
        color: {p['text_secondary']};
        border: none;
        border-bottom: 1px solid {p['border']};
        padding: 8px;
        font-weight: 600;
    }}
    QPushButton {{
        background-color: {p['button_bg']};
        color: {p['button_secondary_text']};
        border: 1px solid {p['border_strong']};
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {p['button_hover']};
        color: {p['text_primary']};
    }}
    #PrimaryButton {{
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 {p['accent_deep']},
            stop: 1 {p['accent']}
        );
        color: {p['button_text']};
        border: none;
        font-weight: 600;
    }}
    #PrimaryButton:hover {{
        background: {p['accent_hover']};
    }}
    #DangerButton {{
        background-color: {p['danger_bg']};
        color: {p['danger_text']};
        border: 1px solid {p['danger_text']};
    }}
    #DangerButton:hover {{
        background-color: {p['danger_hover']};
        color: {p['button_text']};
    }}
    QCheckBox {{
        color: {p['text_secondary']};
        background: transparent;
        spacing: 8px;
    }}
    QScrollBar:vertical {{
        background: {p['bg_panel_alt']};
        width: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {p['border_strong']};
        border-radius: 5px;
        min-height: 24px;
    }}
    """
