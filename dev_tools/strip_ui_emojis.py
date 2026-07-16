"""One-off helper to remove emoji decorations from presentation views."""

from __future__ import annotations

import re
from pathlib import Path

REPLACEMENTS = [
    ("📁  Folder Indexing", "Folder Indexing"),
    ("📂  Tile Image Folder", "Tile Image Folder"),
    ("⚙ Indexing Progress", "Indexing Progress"),
    ("▶  Start Indexing", "Start Indexing"),
    ("▶  Resume", "Resume"),
    ("▶  Scan for Duplicates", "Scan for Duplicates"),
    ("⏸  Pause", "Pause"),
    ("✕  Cancel", "Cancel"),
    ("✕  Clear", "Clear"),
    ("📊  Results Summary", "Results Summary"),
    ("📋  Activity Log", "Activity Log"),
    ("📋  Copy Path", "Copy Path"),
    ("📋 Copy", "Copy"),
    ("⬇  Export Catalogue", "Export Catalogue"),
    ("✂️  Crop & Search", "Crop and Search"),
    ("✂️  Search This Region", "Search This Region"),
    ("🕐  Recent Searches", "Recent Searches"),
    ("🕐 Trial —", "Trial —"),
    ("🔍  Duplicate & Near-Duplicate Tile Detection", "Duplicate and Near-Duplicate Tile Detection"),
    ("🔍  ", ""),
    ("🔓  Activate License", "Activate License"),
    ("🔓 Unlicensed", "Unlicensed"),
    ("➕  Add Folder", "Add Folder"),
    ("➖  Remove Selected", "Remove Selected"),
    ("💾  Backup Database", "Backup Database"),
    ("📤  Export Logs", "Export Logs"),
    ("🔄  Rebuild FAISS Index", "Rebuild FAISS Index"),
    ("🧹  Clear Cache", "Clear Cache"),
    ("🖼️  Open Image", "Open Image"),
    ("🖼️  Screenshot coming soon", "Screenshot coming soon"),
    ("👁 View", "View"),
    ("⚠️ ", "Warning: "),
    ("⚠  ", "Warning: "),
    ("✅ ", ""),
    ("⛔ ", ""),
    ("❌ ", ""),
    ("⏳ ", ""),
    ("🟢 Exact", "Exact"),
    ("🟡 Near", "Near"),
    ("🔐 ", ""),
    ("🔓 ", ""),
    ('_StatCard("🖼️",', "_StatCard("),
    ('_StatCard("📂",', "_StatCard("),
    ('_StatCard("🗄️",', "_StatCard("),
    ('_StatCard("🧮",', "_StatCard("),
    ('_StatCard("🔍",', "_StatCard("),
    ('_StatCard("⏳",', "_StatCard("),
    ('_StatCard("👁️",', "_StatCard("),
    ('_make_stat_block("🆕",', '_make_stat_block("",'),
    ('_make_stat_block("✏️",', '_make_stat_block("",'),
    ('_make_stat_block("🗑",', '_make_stat_block("",'),
    ('_make_stat_block("⏭",', '_make_stat_block("",'),
    ('("🔍", "Search"', '("Search", "Search"'),
    ('("📁", "Index Folder"', '("Index", "Index Folder"'),
    ('("🧬", "Duplicate Detection"', '("Duplicates", "Duplicate Detection"'),
    ('("⚙️", "Settings"', '("Settings", "Settings"'),
    ('return "Trial", "🕐"', 'return "Trial", ""'),
    ('return self._license_details.get("license_type", "Licensed"), "🔐"', 'return self._license_details.get("license_type", "Licensed"), ""'),
    ('return "Unlicensed", "🔓"', 'return "Unlicensed", ""'),
    ('QLabel("🖼️")', 'QLabel("Tile Image")'),
    ('QLabel("🔍")', 'QLabel("")'),
    ('QPushButton("🖼️")', 'QPushButton("Open")'),
    ('QPushButton("🗑")', 'QPushButton("Delete")'),
    ("⚠️ Indexed features are outdated ", "Indexed features are outdated — "),
    (', "🖱️",', ', "",'),
]

def main() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "presentation" / "views"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")
            print(path.relative_to(root.parent.parent))


if __name__ == "__main__":
    main()
