# /// script
# requires-python = ">=3.10"
# dependencies = ["PySide6-Essentials", "opencv-python-headless>=4.8,<5", "numpy", "requests"]
# ///
"""Sticker Forge - point it at a folder of videos, get a Telegram video sticker pack.

Run:  uv run stickermaker.py [folder]
"""
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

try:
    import cv2, numpy, requests, PySide6  # noqa: E401,F401
except ImportError:  # plain `python stickermaker.py`: re-launch through uv, which installs the deps in the header
    import shutil
    if os.environ.get("STICKERFORGE_UV") or not shutil.which("uv"):
        sys.exit("Missing dependencies. Run:  uv run stickermaker.py\n"
                 "or:  pip install PySide6 'opencv-python-headless<5' numpy requests")
    os.environ["STICKERFORGE_UV"] = "1"
    os.execvp("uv", ["uv", "run", __file__, *sys.argv[1:]])

from PySide6.QtCore import QByteArray, QObject, QSettings, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QVBoxLayout, QWidget)

import sticker_engine as eng

ICON = 150
MODES = ["auto", "face", "body", "center"]
SHAPES = ["square", "rounded", "circle"]

QSS = """
* { font-size: 13px; color: #e9eef3; }
QMainWindow, QWidget#root { background: #0e1621; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QGroupBox { background: #17212b; border: 1px solid #22303e; border-radius: 12px; margin-top: 18px; padding: 12px 10px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #6ab2f2; font-weight: 600; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #242f3d; border: 1px solid #2b3a4a; border-radius: 8px; padding: 5px 8px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #2aabee; }
QComboBox QAbstractItemView { background: #242f3d; selection-background-color: #2b5278; }
QPushButton { background: #242f3d; border: 1px solid #2b3a4a; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #2b3a4a; }
QPushButton:disabled { color: #5c6b7a; }
QPushButton#primary { background: #2aabee; border: none; color: white; }
QPushButton#primary:hover { background: #3cb6f3; }
QPushButton#accent { background: #4fae4e; border: none; color: white; }
QPushButton#accent:hover { background: #5cc05b; }
QListWidget { background: #0e1621; border: none; outline: none; }
QListWidget::item { background: #17212b; border-radius: 14px; margin: 6px; padding: 8px; color: #c7d2dc; }
QListWidget::item:hover { background: #1f2c3a; }
QListWidget::item:selected { background: #2b5278; color: white; }
QProgressBar { background: #242f3d; border: none; border-radius: 6px; height: 12px; text-align: center; font-size: 11px; }
QProgressBar::chunk { background: #2aabee; border-radius: 6px; }
QPlainTextEdit { background: #17212b; border: 1px solid #22303e; border-radius: 10px; font-family: monospace; font-size: 12px; color: #9fb0c0; }
QLabel#muted { color: #7f91a4; }
QLabel#title { font-size: 20px; font-weight: 700; }
QLabel#preview { background: #17212b; border-radius: 16px; }
QSplitter::handle { background: #0e1621; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { width: 0; border: none; }  /* wheel/arrows still work */
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar:horizontal { background: transparent; height: 8px; }
QScrollBar::handle { background: #2b3a4a; border-radius: 4px; min-height: 30px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { width: 0; height: 0; background: none; }
QCheckBox::indicator { width: 16px; height: 16px; }
"""


@dataclass
class Clip:
    path: Path
    segment: int = 0
    over: dict = field(default_factory=dict)  # per-clip Opts overrides
    emoji: str = ""
    status: str = "pending"                   # pending | queued | working | done | error
    result: dict | None = None
    key: str = ""                             # opts json used for the last good render
    error: str = ""
    uploaded: str = ""                        # "pack|key" once this exact render is in that Telegram pack
    label: str = ""                           # path relative to the folder, without extension
    ticket: object = None                     # identifies the live render job; None = cancelled

    @property
    def name(self):
        return (self.label or self.path.stem) + (f" #{self.segment + 1}" if self.segment else "")


class Bus(QObject):
    """Worker threads -> GUI thread (signals are queued across threads)."""
    scanned = Signal(int, object)
    thumb = Signal(int, int, object)
    rendered = Signal(int, int, str, object)
    failed = Signal(int, int, str)
    preview = Signal(int, int, object)
    log = Signal(str)
    published = Signal(object)
    uploaded = Signal(str, str)
    publish_failed = Signal(str)


def to_qimage(arr):
    """numpy BGR/BGRA -> owning QImage."""
    h, w, c = arr.shape
    if c == 3:
        return QImage(arr.tobytes(), w, h, 3 * w, QImage.Format_BGR888).copy()
    return QImage(arr.tobytes(), w, h, 4 * w, QImage.Format_ARGB32).copy()


def card_icon(img):
    """Fit an image into a rounded square tile for the grid."""
    pm = QPixmap(ICON, ICON)
    pm.fill(Qt.transparent)
    src = QPixmap.fromImage(img).scaled(ICON, ICON, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#0e1621"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(pm.rect(), 12, 12)
    p.drawPixmap((ICON - src.width()) // 2, (ICON - src.height()) // 2, src)
    p.end()
    return QIcon(pm)


def spin(lo, hi, val, step=1.0, suffix="", special=None, decimals=2):
    w = QDoubleSpinBox() if isinstance(step, float) else QSpinBox()
    if isinstance(w, QDoubleSpinBox):
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(val)
    w.setSuffix(suffix)
    if special:
        w.setSpecialValueText(special)
    return w


def combo(items, current):
    w = QComboBox()
    w.addItems(items)
    w.setCurrentText(current)
    return w


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sticker Forge")
        self.resize(1440, 900)
        self.setAcceptDrops(True)
        self.settings = QSettings("stickerforge", "stickerforge")
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max(1, (os.cpu_count() or 4) // 4))
        self.bus = Bus()
        self.clips: list[Clip] = []
        self.folder: Path | None = None
        self.gen = 0               # bumps when a new folder opens; stale worker results are ignored
        self.pending = 0
        self.total = 0
        self.publish_after = False
        self.up_total = self.up_done = 0
        self.stop_upload = False
        self.t0 = 0.0
        self.anim, self.anim_i = [], 0
        self.timer = QTimer(self, timeout=self.tick)

        self.bus.scanned.connect(self.on_scanned)
        self.bus.thumb.connect(self.on_thumb)
        self.bus.rendered.connect(self.on_rendered)
        self.bus.failed.connect(self.on_failed)
        self.bus.preview.connect(self.on_preview)
        self.bus.log.connect(self.log)
        self.bus.published.connect(self.on_published)
        self.bus.uploaded.connect(self.on_uploaded)
        self.bus.publish_failed.connect(self.on_publish_failed)

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.addLayout(self.build_topbar())
        split = self.split = QSplitter()
        split.addWidget(self.build_settings())
        split.addWidget(self.build_grid())
        split.addWidget(self.build_detail())
        split.setSizes([360, 730, 350])
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)
        self.restoreGeometry(QByteArray(self.settings.value("geometry", QByteArray())))
        self.split.restoreState(QByteArray(self.settings.value("split", QByteArray())))
        self.logbox = QPlainTextEdit(readOnly=True, maximumHeight=110)
        outer.addWidget(self.logbox)
        self.load_settings()
        for w in self.global_widgets.values():
            sig = getattr(w, "valueChanged", None) or getattr(w, "currentTextChanged", None) \
                or getattr(w, "toggled", None) or w.textChanged
            sig.connect(self.refresh_items)
        self.show_clip(None)
        self.log("Open or drop a folder of videos to start. Rendered stickers go to <folder>/stickers/")

    # ------------------------------------------------------------ layout
    def build_topbar(self):
        bar = QHBoxLayout()
        title = QLabel("Sticker Forge", objectName="title")
        self.folder_lbl = QLabel("No folder", objectName="muted")
        open_btn = QPushButton("📂  Open Folder…", clicked=self.pick_folder)
        self.process_btn = QPushButton("⚙  Process All", objectName="primary", clicked=self.process_all)
        self.pack_btn = QPushButton("🚀  Make Pack", objectName="accent", clicked=self.make_pack)
        out_btn = QPushButton("Show Output", clicked=lambda: self.folder and self.open_path(self.out_dir()))
        self.progress = QProgressBar(minimumWidth=260, visible=False)
        self.stop_btn = QPushButton("■  Stop", visible=False, clicked=self.stop)
        for w in (title, self.folder_lbl):
            bar.addWidget(w)
        bar.addStretch()
        for w in (self.progress, self.stop_btn, open_btn, out_btn, self.process_btn, self.pack_btn):
            bar.addWidget(w)
        return bar

    def build_settings(self):
        g = self.global_widgets = {
            "mode": combo(MODES, "auto"),
            "speed": spin(0, 4, 0, 0.25, "×", "Auto"),
            "duration": spin(0.5, 3.0, 3.0, 0.25, " s"),
            "fps": spin(10, 30, 30, 1, " fps"),
            "zoom": spin(0.5, 3.0, 1.0, 0.1, "×"),
            "shape": combo(SHAPES, "square"),
            "boomerang": QCheckBox("Ping-pong loop"),
            "max_kb": spin(32, 15360, eng.TG_MAX_KB, 32, " KB"),
            "per_video": spin(1, 20, 1, 1),
            "emoji": QLineEdit("🙂"),
            "auto_process": QCheckBox("Auto-process when a folder opens"),
            "recursive": QCheckBox("Include subfolders"),
            "token": QLineEdit(echoMode=QLineEdit.Password, placeholderText="123456:ABC… from @BotFather"),
            "user_id": QLineEdit(placeholderText="your numeric id (@userinfobot)"),
            "pack": QLineEdit(placeholderText="my_stickers"),
            "title": QLineEdit(placeholderText="My Stickers"),
            "replace": QCheckBox("Replace pack if it already exists"),
        }
        g["auto_process"].setChecked(True)
        g["zoom"].setToolTip("1.0 = automatic framing; higher = tighter on the face/body")
        g["max_kb"].setToolTip("Telegram rejects video stickers over 256 KB. Raise it only if you hand the files "
                               "to a bot like @fStikBot that re-compresses (accepts up to 15 MB).")
        g["per_video"].setToolTip("Make several stickers from the best distinct moments of long videos")
        g["emoji"].setToolTip("Space-separated emoji, e.g. '😂 🔥'")

        look = QGroupBox("Sticker")
        f = QFormLayout(look)
        for label, key in [("Focus", "mode"), ("Speed", "speed"), ("Length", "duration"), ("Frame rate", "fps"),
                           ("Zoom", "zoom"), ("Shape", "shape"), ("", "boomerang"), ("Max size", "max_kb"),
                           ("Moments / video", "per_video"), ("Default emoji", "emoji"), ("", "auto_process"),
                           ("", "recursive")]:
            f.addRow(label, g[key])

        tg = QGroupBox("Telegram")
        f = QFormLayout(tg)
        for label, key in [("Bot token", "token"), ("User ID", "user_id"), ("Pack name", "pack"),
                           ("Pack title", "title"), ("", "replace")]:
            f.addRow(label, g[key])
        help_lbl = QLabel(
            "1. Create a bot with <a style='color:#6ab2f2' href='https://t.me/BotFather'>@BotFather</a>, paste its token.<br>"
            "2. Get your ID from <a style='color:#6ab2f2' href='https://t.me/userinfobot'>@userinfobot</a>.<br>"
            "3. Send <b>/start</b> to your bot once.<br>"
            "The pack is owned by you; its link ends in <i>_by_yourbot</i>.",
            objectName="muted", wordWrap=True, openExternalLinks=True)
        f.addRow(help_lbl)

        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 14, 0)
        v.addWidget(look)
        v.addWidget(tg)
        v.addStretch()
        scroll = QScrollArea(widgetResizable=True, minimumWidth=320,
                             horizontalScrollBarPolicy=Qt.ScrollBarAlwaysOff)
        scroll.setWidget(box)
        return scroll

    def build_grid(self):
        self.grid = QListWidget(viewMode=QListView.IconMode, movement=QListView.Static,
                                resizeMode=QListView.Adjust, wordWrap=False, uniformItemSizes=True,
                                spacing=4, selectionMode=QListWidget.SingleSelection)
        self.grid.setIconSize(QSize(ICON, ICON))
        self.grid.setGridSize(QSize(ICON + 40, ICON + 78))
        self.grid.currentRowChanged.connect(lambda r: self.show_clip(r if r >= 0 else None))
        self.grid.itemChanged.connect(lambda _: (self.update_counts(), self.save_meta()))
        self.grid.itemDoubleClicked.connect(lambda it: self.open_clip(self.clips[self.grid.row(it)]))
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self.card_menu)
        self.empty = QLabel("Drop a folder of videos here\nor click “Open Folder…”", alignment=Qt.AlignCenter,
                            objectName="muted")
        self.empty.setStyleSheet("font-size: 17px;")
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.empty, 1)
        v.addWidget(self.grid, 1)
        self.count_lbl = QLabel("", objectName="muted")
        row = QHBoxLayout()
        row.addWidget(self.count_lbl, 1)
        for text, state in (("Select all", Qt.Checked), ("Select none", Qt.Unchecked)):
            row.addWidget(QPushButton(text, clicked=lambda _=False, s=state: self.check_all(s)))
        v.addLayout(row)
        return box

    def check_all(self, state):
        self.grid.blockSignals(True)
        for i in range(self.grid.count()):
            self.grid.item(i).setCheckState(state)
        self.grid.blockSignals(False)
        self.update_counts()
        self.save_meta()

    def card_menu(self, pos):
        it = self.grid.itemAt(pos)
        if not it:
            return
        i = self.grid.row(it)
        c = self.clips[i]
        m = QMenu(self)
        m.addAction("Re-render", lambda: c.status not in ("queued", "working") and self.queue(i))
        m.addAction("Open sticker", lambda: self.open_path(self.out_path(c))).setEnabled(self.out_path(c).exists())
        m.addAction("Open source video", lambda: self.open_path(c.path))
        m.addAction("Show output folder", lambda: self.open_path(self.out_dir()))
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def open_clip(self, c):
        self.open_path(self.out_path(c) if self.out_path(c).exists() else c.path)

    def build_detail(self):
        self.preview = QLabel(alignment=Qt.AlignCenter, objectName="preview", minimumSize=QSize(300, 300))
        self.clip_name = QLabel(objectName="title", wordWrap=True)
        self.clip_name.setStyleSheet("font-size: 15px;")
        self.clip_info = QLabel(objectName="muted", wordWrap=True)
        self.c_start = spin(-1, 36000, -1, 0.25, " s", "Auto")
        self.c_speed = spin(0, 4, 0, 0.25, "×", "Global")
        self.c_mode = combo(["global"] + MODES, "global")
        self.c_zoom = spin(0, 3, 0, 0.1, "×", "Global")
        self.c_shape = combo(["global"] + SHAPES, "global")
        self.c_emoji = QLineEdit(placeholderText="default")
        self.c_box = QGroupBox("This sticker")
        f = QFormLayout(self.c_box)
        for label, w in [("Start", self.c_start), ("Speed", self.c_speed), ("Focus", self.c_mode),
                         ("Zoom", self.c_zoom), ("Shape", self.c_shape), ("Emoji", self.c_emoji)]:
            f.addRow(label, w)
        row = QHBoxLayout()
        row.addWidget(QPushButton("Re-render", objectName="primary", clicked=self.apply_clip))
        row.addWidget(QPushButton("Reset", clicked=self.reset_clip))
        row.addWidget(QPushButton("Open", clicked=lambda: self.cur() and self.open_path(self.out_path(self.cur()))))
        f.addRow(row)
        self.c_emoji.editingFinished.connect(self.save_emoji)

        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 0, 0, 0)
        v.addWidget(self.preview)
        v.addWidget(self.clip_name)
        v.addWidget(self.clip_info)
        v.addWidget(self.c_box)
        v.addStretch()
        return box

    # ------------------------------------------------------------ settings
    def load_settings(self):
        for k, w in self.global_widgets.items():
            v = self.settings.value(k)
            if v is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(v in (True, "true"))
            elif isinstance(w, QComboBox):
                w.setCurrentText(v)
            elif isinstance(w, QLineEdit):
                w.setText(v)
            else:
                w.setValue(type(w.value())(v))

    def closeEvent(self, e):
        for k, w in self.global_widgets.items():
            self.settings.setValue(k, self.value(k))
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("split", self.split.saveState())
        self.stop_upload = True
        self.pool.clear()
        super().closeEvent(e)

    def value(self, k):
        w = self.global_widgets[k]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return w.value()

    def global_opts(self):
        return eng.Opts(**{k: self.value(k) for k in asdict(eng.Opts()) if k in self.global_widgets})

    def opts_for(self, c):
        return replace(self.global_opts(), segment=c.segment, **c.over)

    def key_for(self, c):
        return json.dumps(asdict(self.opts_for(c)), sort_keys=True)

    # ------------------------------------------------------------ folder / scan
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        p = Path(e.mimeData().urls()[0].toLocalFile())
        self.open_folder(p if p.is_dir() else p.parent)

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Video folder", str(self.folder or Path.home()))
        if d:
            self.open_folder(Path(d))

    def open_folder(self, folder):
        self.pool.clear()
        self.gen += 1
        self.pending = self.total = 0
        self.publish_after = False
        self.folder = folder
        self.settings.setValue("last_folder", str(folder))
        self.folder_lbl.setText(str(folder))
        self.clips = []
        self.grid.clear()
        self.update_counts()
        self.log(f"Scanning {folder} …")
        gen, per_video, opts, rec = self.gen, self.value("per_video"), self.global_opts(), self.value("recursive")

        def scan():
            found = []
            for v in eng.videos_in(folder, rec):
                try:
                    found.append((v, min(per_video, eng.segment_count(v, opts))))
                except Exception as ex:
                    self.bus.log.emit(f"skip {v.name}: {ex}")
            self.bus.scanned.emit(gen, found)
        self.pool.start(scan)

    def on_scanned(self, gen, found):
        if gen != self.gen:
            return
        self.out_dir().mkdir(exist_ok=True)
        meta = self.load_meta()
        for v, n in found:
            for s in range(n):
                c = Clip(v, s, label=str(v.relative_to(self.folder).with_suffix("")))
                m = meta.get(c.name)
                if m:  # restore earlier session's work for this folder
                    c.over, c.emoji, c.uploaded = m.get("over", {}), m.get("emoji", ""), m.get("uploaded", "")
                    if m.get("key") and m.get("result") and self.out_path(c).exists():
                        c.key, c.result, c.status = m["key"], m.get("result"), "done"
                self.clips.append(c)
                it = QListWidgetItem(c.name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked if m and m.get("skip") else Qt.Checked)
                it.setSizeHint(QSize(ICON + 40, ICON + 78))
                self.grid.addItem(it)
        self.log(f"Found {len(found)} videos → {len(self.clips)} stickers")
        for i, c in enumerate(self.clips):
            self.pool.start(lambda i=i, c=c, g=gen: self.load_thumb(g, i, c))
        self.refresh_items()
        if self.clips:
            self.grid.setCurrentRow(0)
        if self.value("auto_process"):
            self.process_all()

    def load_thumb(self, gen, i, c):
        try:
            if c.status == "done":
                img = eng.decode_preview(self.out_path(c), ICON)[0]
            else:
                img = eng.thumbnail(c.path, ICON * 2)
            if img is not None:
                self.bus.thumb.emit(gen, i, to_qimage(img))
        except Exception:
            pass

    def on_thumb(self, gen, i, img):
        if gen == self.gen:
            self.grid.item(i).setIcon(card_icon(img))

    # ------------------------------------------------------------ render
    def out_dir(self):
        return self.folder / "stickers"

    def out_path(self, c):
        stem = re.sub(r"[^\w.-]+", "_", c.label or c.path.stem)
        return self.out_dir() / f"{stem}_{c.segment}.webm"

    def stale(self, c):
        return c.status != "done" or c.key != self.key_for(c)

    def checked(self):
        return [i for i in range(len(self.clips)) if self.grid.item(i).checkState() == Qt.Checked]

    def process_all(self):
        todo = [i for i in self.checked() if self.stale(self.clips[i]) and self.clips[i].status not in ("queued", "working")]
        for i in todo:
            self.queue(i)
        if not todo and not self.pending and not self.publish_after:
            self.log("Everything is up to date.")

    def queue(self, i):
        c = self.clips[i]
        c.status = "queued"
        c.ticket = ticket = object()
        opts, key, out, gen = self.opts_for(c), self.key_for(c), self.out_path(c), self.gen
        if not self.total:
            self.t0 = time.monotonic()
        self.pending += 1
        self.total += 1
        self.update_progress()
        self.refresh_item(i)

        def work():
            if gen != self.gen or c.ticket is not ticket:  # folder changed or job stopped
                return
            self.bus.log.emit(f"Rendering {c.name} …")
            c.status = "working"
            try:
                res = eng.render(c.path, out, opts)
                self.bus.rendered.emit(gen, i, key, (res, to_qimage(eng.decode_preview(out, ICON)[0])))
            except Exception as ex:
                self.bus.failed.emit(gen, i, str(ex))
        self.pool.start(work)

    def on_rendered(self, gen, i, key, payload):
        if gen != self.gen:
            return
        res, img = payload
        c = self.clips[i]
        c.status, c.result, c.key, c.error = "done", res, key, ""
        self.grid.item(i).setIcon(card_icon(img))
        self.log(f"✓ {c.name}: {res['kb']} KB, {res['seconds']}s @ {res['speed']}× from {res['start']}s, {res['mode']}")
        self.job_done(i)

    def on_failed(self, gen, i, err):
        if gen != self.gen:
            return
        c = self.clips[i]
        c.status, c.error = "error", err
        self.log(f"✗ {c.name}: {err}")
        self.job_done(i)

    def job_done(self, i):
        self.pending -= 1
        self.update_progress()
        self.refresh_item(i)
        self.save_meta()
        if self.grid.currentRow() == i:
            self.show_clip(i)
        if not self.pending:
            self.total = 0
            self.update_progress()
            if self.publish_after:
                self.publish_after = False
                self.publish()

    def update_progress(self):
        if self.up_total:
            what, done, total = "Uploading", self.up_done, self.up_total
        else:
            what, done, total = "Rendering", self.total - self.pending, self.total
        self.progress.setVisible(total > 0)
        self.stop_btn.setVisible(total > 0)
        if not total:
            return
        eta = ""
        if done:
            left = int((time.monotonic() - self.t0) / done * (total - done))
            eta = f" · ~{left // 60}m {left % 60:02d}s left" if left >= 60 else f" · ~{left}s left"
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.progress.setFormat(f"{what} {done}/{total}{eta}")

    def stop(self):
        stopped = 0
        for i, c in enumerate(self.clips):
            if c.status == "queued":
                c.status, c.ticket = "pending", None
                stopped += 1
                self.refresh_item(i, counts=False)
        self.pending -= stopped
        self.publish_after = False
        if self.up_total:
            self.stop_upload = True
        self.log(f"Stopping… ({stopped} queued renders cancelled, running ones will finish)")
        if not self.pending:
            self.total = 0
        self.update_counts()
        self.update_progress()

    # ------------------------------------------------------------ grid state
    def refresh_items(self, *_):
        for i in range(len(self.clips)):
            self.refresh_item(i, counts=False)
        self.update_counts()

    def refresh_item(self, i, counts=True):
        c = self.clips[i]
        if c.status in ("queued", "working"):
            badge = "⏳ rendering…" if c.status == "working" else "⏳ queued"
        elif c.status == "error":
            badge = "⚠ failed"
        elif c.status == "done" and not self.stale(c):
            over = c.result["kb"] > eng.TG_MAX_KB
            badge = f"{'⚠' if over else '✓'} {c.result['kb']} KB · {c.result['speed']}×"
            if c.uploaded == f"{self.value('pack')}|{c.key}":
                badge += " · ✈"
        elif c.status == "done":
            badge = "↻ settings changed"
        else:
            badge = "• not rendered"
        it = self.grid.item(i)
        self.grid.blockSignals(True)
        name = c.name if len(c.name) <= 22 else c.name[:20] + "…"
        it.setText(f"{name}\n{badge}")
        it.setToolTip(c.error or str(c.path))
        self.grid.blockSignals(False)
        if counts:
            self.update_counts()

    def update_counts(self):
        self.empty.setVisible(not self.clips)
        self.grid.setVisible(bool(self.clips))
        ready = sum(1 for i in self.checked() if not self.stale(self.clips[i]))
        sent = sum(1 for c in self.clips if c.uploaded and c.uploaded == f"{self.value('pack')}|{c.key}")
        self.count_lbl.setText(f"{len(self.checked())} selected · {ready} ready · {sent} uploaded ✈ · "
                               f"{len(self.clips)} total" if self.clips else "")

    # ------------------------------------------------------------ detail panel
    def cur(self):
        r = self.grid.currentRow()
        return self.clips[r] if 0 <= r < len(self.clips) else None

    def show_clip(self, i):
        self.timer.stop()
        self.anim = []
        c = self.clips[i] if i is not None and i < len(self.clips) else None
        self.c_box.setEnabled(c is not None)
        if not c:
            self.preview.setText("Preview")
            self.preview.setPixmap(QPixmap())
            self.clip_name.setText("")
            self.clip_info.setText("")
            return
        self.clip_name.setText(c.name)
        o = self.opts_for(c)
        if c.status == "error":
            info = f"⚠ {c.error}"
        elif c.result:
            r = c.result
            info = (f"{r['kb']} KB · {r['seconds']} s · {r['speed']}× speed\n"
                    f"from {r['start']} s · focus: {r['mode']}")
            if r["kb"] > eng.TG_MAX_KB:
                info += f"\n⚠ over Telegram's {eng.TG_MAX_KB} KB limit — use @fStikBot or lower Max size"
            if self.stale(c):
                info += "\n↻ settings changed — re-render to apply"
        else:
            info = "Not rendered yet"
        self.clip_info.setText(info)
        for w, v in [(self.c_start, c.over.get("start", -1)), (self.c_speed, c.over.get("speed", 0)),
                     (self.c_zoom, c.over.get("zoom", 0))]:
            w.setValue(v)
        self.c_mode.setCurrentText(c.over.get("mode", "global"))
        self.c_shape.setCurrentText(c.over.get("shape", "global"))
        self.c_emoji.setText(c.emoji)
        self.c_emoji.setPlaceholderText(self.value("emoji") or "🙂")
        if c.status == "done" and self.out_path(c).exists():
            self.preview.setText("loading…")
            gen, out = self.gen, self.out_path(c)
            self.pool.start(lambda: self.bus.preview.emit(gen, i, [to_qimage(f) for f in eng.decode_preview(out, 300)]))
            self.anim_fps = o.fps
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText({"queued": "Queued…", "working": "Rendering…"}.get(c.status, "Not rendered"))

    def on_preview(self, gen, i, frames):
        if gen != self.gen or i != self.grid.currentRow() or not frames:
            return
        self.anim, self.anim_i = [QPixmap.fromImage(f) for f in frames], 0
        self.timer.start(int(1000 / max(1, self.anim_fps)))

    def tick(self):
        if self.anim:
            self.preview.setPixmap(self.anim[self.anim_i % len(self.anim)])
            self.anim_i += 1

    def apply_clip(self):
        i = self.grid.currentRow()
        c = self.cur()
        if not c:
            return
        over = {}
        if self.c_start.value() >= 0:
            over["start"] = self.c_start.value()
        if self.c_speed.value() > 0:
            over["speed"] = self.c_speed.value()
        if self.c_zoom.value() > 0:
            over["zoom"] = self.c_zoom.value()
        for k, w in (("mode", self.c_mode), ("shape", self.c_shape)):
            if w.currentText() != "global":
                over[k] = w.currentText()
        c.over = over
        self.save_emoji()
        if c.status not in ("queued", "working"):
            self.queue(i)

    def reset_clip(self):
        c = self.cur()
        if c:
            c.over, c.emoji = {}, ""
            self.apply_clip()

    def save_emoji(self):
        c = self.cur()
        if c:
            c.emoji = self.c_emoji.text().strip()
            self.save_meta()

    # ------------------------------------------------------------ persistence (per folder)
    def meta_file(self):
        return self.out_dir() / ".stickerforge.json"

    def load_meta(self):
        try:
            return json.loads(self.meta_file().read_text())
        except (OSError, ValueError):
            return {}

    def save_meta(self):
        if not self.folder:
            return
        data = {c.name: {"over": c.over, "emoji": c.emoji, "key": c.key, "result": c.result, "uploaded": c.uploaded,
                         "skip": self.grid.item(i).checkState() != Qt.Checked}
                for i, c in enumerate(self.clips)}
        try:
            self.meta_file().write_text(json.dumps(data, indent=1))
        except OSError:
            pass

    # ------------------------------------------------------------ telegram
    def make_pack(self):
        missing = [k for k in ("token", "user_id", "pack") if not self.value(k)]
        if missing:
            QMessageBox.warning(self, "Telegram", "Fill in the Telegram section first: " + ", ".join(missing))
            return
        if not self.value("user_id").isdigit():
            QMessageBox.warning(self, "Telegram", "User ID must be the numeric id from @userinfobot.")
            return
        if not self.checked():
            QMessageBox.information(self, "Telegram", "No stickers selected.")
            return
        self.save_meta()
        self.publish_after = True
        self.pack_btn.setEnabled(False)
        self.process_all()
        if not self.pending:
            self.publish_after = False
            self.publish()

    def publish(self):
        ready = [self.clips[i] for i in self.checked() if not self.stale(self.clips[i])]
        too_big = [c for c in ready if c.result["kb"] > eng.TG_MAX_KB]
        if too_big:
            self.pack_btn.setEnabled(True)
            QMessageBox.warning(
                self, "Too big for the Bot API",
                f"{len(too_big)} sticker(s) are over {eng.TG_MAX_KB} KB, which Telegram rejects.\n\n"
                f"Set Max size to {eng.TG_MAX_KB} KB and re-process, or send the files from the "
                "output folder to @fStikBot, which re-compresses them.")
            return
        if not ready:
            self.pack_btn.setEnabled(True)
            self.log("Nothing rendered successfully — nothing to upload.")
            return
        pack, replace_packs = self.value("pack"), self.value("replace")
        if replace_packs:  # one-shot: a later resume must not wipe what this run uploads
            self.global_widgets["replace"].setChecked(False)
            for c in self.clips:
                c.uploaded = ""
        todo = [c for c in ready if c.uploaded != f"{pack}|{c.key}"]
        if len(todo) < len(ready):
            self.log(f"Skipping {len(ready) - len(todo)} stickers already in the pack (tick Replace to redo everything)")
        if not todo:
            self.pack_btn.setEnabled(True)
            self.log("Everything is already uploaded.")
            return
        default = self.value("emoji") or "🙂"
        stickers = [(str(self.out_path(c)), c.emoji or default) for c in todo]
        args = (self.value("token"), self.value("user_id"), pack,
                self.value("title") or pack, stickers, replace_packs)
        n_packs = -(-len(ready) // eng.PACK_MAX)
        self.up_total, self.up_done, self.stop_upload, self.t0 = len(stickers), 0, False, time.monotonic()
        self.update_progress()
        self.log(f"Uploading {len(stickers)} stickers to Telegram"
                 + (f" (Telegram caps packs at {eng.PACK_MAX}, so this spans {n_packs} packs)" if n_packs > 1 else "") + " …")

        def work():
            try:
                self.bus.published.emit(eng.publish(*args, log=self.bus.log.emit,
                                                    on_added=lambda p: self.bus.uploaded.emit(p, pack),
                                                    should_stop=lambda: self.stop_upload))
            except Exception as ex:
                self.bus.publish_failed.emit(str(ex))
        self.pool.start(work)

    def on_uploaded(self, path, pack):
        for i, c in enumerate(self.clips):
            if str(self.out_path(c)) == path:
                c.uploaded = f"{pack}|{c.key}"
                self.refresh_item(i)
        self.up_done += 1
        self.update_progress()
        self.save_meta()

    def upload_finished(self):
        self.pack_btn.setEnabled(True)
        self.up_total = 0
        self.update_progress()

    def on_published(self, urls):
        self.upload_finished()
        box = QMessageBox(self)
        box.setWindowTitle("Pack ready 🎉")
        box.setTextFormat(Qt.RichText)
        links = "<br>".join(f"<a style='color:#6ab2f2' href='{u}'>{u}</a>" for u in urls)
        box.setText(f"Your sticker pack{'s are' if len(urls) > 1 else ' is'} live:<br>{links}")
        box.addButton("Open in Telegram", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        if box.exec() == 0:
            for u in urls:
                QDesktopServices.openUrl(QUrl(u))

    def on_publish_failed(self, err):
        self.upload_finished()
        self.log(f"✗ Telegram: {err}")
        if "stopped by user" in err:
            return
        hint = "\n\nTip: send /start to your bot first." if "PEER_ID_INVALID" in err or "user not found" in err.lower() else ""
        QMessageBox.critical(self, "Telegram upload failed", err + hint)

    # ------------------------------------------------------------ misc
    def log(self, msg):
        self.logbox.appendPlainText(msg)

    @staticmethod
    def open_path(p):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Sticker Forge")
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    last = w.settings.value("last_folder")
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        w.open_folder(Path(sys.argv[1]).resolve())
    elif last and Path(last).is_dir():
        w.open_folder(Path(last))
    sys.exit(app.exec())
