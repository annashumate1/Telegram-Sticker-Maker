# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python-headless>=4.8,<5", "numpy", "requests"]
# ///
"""Video -> Telegram video sticker engine (no Qt here; the GUI is stickermaker.py).

Pipeline per sticker:
  analyze()  - sample the whole video at 4 fps: motion, scene cuts, faces
  render()   - pick the best moment, decide speed, track face/body, crop to 512x512,
               shape mask, encode VP9 WEBM and shrink bitrate until it fits the size cap
  publish()  - create / extend a sticker set through the Telegram Bot API

CLI:  python sticker_engine.py DIR            render every video in DIR to DIR/stickers
      python sticker_engine.py --selftest     synthetic end-to-end check
"""
import json
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".3gp", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}
SIDE = 512                  # Telegram: one side exactly 512 px
TG_MAX_KB = 256             # Telegram Bot API hard limit for video stickers
PACK_MAX = 120              # Telegram limit of stickers per video sticker set
SAMPLE_FPS = 4              # analysis sampling rate
CUT = 0.18                  # ponytail: mean abs-diff above this = scene cut; tune if cuts slip through
STATIC, FRANTIC = 0.008, 0.07  # ponytail: auto-speed calibration (typical motion per 1/4 s); speed up below, slow-mo above
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
CACHE = Path.home() / ".cache" / "stickerforge"


@dataclass
class Opts:
    mode: str = "auto"       # auto | face | body | center
    speed: float = 0.0       # 0 = auto
    duration: float = 3.0    # seconds, Telegram max 3
    fps: int = 30            # Telegram max 30
    zoom: float = 1.0        # >1 = tighter framing
    shape: str = "square"    # square | rounded | circle
    boomerang: bool = False  # play forward then backward (seamless loop)
    max_kb: int = TG_MAX_KB
    start: float = -1.0      # -1 = auto pick best moment
    segment: int = 0         # k-th best non-overlapping moment of the video


# ---------------------------------------------------------------- video io

def probe(path):
    """-> (duration_s, width, height) with phone rotation applied."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_tags=rotate:stream_side_data=rotation:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    if not j.get("streams"):
        raise ValueError("no video stream")
    s = j["streams"][0]
    w, h = s["width"], s["height"]
    rot = s.get("tags", {}).get("rotate") or next(
        (sd["rotation"] for sd in s.get("side_data_list", []) if "rotation" in sd), 0)
    if abs(int(float(rot))) % 180 == 90:
        w, h = h, w
    try:
        dur = float(j.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        dur = 0.0
    return dur, w, h


def fit(w, h, longest):
    """Scale (w, h) so the longer side is <= longest, keeping even dims."""
    s = min(1.0, longest / max(w, h))
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


def read_frames(path, w, h, start=0.0, length=None, fps=None, speed=1.0):
    """Yield BGR frames (h, w, 3) decoded by ffmpeg, already retimed by `speed`."""
    vf = [f"setpts=(PTS-STARTPTS)/{speed}"] if speed != 1 else []
    vf += [f"fps={fps}"] if fps else []
    vf += [f"scale={w}:{h}"]
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if length:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-i", str(path), "-an", "-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        while len(buf := p.stdout.read(n)) == n:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.kill()
        p.wait()


def thumbnail(path, size=200):
    dur, w, h = probe(path)
    tw, th = fit(w, h, size)
    return next(read_frames(path, tw, th, start=dur / 3 if dur > 1 else 0), None)


def decode_preview(path, size=256):
    """Decode a rendered sticker (keeping alpha) -> list of BGRA frames."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-c:v", "libvpx-vp9", "-i", str(path),
           "-vf", f"scale={size}:{size}", "-f", "rawvideo", "-pix_fmt", "bgra", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, size, size, 4))


# ---------------------------------------------------------------- detection

_tls = threading.local()
_dl_lock = threading.Lock()


def _face_detector():
    if not hasattr(_tls, "face"):
        model = CACHE / "yunet.onnx"
        with _dl_lock:
            if not model.exists():
                try:
                    CACHE.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(YUNET_URL, model.with_suffix(".part"))
                    model.with_suffix(".part").rename(model)
                except OSError:
                    pass
        if model.exists():
            _tls.face = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.7)
        else:  # offline fallback: bundled Haar cascade (worse on angled faces)
            _tls.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    return _tls.face


def detect_face(img):
    """-> list of (x, y, w, h), most prominent first."""
    det = _face_detector()
    if isinstance(det, cv2.CascadeClassifier):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(g, 1.1, 6, minSize=(min(g.shape) // 12,) * 2)
        return sorted(map(tuple, faces), key=lambda f: -f[2] * f[3])
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None:
        return []
    return [tuple(f[:4]) for f in sorted(faces, key=lambda f: -f[2] * f[3] * f[14])]


def detect_body(img):
    if not hasattr(_tls, "hog"):
        _tls.hog = cv2.HOGDescriptor()
        _tls.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = _tls.hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return sorted(map(tuple, rects), key=lambda r: -r[2] * r[3])


# ---------------------------------------------------------------- analysis

@lru_cache(maxsize=64)
def analyze(path):
    """-> (times, motion, face_present) sampled at SAMPLE_FPS over the whole video."""
    dur, w, h = probe(path)
    sw, sh = fit(w, h, 320)
    motion, faces, prev = [], [], None
    for f in read_frames(path, sw, sh, fps=SAMPLE_FPS):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        motion.append(0.0 if prev is None else float(cv2.absdiff(g, prev).mean()) / 255)
        faces.append(bool(detect_face(f)))
        prev = g
    if not motion:
        raise ValueError("could not decode video")
    m = np.array(motion)
    if len(m) > 1:
        m[0] = m[1]
    return np.arange(len(m)) / SAMPLE_FPS, m, np.array(faces, float)


def auto_speed(path, out_dur):
    dur, _, _ = probe(path)
    _, m, _ = analyze(path)
    calm = m[m < CUT]
    typical = float(np.median(calm)) if len(calm) else 0.02
    speed = min(2.0, STATIC / max(typical, 1e-4)) if typical < STATIC else 0.75 if typical > FRANTIC else 1.0
    if dur:  # don't need more source than exists
        speed = min(speed, max(1.0, dur / out_dur))
    return round(speed * 4) / 4


def best_starts(path, window, k):
    """Start times of the k+1 best non-overlapping windows (lively, faces, no cuts)."""
    t, m, f = analyze(path)
    n = max(1, int(round(window * SAMPLE_FPS)))
    if len(t) <= n:
        return [0.0]
    cut = m >= CUT
    lively = np.clip(m / (np.percentile(m[~cut], 90) if (~cut).any() else 1) + 1e-6, 0, 1)
    score = 0.5 * np.where(cut, 0, lively) + 0.5 * f
    # cut at sample i is a jump between i-1 and i: penalise windows that contain one after their first frame
    c = np.concatenate([[0], np.cumsum(cut)])
    starts = np.arange(len(t) - n + 1)
    win = np.convolve(score, np.ones(n) / n, "valid") - 2 * (c[starts + n] - c[starts + 1])
    picks = []
    for i in np.argsort(-win):
        if all(abs(i - j) >= n for j in picks):
            picks.append(int(i))
            if len(picks) > k:
                break
    return [float(t[i]) for i in picks]


def segment_count(path, opts):
    """How many distinct moments of `opts.duration` the video can give (rough, pre-analysis)."""
    dur, _, _ = probe(path)
    return max(1, int(dur // (opts.duration * (opts.speed or 2.0)))) if dur else 1  # 2.0 = max auto speed


# ---------------------------------------------------------------- framing

def _smooth(x, k=15):
    if len(x) < 3:
        return x
    k = min(k, len(x) // 2 * 2 - 1)
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, "valid")


def crop_track(frames, mode, zoom):
    """-> (mode_used, [(cx, cy, side)]) normalised: cx/w, cy/h, side/min(w,h)."""
    h, w = frames[0].shape[:2]
    short = min(w, h)
    n = len(frames)
    idx = list(range(0, n, 2))
    # (detector, crop size multiplier of box, vertical shift of centre in box heights)
    plans = {"face": (detect_face, 2.6, 0.15), "body": (detect_body, 1.1, 0.0)}
    order = {"auto": ["face", "body"], "face": ["face"], "body": ["body"]}.get(mode, [])
    for name in order:
        det, mult, shift = plans[name]
        hits, last = [], None
        for i in idx:  # stick with the same subject: nearest to the previous pick
            boxes = det(frames[i])
            if boxes:
                last = boxes[0] if last is None else min(
                    boxes, key=lambda b: abs(b[0] + b[2] / 2 - last[0] - last[2] / 2) + abs(b[1] + b[3] / 2 - last[1] - last[3] / 2))
                hits.append((i, last))
        if len(hits) < 0.3 * len(idx):
            continue
        ii = [i for i, _ in hits]
        b = np.array([bb for _, bb in hits], float)
        cx = np.interp(range(n), ii, b[:, 0] + b[:, 2] / 2)
        cy = np.interp(range(n), ii, b[:, 1] + b[:, 3] / 2 + shift * b[:, 3])
        side = np.median(np.maximum(b[:, 2], b[:, 3])) * mult / zoom
        side = float(np.clip(side, 0.35 * short, short))
        cx = np.clip(_smooth(cx), side / 2, w - side / 2)
        cy = np.clip(_smooth(cy), side / 2, h - side / 2)
        return name, [(x / w, y / h, side / short) for x, y in zip(cx, cy)]
    side = min(1.0, 1 / zoom)
    return "center", [(0.5, 0.5, side)] * n


_masks = {}


def shape_mask(shape):
    if shape not in _masks:
        big = SIDE * 4  # draw at 4x then downsample = anti-aliased edge
        m = np.zeros((big, big), np.uint8)
        if shape == "circle":
            cv2.circle(m, (big // 2, big // 2), big // 2, 255, -1)
        elif shape == "rounded":
            r = big // 6
            cv2.rectangle(m, (r, 0), (big - r, big), 255, -1)
            cv2.rectangle(m, (0, r), (big, big - r), 255, -1)
            for x in (r, big - r):
                for y in (r, big - r):
                    cv2.circle(m, (x, y), r, 255, -1)
        else:
            m[:] = 255
        _masks[shape] = cv2.resize(m, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    return _masks[shape]


# ---------------------------------------------------------------- encode

def encode(frames, out, fps, max_kb, alpha):
    """VP9 WEBM, lowering bitrate until the file fits max_kb. -> size in KB."""
    dur = len(frames) / fps
    kbps = min(3000, max_kb * 8 * 0.92 / dur)
    data = b"".join(f.tobytes() for f in frames)
    out = Path(out)
    for _ in range(10):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{SIDE}x{SIDE}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libvpx-vp9",
               "-pix_fmt", "yuva420p" if alpha else "yuv420p",
               "-crf", "30", "-b:v", f"{int(kbps)}k", "-deadline", "good", "-cpu-used", "2", "-row-mt", "1",
               str(out)]
        subprocess.run(cmd, input=data, check=True, capture_output=True)
        kb = out.stat().st_size / 1024
        if kb <= max_kb:
            return kb
        kbps *= 0.85 * max_kb / kb
    raise RuntimeError(f"could not get under {max_kb} KB (last {kb:.0f} KB)")


# ---------------------------------------------------------------- render

def render(path, out, o: Opts, progress=lambda frac: None):
    """Make one sticker. -> dict(start, speed, mode, kb, seconds)."""
    path = str(path)
    dur, W, H = probe(path)
    speed = o.speed or auto_speed(path, o.duration)
    n_out = max(1, int(round(min(o.duration, 3.0) * min(o.fps, 30))))
    n_fwd = n_out // 2 + 1 if o.boomerang else n_out
    window = n_fwd / o.fps * speed
    if dur:
        window = min(window, dur)
    progress(0.1)
    if o.start >= 0:
        start = o.start
    else:
        starts = best_starts(path, window, o.segment)
        if o.segment >= len(starts):
            raise ValueError(f"video too short for moment #{o.segment + 1}")
        start = starts[o.segment]
    if dur:
        start = max(0.0, min(start, dur - window))
    progress(0.3)

    dw, dh = fit(W, H, 640)
    small = list(read_frames(path, dw, dh, start, window, o.fps, speed))[:n_fwd]
    if not small:
        raise ValueError("no frames in selected range")
    mode, track = crop_track(small, o.mode, o.zoom)
    del small
    progress(0.6)

    rw, rh = fit(W, H, 1920)
    short = min(rw, rh)
    mask = shape_mask(o.shape)
    frames = []
    for (cx, cy, s), fr in zip(track, read_frames(path, rw, rh, start, window, o.fps, speed)):
        s = int(round(s * short))
        x0 = int(np.clip(round(cx * rw - s / 2), 0, rw - s))
        y0 = int(np.clip(round(cy * rh - s / 2), 0, rh - s))
        img = cv2.resize(fr[y0:y0 + s, x0:x0 + s], (SIDE, SIDE),
                         interpolation=cv2.INTER_AREA if s > SIDE else cv2.INTER_CUBIC)
        frames.append(np.dstack([img, mask]))
    if o.boomerang and len(frames) > 2:
        frames = frames + frames[-2:0:-1]
    frames = frames[:n_out]
    progress(0.8)
    kb = encode(frames, out, o.fps, o.max_kb, alpha=o.shape != "square")
    progress(1.0)
    return {"start": round(start, 2), "speed": speed, "mode": mode, "kb": round(kb), "seconds": round(len(frames) / o.fps, 2)}


# ---------------------------------------------------------------- telegram

class Telegram:
    def __init__(self, token):
        self.base = f"https://api.telegram.org/bot{token.strip()}/"

    def call(self, method, files=None, **params):
        for _ in range(6):
            r = requests.post(self.base + method, data=params, files=files, timeout=180)
            try:
                j = r.json()
            except ValueError:
                raise RuntimeError(f"{method}: HTTP {r.status_code}")
            if j.get("ok"):
                return j["result"]
            wait = j.get("parameters", {}).get("retry_after")
            if not wait:
                raise RuntimeError(f"{method}: {j.get('description')}")
            time.sleep(wait + 1)
            for fh in (files or {}).values():
                fh.seek(0)
        raise RuntimeError(f"{method}: rate limited")


def pack_name(pack, bot_username):
    suffix = f"_by_{bot_username}".lower()
    s = re.sub(r"[^a-z0-9_]+", "_", pack.lower()).strip("_")
    s = re.sub(r"_+", "_", s.removesuffix(suffix))
    if not s or not s[0].isalpha():
        s = "pack_" + s
    return s[:64 - len(suffix)].rstrip("_") + suffix


def emoji_list(s):
    return s.split()[:20] or ["🙂"]


def publish(token, user_id, pack, title, stickers, replace=False, log=print, on_added=lambda path: None,
            should_stop=lambda: False):
    """stickers: [(webm_path, "😀 🔥")]. Fills pack, pack_2, pack_3… (PACK_MAX each). -> list of t.me links.
    Without `replace`, existing packs are topped up; on_added(path) fires after each sticker lands."""
    tg = Telegram(token)
    bot = tg.call("getMe")["username"]
    stem = pack_name(pack, bot).removesuffix(f"_by_{bot}".lower())
    user_id = int(user_id)

    def nth(k):
        return pack_name(stem if k == 1 else f"{stem}_{k}", bot)

    def count(name):  # stickers in the set, None if it doesn't exist
        try:
            return len(tg.call("getStickerSet", name=name)["stickers"])
        except RuntimeError:
            return None

    def inp(key, emo):
        return {"sticker": f"attach://{key}", "format": "video", "emoji_list": emoji_list(emo)}

    if replace:
        k = 1
        while count(nth(k)) is not None:
            log(f"Deleting existing pack {nth(k)}")
            tg.call("deleteStickerSet", name=nth(k))
            k += 1

    todo, urls, k = list(stickers), [], 0
    while todo:
        k += 1
        name = nth(k)
        n = count(name)
        if n is None:
            if should_stop():
                raise RuntimeError("stopped by user")
            first, todo = todo[:50], todo[50:]
            files = {f"f{i}": open(p, "rb") for i, (p, _) in enumerate(first)}
            try:
                log(f"Creating pack {name} with {len(first)} stickers…")
                tg.call("createNewStickerSet", files=files, user_id=user_id, name=name,
                        title=title[:60] + (f" {k}" if k > 1 else ""),
                        stickers=json.dumps([inp(f"f{i}", e) for i, (_, e) in enumerate(first)]))
            finally:
                for fh in files.values():
                    fh.close()
            for p, _ in first:
                on_added(p)
            n = len(first)
        batch, todo = todo[:max(0, PACK_MAX - n)], todo[max(0, PACK_MAX - n):]
        for i, (p, e) in enumerate(batch, 1):
            if should_stop():
                raise RuntimeError("stopped by user")
            log(f"{name}: adding {i}/{len(batch)} {Path(p).name}")
            with open(p, "rb") as fh:
                tg.call("addStickerToSet", files={"f": fh}, user_id=user_id, name=name, sticker=json.dumps(inp("f", e)))
            on_added(p)
        if batch or n:
            urls.append(f"https://t.me/addstickers/{name}")
            try:  # drop one sticker of the pack into the user's chat with the bot, so it's saved there
                tg.call("sendSticker", chat_id=user_id, sticker=tg.call("getStickerSet", name=name)["stickers"][0]["file_id"])
            except (RuntimeError, IndexError, KeyError) as ex:
                log(f"couldn't send {name} to chat: {ex}")
    log("Done: " + "  ".join(urls))
    return urls


# ---------------------------------------------------------------- cli / self-check

def videos_in(folder, recursive=False):
    folder = Path(folder)
    files = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in files if p.suffix.lower() in VIDEO_EXTS and p.is_file()
                  and "stickers" not in p.relative_to(folder).parts[:-1])


def _test_publish():
    """Fake Bot API that enforces PACK_MAX: 255 stickers onto a pack holding 100 -> top up, then spill into _2, _3."""
    sets, sent = {"pk_by_bot": 100}, []

    def fake(self, method, files=None, **kw):
        name = kw.get("name")
        if method == "getMe":
            return {"username": "Bot"}
        if method == "getStickerSet":
            if name not in sets:
                raise RuntimeError("STICKERSET_INVALID")
            return {"stickers": [{"file_id": name}] * sets[name]}
        if method == "createNewStickerSet":
            assert name not in sets and len(json.loads(kw["stickers"])) <= 50
            sets[name] = len(json.loads(kw["stickers"]))
        if method == "addStickerToSet":
            assert sets[name] < PACK_MAX, "STICKERPACK_STICKERS_TOO_MUCH"
            sets[name] += 1
        if method == "deleteStickerSet":
            del sets[name]
        if method == "sendSticker":
            sent.append(kw["sticker"])
        return True
    real, Telegram.call = Telegram.call, fake
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm") as f:
            added = []
            urls = publish("t", "1", "pk", "Pk", [(f.name, "🙂")] * 255, log=lambda m: None, on_added=added.append)
            assert sets == {"pk_by_bot": 120, "pk_2_by_bot": 120, "pk_3_by_bot": 115}, sets
            assert len(urls) == 3 and len(added) == 255
            assert sent == ["pk_by_bot", "pk_2_by_bot", "pk_3_by_bot"], sent
            publish("t", "1", "pk", "Pk", [(f.name, "🙂")] * 10, replace=True, log=lambda m: None)
            assert sets == {"pk_by_bot": 10}, sets
    finally:
        Telegram.call = real


def _selftest():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "src.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=8",
                        "-pix_fmt", "yuv420p", str(src)], check=True)
        for o in (Opts(), Opts(shape="circle", boomerang=True, speed=1.5, max_kb=120)):
            out = Path(d) / "out.webm"
            res = render(src, out, o)
            dur, w, h = probe(out)
            assert (w, h) == (SIDE, SIDE), (w, h)
            assert dur <= 3.0 + 1e-3 and res["seconds"] <= 3.0, dur
            assert out.stat().st_size <= o.max_kb * 1024, res
            assert len(decode_preview(out, 64)) == int(res["seconds"] * o.fps)
            print("ok", o.shape, res)
    _test_publish()
    assert pack_name("My Pack!", "CoolBot") == "my_pack_by_coolbot"
    assert pack_name("9lives_by_coolbot", "CoolBot") == "pack_9lives_by_coolbot"
    print("selftest passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    for k, v in asdict(Opts()).items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else lambda s: s.lower() in ("1", "true", "yes"), default=v)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.folder:
        opts = Opts(**{k: getattr(a, k) for k in asdict(Opts())})
        outdir = Path(a.folder) / "stickers"
        outdir.mkdir(exist_ok=True)
        for v in videos_in(a.folder):
            try:
                print(v.name, render(v, outdir / f"{v.stem}_0.webm", opts))
            except Exception as e:
                print(v.name, "FAILED:", e)
    else:
        ap.print_help()
