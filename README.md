# Telegram Sticker Maker

**Sticker Forge**: a desktop app that turns a folder of videos into Telegram video sticker packs, mostly automatically.

```sh
git clone https://github.com/annashumate1/Telegram-Sticker-Maker && cd Telegram-Sticker-Maker
uv run stickermaker.py            # GUI (deps auto-install on first run)
uv run stickermaker.py ~/Videos   # GUI, open a folder straight away
uv run sticker_engine.py DIR      # headless: render every video in DIR to DIR/stickers
uv run sticker_engine.py --selftest
```
Needs [uv](https://docs.astral.sh/uv/) and `ffmpeg` with libvpx-vp9. Python dependencies (PySide6, OpenCV, numpy, requests) install automatically. `python stickermaker.py` also works; it relaunches itself through uv.

**What it does by default**
- Scans the whole video and picks the best 3 s: lively, faces on screen, no scene cuts.
- Tracks the face (YuNet) or body (HOG), holds the same subject and pans smoothly. With no subject it center-crops.
- Auto speed: speeds up static clips (max 2x) and slows down frantic ones (0.75x).
- Encodes VP9 WEBM at 512x512, max 3 s, max 30 fps, no audio, and drops the bitrate until the file fits the size cap.
- Optional extras: rounded or circle shape (transparent corners), ping-pong loop, several moments per long video.
- Per-sticker overrides: start, speed, focus, zoom, shape and emoji. Work is saved in `stickers/.stickerforge.json`, so reopening a folder doesn't re-render.

**Convenience:** the app reopens your last folder on launch, and there's an option to include subfolders. You get a Stop button, an ETA for rendering and uploading, and Select all / none. Double-click a card to play it; right-click for re-render, open sticker or source, or show the folder. Cards marked as uploaded are already in your pack.

**Telegram pack:** create a bot with @BotFather, get your numeric ID from @userinfobot, send `/start` to your bot, fill in the Telegram box, then click **Make Pack**.
Telegram's Bot API rejects video stickers over 256 KB. If you raise *Max size* (up to 15 MB), send the files from `stickers/` to @fStikBot instead; it re-compresses them.
The bot token is stored in plain text in `~/.config/stickerforge/stickerforge.conf`.

## Terms and Conditions

By downloading and using this program you agree that Anna is beautiful and deserves every ounce of respect you have.

![annamainpic](https://github.com/user-attachments/assets/e66ffc59-f920-4a9d-b4cb-d18a11482e3e)
