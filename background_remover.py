"""
Demonstrates: Image segmentation (GrabCut = region growing + edge-based),
              Iterative mask refinement, Foreground/background classification

Run: python background_remover.py

Requirements:
    pip install opencv-python numpy Pillow
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#0d0d14"
CARD    = "#14141f"
ACCENT  = "#7c6ff7"
ACCENT2 = "#4ecdc4"
TXT     = "#e8e8f0"
TXT_DIM = "#6b6b80"
BORDER  = "#252535"
GREEN   = "#56cf8a"
ORANGE  = "#f5a623"
RED     = "#f25f5c"
PINK    = "#ff6b9d"

WIN_W, WIN_H = 1300, 840

# GrabCut mask values
GC_BGD    = cv2.GC_BGD     # 0 definite background
GC_FGD    = cv2.GC_FGD     # 1 definite foreground
GC_PR_BGD = cv2.GC_PR_BGD  # 2 probable background
GC_PR_FGD = cv2.GC_PR_FGD  # 3 probable foreground

MODES = ["rect", "fg", "bg"]   # drawing modes

CAPTIONS = {
    "Original":
        "① Draw a rectangle around your subject. The algorithm treats "
        "everything outside as background, inside as probable foreground.",
    "Segmentation Mask":
        "② GrabCut mask: white = definite FG, grey = probable FG, "
        "dark = background. This is the region growing result.",
    "Cutout":
        "③ Final result with background removed. Paint green (FG) or "
        "red (BG) strokes on the original to refine the boundary.",
}


def make_demo():
    img = np.full((400, 500, 3), 70, dtype=np.uint8)
    cv2.ellipse(img, (250, 200), (100, 130), 0, 0, 360, (200, 160, 130), -1)
    cv2.circle(img, (250, 115), 55, (210, 170, 140), -1)
    cv2.putText(img, "Load a photo to begin", (90, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 170), 1)
    return img


def fit(bgr, max_w, max_h):
    h, w   = bgr.shape[:2]
    scale  = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), scale


def to_tk(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def checkerboard(h, w):
    """Grey checkerboard to show transparency."""
    board = np.full((h, w, 3), 180, dtype=np.uint8)
    sz = 12
    for y in range(0, h, sz):
        for x in range(0, w, sz):
            if (x // sz + y // sz) % 2 == 0:
                board[y:y+sz, x:x+sz] = 210
    return board


class BackgroundRemover(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Background Remover")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.configure(bg=BG)
        self.resizable(True, True)

        # image state
        self.orig_bgr   = make_demo()           # full-res original
        self.disp_bgr   = None                  # resized for display
        self.disp_scale = 1.0
        self.mask       = None                  # GrabCut mask (full-res)
        self.bgd_model  = None
        self.fgd_model  = None
        self.rect       = None                  # (x,y,w,h) in display coords
        self.has_rect   = False
        self.gc_done    = False

        # drawing state
        self.mode       = tk.StringVar(value="rect")  # rect / fg / bg
        self.brush_r    = tk.IntVar(value=8)
        self.iter_n     = tk.IntVar(value=5)
        self._drawing   = False
        self._rx0 = self._ry0 = 0
        self._stroke_pts = []

        self._refs = []
        self._status = tk.StringVar(value="Step 1: Load a photo, then draw a rectangle around your subject.")

        self._build_ui()
        self._show_original()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # top bar
        bar = tk.Frame(self, bg="#0a0a12", height=56)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="Background Remover",
                 font=("Georgia", 17, "bold"),
                 bg="#0a0a12", fg=TXT).pack(side="left", padx=20, pady=10)
        tk.Label(bar, text="Segmentation & Region Growing",
                 font=("Courier", 10), bg="#0a0a12", fg=ACCENT).pack(side="left", padx=4)

        tk.Button(bar, text="Save PNG", command=self._save,
                  bg=GREEN, fg="#0d0d14", font=("Helvetica", 10, "bold"),
                  relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2").pack(side="right", padx=8, pady=10)
        tk.Button(bar, text="Reset", command=self._reset,
                  bg="#333344", fg=TXT, font=("Helvetica", 10, "bold"),
                  relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2").pack(side="right", padx=4, pady=10)
        tk.Button(bar, text="Load Photo", command=self._load,
                  bg=ACCENT, fg="white", font=("Helvetica", 10, "bold"),
                  relief="flat", bd=0, padx=14, pady=6,
                  cursor="hand2").pack(side="right", padx=4, pady=10)

        # status bar
        sb = tk.Frame(self, bg="#0a0a12", height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self._status,
                 bg="#0a0a12", fg=TXT_DIM, font=("Courier", 9)).pack(
                 side="left", padx=16, pady=4)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # ── left: canvas area ─────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # canvas (original + drawing)
        orig_card = tk.Frame(left, bg=CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        orig_card.pack(fill="both", expand=False)

        tk.Label(orig_card, text="①  Original  —  draw here",
                 bg=CARD, fg=ACCENT2,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=10, pady=(8,2))

        self.canvas = tk.Canvas(orig_card, bg="#111120",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(padx=6, pady=4)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        tk.Label(orig_card, text=CAPTIONS["Original"],
                 bg=CARD, fg=TXT_DIM, font=("Helvetica", 9),
                 wraplength=560, justify="left").pack(anchor="w", padx=10, pady=(0,8))

        # bottom two panels side by side
        bot = tk.Frame(left, bg=BG)
        bot.pack(fill="both", expand=True, pady=(6,0))

        self._mask_card  = self._small_card(bot, "②  Segmentation Mask", "Segmentation Mask", 0)
        self._cut_card   = self._small_card(bot, "③  Cutout (transparent BG)", "Cutout", 1)

        # ── right: sidebar ────────────────────────────────────────────────
        self._build_sidebar(body)

    def _small_card(self, parent, title, key, col):
        f = tk.Frame(parent, bg=CARD,
                     highlightbackground=BORDER, highlightthickness=1)
        f.grid(row=0, column=col, padx=(0 if col else 0, 6 if col else 6),
               pady=0, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)

        tk.Label(f, text=title, bg=CARD, fg=ACCENT2,
                 font=("Courier", 10, "bold")).pack(anchor="w", padx=10, pady=(8,2))

        img_lbl = tk.Label(f, bg=CARD)
        img_lbl.pack(padx=6, pady=2)

        stat_lbl = tk.Label(f, text="", bg=CARD, fg=ORANGE, font=("Courier", 9))
        stat_lbl.pack(anchor="w", padx=10)

        tk.Label(f, text=CAPTIONS[key], bg=CARD, fg=TXT_DIM,
                 font=("Helvetica", 9), wraplength=280,
                 justify="left").pack(anchor="w", padx=10, pady=(2,8))

        return img_lbl, stat_lbl

    def _build_sidebar(self, parent):
        side = tk.Frame(parent, bg=CARD, width=250,
                        highlightbackground=BORDER, highlightthickness=1)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="Tools", bg=CARD, fg=TXT,
                 font=("Georgia", 13, "bold")).pack(pady=(18,4), padx=16, anchor="w")
        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)

        # mode buttons
        modes = [
            ("rect", "☐  Draw Rectangle",  ACCENT,  "Step 1 — mark the subject area"),
            ("fg",   "●  Paint Foreground", GREEN,   "Green strokes = keep this"),
            ("bg",   "●  Paint Background", RED,     "Red strokes = remove this"),
        ]
        self._mode_btns = {}
        for val, label, col, hint in modes:
            btn = tk.Button(side, text=label, relief="flat", bd=0,
                            font=("Helvetica", 10, "bold"),
                            bg=col if self.mode.get()==val else "#252535",
                            fg="white", padx=10, pady=6, cursor="hand2",
                            command=lambda v=val: self._set_mode(v))
            btn.pack(fill="x", padx=16, pady=3)
            self._mode_btns[val] = (btn, col, hint)

        self._hint_lbl = tk.Label(side, text="Step 1 — mark the subject area",
                                  bg=CARD, fg=TXT_DIM, font=("Helvetica", 9),
                                  wraplength=210, justify="left")
        self._hint_lbl.pack(padx=16, anchor="w", pady=(2,0))

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)

        # brush size
        self._add_slider(side, "Brush Size", self.brush_r, 2, 30, 1,
                         "Stroke radius for FG/BG painting")
        self._add_slider(side, "GrabCut Iterations", self.iter_n, 1, 15, 1,
                         "More = more accurate, slower")

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=10)

        # how it works
        tk.Label(side, text="How GrabCut works", bg=CARD, fg=TXT,
                 font=("Georgia", 11, "bold")).pack(padx=16, anchor="w")

        steps = [
            (ACCENT,  "Rectangle seeds\nprobable FG region"),
            (ACCENT2, "Gaussian Mixture Models\nlearn FG & BG colours"),
            (GREEN,   "Graph cut finds optimal\nboundary between them"),
            (ORANGE,  "Paint strokes override\nmask to fix mistakes"),
            (RED,     "Algorithm re-runs with\nnew hard constraints"),
        ]
        for col, txt in steps:
            r = tk.Frame(side, bg=CARD)
            r.pack(fill="x", padx=16, pady=2)
            tk.Label(r, text="▶", bg=CARD, fg=col,
                     font=("Helvetica", 9)).pack(side="left", padx=(0,6), anchor="n")
            tk.Label(r, text=txt, bg=CARD, fg=TXT_DIM,
                     font=("Helvetica", 9), justify="left").pack(side="left", anchor="w")

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)
        tk.Label(side,
                 text="concepts shown:\nRegion growing · Edge-based\nsegmentation · Grouping",
                 bg=CARD, fg=ACCENT, font=("Helvetica", 9, "italic"),
                 justify="left").pack(padx=16, anchor="w")

    def _add_slider(self, parent, label, var, mn, mx, res, hint):
        grp = tk.Frame(parent, bg=CARD)
        grp.pack(fill="x", padx=16, pady=(0,10))
        hdr = tk.Frame(grp, bg=CARD)
        hdr.pack(fill="x")
        tk.Label(hdr, text=label, bg=CARD, fg=TXT,
                 font=("Helvetica", 10, "bold")).pack(side="left")
        vl = tk.Label(hdr, text=str(var.get()), bg=CARD, fg=ACCENT,
                      font=("Courier", 10, "bold"))
        vl.pack(side="right")
        tk.Label(grp, text=hint, bg=CARD, fg=TXT_DIM,
                 font=("Helvetica", 8)).pack(anchor="w")
        tk.Scale(grp, from_=mn, to=mx, resolution=res, variable=var,
                 orient="horizontal", bg=CARD, fg=TXT, troughcolor=BORDER,
                 activebackground=ACCENT, highlightthickness=0,
                 showvalue=False, relief="flat",
                 command=lambda v, vl=vl, vr=var: vl.config(
                     text=str(vr.get()))).pack(fill="x")

    # ── mode ─────────────────────────────────────────────────────────────────

    def _set_mode(self, val):
        self.mode.set(val)
        for v, (btn, col, hint) in self._mode_btns.items():
            btn.config(bg=col if v == val else "#252535")
        _, _, hint = self._mode_btns[val]
        self._hint_lbl.config(text=hint)
        cur = "crosshair" if val == "rect" else "pencil"
        self.canvas.config(cursor=cur)

    # ── canvas drawing ────────────────────────────────────────────────────────

    def _on_press(self, e):
        self._drawing = True
        if self.mode.get() == "rect":
            self._rx0, self._ry0 = e.x, e.y
        else:
            self._stroke_pts = [(e.x, e.y)]
            self._paint_dot(e.x, e.y)

    def _on_drag(self, e):
        if not self._drawing: return
        if self.mode.get() == "rect":
            self._redraw_canvas()
            col = ACCENT
            self.canvas.create_rectangle(
                self._rx0, self._ry0, e.x, e.y,
                outline=col, width=2, dash=(6,3))
        else:
            self._stroke_pts.append((e.x, e.y))
            self._paint_dot(e.x, e.y)

    def _on_release(self, e):
        if not self._drawing: return
        self._drawing = False
        if self.mode.get() == "rect":
            x1, y1 = min(self._rx0, e.x), min(self._ry0, e.y)
            x2, y2 = max(self._rx0, e.x), max(self._ry0, e.y)
            w, h   = x2 - x1, y2 - y1
            if w > 10 and h > 10:
                self.rect = (x1, y1, w, h)
                self.has_rect = True
                self._run_grabcut_rect()
        else:
            if self._stroke_pts and self.gc_done:
                self._apply_stroke(self._stroke_pts)

    def _paint_dot(self, x, y):
        """Draw a dot on canvas for visual feedback."""
        r   = self.brush_r.get()
        col = GREEN if self.mode.get() == "fg" else RED
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill=col, outline="", stipple="gray50")

    # ── GrabCut ───────────────────────────────────────────────────────────────

    def _disp_to_full(self, x, y):
        """Convert display canvas coordinates to full-res image coordinates."""
        s = self.disp_scale
        return int(x / s), int(y / s)

    def _run_grabcut_rect(self):
        if self.disp_bgr is None: return
        h, w = self.orig_bgr.shape[:2]
        self.mask       = np.zeros((h, w), dtype=np.uint8)
        self.bgd_model  = np.zeros((1, 65), dtype=np.float64)
        self.fgd_model  = np.zeros((1, 65), dtype=np.float64)

        rx, ry, rw, rh  = self.rect
        # convert to full-res
        fx, fy           = self._disp_to_full(rx, ry)
        fw               = int(rw / self.disp_scale)
        fh               = int(rh / self.disp_scale)
        fw               = max(fw, 1);  fh = max(fh, 1)

        rect_full = (fx, fy, fw, fh)
        try:
            cv2.grabCut(self.orig_bgr, self.mask, rect_full,
                        self.bgd_model, self.fgd_model,
                        self.iter_n.get(), cv2.GC_INIT_WITH_RECT)
            self.gc_done = True
            self._update_outputs()
            self._status.set(
                "Step 2: Paint green (FG) or red (BG) strokes to refine the mask, "
                "then save.")
        except Exception as ex:
            self._status.set(f"GrabCut error: {ex}")

    def _apply_stroke(self, pts):
        """Apply painted FG/BG strokes to the mask and re-run GrabCut."""
        is_fg = self.mode.get() == "fg"
        val   = GC_FGD if is_fg else GC_BGD
        r     = max(1, int(self.brush_r.get() / self.disp_scale))

        for dx, dy in pts:
            fx, fy = self._disp_to_full(dx, dy)
            cv2.circle(self.mask, (fx, fy), r, val, -1)

        try:
            cv2.grabCut(self.orig_bgr, self.mask, None,
                        self.bgd_model, self.fgd_model,
                        self.iter_n.get(), cv2.GC_INIT_WITH_MASK)
            self._update_outputs()
        except Exception as ex:
            self._status.set(f"Refinement error: {ex}")

    # ── rendering ─────────────────────────────────────────────────────────────

    def _update_outputs(self):
        if self.mask is None: return

        # binary mask: foreground = 1
        fg_mask = np.where((self.mask == GC_FGD) | (self.mask == GC_PR_FGD),
                           255, 0).astype(np.uint8)

        # mask visualisation (colour-coded)
        mask_vis = np.zeros((*self.mask.shape, 3), dtype=np.uint8)
        mask_vis[self.mask == GC_FGD]    = [80, 220, 80]    # definite FG: green
        mask_vis[self.mask == GC_PR_FGD] = [180, 220, 180]  # probable FG: light green
        mask_vis[self.mask == GC_PR_BGD] = [80, 80, 100]    # probable BG: dark
        mask_vis[self.mask == GC_BGD]    = [20, 20, 30]     # definite BG: black

        # cutout on checkerboard
        h, w     = self.orig_bgr.shape[:2]
        board    = checkerboard(h, w)
        cutout   = board.copy()
        fg_3ch   = cv2.merge([fg_mask, fg_mask, fg_mask])
        cutout   = np.where(fg_3ch == 255, self.orig_bgr, board)

        # resize for display
        PW, PH = 310, 200
        mh, mw  = self._small_panel_size()

        mask_ph  = to_tk(cv2.resize(mask_vis,  (mw, mh), interpolation=cv2.INTER_AREA))
        cut_ph   = to_tk(cv2.resize(cutout,    (mw, mh), interpolation=cv2.INTER_AREA))

        self._refs.clear()
        self._refs += [mask_ph, cut_ph]

        img_lbl, stat_lbl = self._mask_card
        img_lbl.config(image=mask_ph); img_lbl.image = mask_ph
        fg_pct = int(np.sum(fg_mask > 0) / fg_mask.size * 100)
        stat_lbl.config(text=f"Foreground: {fg_pct}%  |  "
                             f"Green=FG  Light=probable FG  Dark=BG")

        img_lbl2, stat_lbl2 = self._cut_card
        img_lbl2.config(image=cut_ph); img_lbl2.image = cut_ph
        stat_lbl2.config(text="Checkerboard = transparent  |  Click 'Save PNG' to export")

        # redraw canvas with current mask overlay
        self._redraw_canvas()

    def _small_panel_size(self):
        """Sensible fixed size for the two bottom panels."""
        return 190, 295

    def _redraw_canvas(self):
        """Repaint the top canvas: original + mask overlay + rect."""
        if self.disp_bgr is None: return
        vis = self.disp_bgr.copy()

        # overlay mask if available
        if self.gc_done and self.mask is not None:
            h, w   = vis.shape[:2]
            fh, fw = self.mask.shape
            m_small = cv2.resize(
                np.where((self.mask == GC_FGD) | (self.mask == GC_PR_FGD),
                         255, 0).astype(np.uint8),
                (w, h), interpolation=cv2.INTER_NEAREST)
            # dim background
            bg_mask = (m_small == 0)
            vis[bg_mask] = (vis[bg_mask] * 0.35).astype(np.uint8)
            # teal border on FG edge
            edges = cv2.Canny(m_small, 50, 150)
            vis[edges > 0] = [78, 205, 196]

        ph = to_tk(vis)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=ph)
        self.canvas._ph = ph   # keep ref

        # draw rect if present
        if self.has_rect:
            rx, ry, rw, rh = self.rect
            self.canvas.create_rectangle(
                rx, ry, rx+rw, ry+rh,
                outline=ACCENT, width=2, dash=(6,3))

    def _show_original(self):
        """Fit original image into canvas and display."""
        MAX_CW, MAX_CH = WIN_W - 280, 350
        disp, scale    = fit(self.orig_bgr, MAX_CW, MAX_CH)
        self.disp_bgr  = disp
        self.disp_scale = scale
        h, w           = disp.shape[:2]
        self.canvas.config(width=w, height=h)
        self._redraw_canvas()

    # ── file ops ─────────────────────────────────────────────────────────────

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All", "*.*")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open that file.")
            return
        h, w = img.shape[:2]
        if max(h, w) > 1400:
            s = 1400 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
        self.orig_bgr = img
        self._reset()

    def _reset(self):
        self.mask = self.bgd_model = self.fgd_model = None
        self.rect = None
        self.has_rect = self.gc_done = False
        self._show_original()
        # clear output panels
        blank = np.full((190, 295, 3), 18, dtype=np.uint8)
        ph = to_tk(blank)
        for lbl, stat in [self._mask_card, self._cut_card]:
            lbl.config(image=ph); lbl.image = ph
            stat.config(text="")
        self._refs = [ph]
        self._status.set("Step 1: Draw a rectangle around your subject.")

    def _save(self):
        if not self.gc_done or self.mask is None:
            messagebox.showwarning("Nothing to save",
                                   "Run segmentation first by drawing a rectangle.")
            return
        fg_mask = np.where((self.mask == GC_FGD) | (self.mask == GC_PR_FGD),
                           255, 0).astype(np.uint8)
        # BGRA with alpha
        bgra        = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:,:,3] = fg_mask

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG with transparency", "*.png")])
        if path:
            cv2.imwrite(path, bgra)
            messagebox.showinfo("Saved", f"Transparent PNG saved:\n{path}")


if __name__ == "__main__":
    try:
        import cv2, numpy
        from PIL import Image
    except ImportError as e:
        print(f"\nMissing library: {e}")
        print("Install:  pip install opencv-python numpy Pillow\n")
        exit(1)

    BackgroundRemover().mainloop()
