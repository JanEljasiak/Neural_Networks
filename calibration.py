"""
Kalibrator pedzla - narzedzie do dobrania parametrow "miekkosci" pedzla
(core_radius, sigma) na oko, z podgladem obok prawdziwych przykladow MNIST.

Uruchom:
    python kalibrator_pedzla.py

Wymaga pliku mnist_samples.npz w tym samym folderze (kilka prawdziwych
obrazkow MNIST do porownania).

Jak uzyc:
1. Rysuj cyfry po lewej.
2. Przesuwaj suwaki CORE RADIUS i SIGMA - podglad 28x28 aktualizuje sie
   na biezaco.
3. Klikaj "Losuj przyklad MNIST", zeby zobaczyc prawdziwa cyfre o
   podobnym ksztalcie obok Twojego rysunku - porownaj "miekkosc" krawedzi.
4. Kiedy podglad Twojego rysunku bedzie wygladal podobnie do przykladu
   MNIST (podobny gradient/poswiata na krawedziach), przepisz wartosci
   z suwakow (pokazane liczbowo) do gui_rozpoznawanie_cyfr.py
   (stale BRUSH_CORE_RADIUS / BRUSH_SIGMA - pamietaj o przeskalowaniu,
   patrz komentarz przy suwakach).
"""

import os
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

CANVAS_SIZE = 320
MNIST_SIZE = 28
PREVIEW_SIZE = 224  # powiekszony podglad 28x28, dla wygody porownania

COL_BG = "#12141c"
COL_PANEL = "#1b1e2b"
COL_TEXT = "#e8e9f0"
COL_TEXT_MUTED = "#7d84a3"
COL_ACCENT = "#4fd1c5"
FONT = "Segoe UI"


def make_brush_kernel(core_radius, sigma, kernel_radius):
    yy, xx = np.mgrid[-kernel_radius:kernel_radius + 1, -kernel_radius:kernel_radius + 1]
    dist = np.sqrt(xx ** 2 + yy ** 2)
    kernel = np.where(
        dist <= core_radius,
        1.0,
        np.exp(-((dist - core_radius) ** 2) / (2 * sigma ** 2)),
    )
    return kernel.astype(np.float32)


class Calibrator:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalibrator pedzla - porownanie z prawdziwym MNIST")
        self.root.configure(bg=COL_BG)

        # --- dane MNIST do porownania ---
        samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_samples.npz")
        if os.path.exists(samples_path):
            data = np.load(samples_path)
            self.mnist_images = data["images"]
            self.mnist_labels = data["labels"]
        else:
            self.mnist_images = None
            self.mnist_labels = None
        self.current_sample_idx = 0

        self.paint_array = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last_x, self.last_y = None, None

        # --- parametry pedzla (suwaki dzialaja w skali 28x28, wygodniej ocenic) ---
        self.core_radius_28 = tk.DoubleVar(value=0.24)
        self.sigma_28 = tk.DoubleVar(value=1.17)

        self._build_ui()
        self._update_kernel()
        self._refresh_canvas()
        self._show_mnist_sample()

    # ------------------------------------------------------------------
    def _build_ui(self):
        main = tk.Frame(self.root, bg=COL_BG)
        main.pack(padx=16, pady=16)

        # --- lewa kolumna: rysowanie ---
        left = tk.Frame(main, bg=COL_PANEL)
        left.grid(row=0, column=0, padx=8, sticky="n")
        tk.Label(left, text="TWOJ RYSUNEK", font=(FONT, 10, "bold"),
                 bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(pady=(10, 4))
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg="#05050a", cursor="cross")
        self.canvas.pack(padx=10)
        self._canvas_item = self.canvas.create_image(0, 0, anchor="nw")
        self._canvas_photo = None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self._reset_stroke)

        tk.Button(left, text="Wyczysc", command=self.clear,
                  bg="#2a2e42", fg=COL_TEXT, relief="flat",
                  font=(FONT, 10), padx=10, pady=6).pack(pady=10)

        # --- srodkowa kolumna: podglad 28x28 Twojego rysunku ---
        mid = tk.Frame(main, bg=COL_PANEL)
        mid.grid(row=0, column=1, padx=8, sticky="n")
        tk.Label(mid, text="PODGLAD 28x28 (Twoj rysunek)", font=(FONT, 10, "bold"),
                 bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(pady=(10, 4))
        self.preview_label = tk.Label(mid, bg="#05050a")
        self.preview_label.pack(padx=10, pady=(0, 10))
        self._preview_photo = None

        # --- prawa kolumna: prawdziwy przyklad MNIST ---
        right = tk.Frame(main, bg=COL_PANEL)
        right.grid(row=0, column=2, padx=8, sticky="n")
        tk.Label(right, text="PRAWDZIWY PRZYKLAD MNIST", font=(FONT, 10, "bold"),
                 bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(pady=(10, 4))
        self.mnist_label = tk.Label(right, bg="#05050a")
        self.mnist_label.pack(padx=10, pady=(0, 6))
        self.mnist_caption = tk.Label(right, text="", font=(FONT, 10),
                                       bg=COL_PANEL, fg=COL_TEXT)
        self.mnist_caption.pack(pady=(0, 6))
        tk.Button(right, text="Losuj przyklad MNIST", command=self._show_mnist_sample,
                  bg="#2a2e42", fg=COL_TEXT, relief="flat",
                  font=(FONT, 10), padx=10, pady=6).pack(pady=(0, 10))

        # --- dolny panel: suwaki ---
        bottom = tk.Frame(self.root, bg=COL_BG)
        bottom.pack(padx=16, pady=(0, 16), fill="x")

        self._build_slider(bottom, "Core radius (rdzen kreski, skala 28x28)",
                            self.core_radius_28, 0.0, 2.0, 0.01)
        self._build_slider(bottom, "Sigma (miekkosc krawedzi, skala 28x28)",
                            self.sigma_28, 0.2, 3.0, 0.01)

        self.readout_label = tk.Label(bottom, text="", font=("Consolas", 10),
                                       bg=COL_BG, fg=COL_ACCENT, justify="left")
        self.readout_label.pack(anchor="w", pady=(8, 0))

    def _build_slider(self, parent, label, var, frm, to, resolution):
        row = tk.Frame(parent, bg=COL_BG)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, font=(FONT, 10), bg=COL_BG, fg=COL_TEXT,
                 width=38, anchor="w").pack(side=tk.LEFT)
        scale = tk.Scale(row, variable=var, from_=frm, to=to, resolution=resolution,
                          orient="horizontal", length=400, bg=COL_BG, fg=COL_TEXT,
                          troughcolor="#2a2e42", highlightthickness=0,
                          command=lambda e: self._on_param_change())
        scale.pack(side=tk.LEFT, padx=10)

    # ------------------------------------------------------------------
    def _on_param_change(self):
        self._update_kernel()
        self._update_preview()
        self._update_readout()

    def _update_kernel(self):
        scale = CANVAS_SIZE / MNIST_SIZE
        core_canvas = self.core_radius_28.get() * scale
        sigma_canvas = self.sigma_28.get() * scale
        kernel_radius = max(int(np.ceil(core_canvas + 3.2 * sigma_canvas)), 2)
        self.kernel = make_brush_kernel(core_canvas, sigma_canvas, kernel_radius)
        self.kernel_radius = kernel_radius

    def _update_readout(self):
        scale = CANVAS_SIZE / MNIST_SIZE
        core_c = self.core_radius_28.get() * scale
        sigma_c = self.sigma_28.get() * scale
        self.readout_label.config(text=(
            f"Wartosci do wklejenia w gui_rozpoznawanie_cyfr.py:\n"
            f"  core_radius_28 = {self.core_radius_28.get():.3f}   (canvas px: {core_c:.1f})\n"
            f"  sigma_28       = {self.sigma_28.get():.3f}   (canvas px: {sigma_c:.1f})\n"
            f"  promien stempla (canvas px): {self.kernel_radius}"
        ))

    # ------------------------------------------------------------------
    def _stamp_at(self, x, y):
        r = self.kernel_radius
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = ix - r, ix + r + 1
        y0, y1 = iy - r, iy + r + 1
        kx0, ky0 = 0, 0
        kx1, ky1 = self.kernel.shape[1], self.kernel.shape[0]
        if x0 < 0:
            kx0 = -x0; x0 = 0
        if y0 < 0:
            ky0 = -y0; y0 = 0
        if x1 > CANVAS_SIZE:
            kx1 -= (x1 - CANVAS_SIZE); x1 = CANVAS_SIZE
        if y1 > CANVAS_SIZE:
            ky1 -= (y1 - CANVAS_SIZE); y1 = CANVAS_SIZE
        if x0 >= x1 or y0 >= y1:
            return
        region = self.paint_array[y0:y1, x0:x1]
        kernel_region = self.kernel[ky0:ky1, kx0:kx1] * 255.0
        np.maximum(region, kernel_region, out=region)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            dist = np.hypot(x - self.last_x, y - self.last_y)
            step = max(self.kernel_radius / 5.0, 1.5)
            steps = max(1, int(dist / step))
            for i in range(steps + 1):
                t = i / steps
                self._stamp_at(self.last_x + (x - self.last_x) * t,
                                self.last_y + (y - self.last_y) * t)
        else:
            self._stamp_at(x, y)
        self.last_x, self.last_y = x, y
        self._refresh_canvas()
        self._update_preview()

    def _reset_stroke(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.paint_array[:] = 0
        self._refresh_canvas()
        self._update_preview()

    def _refresh_canvas(self):
        img = Image.fromarray(self.paint_array.astype(np.uint8), mode="L")
        if self._canvas_photo is None:
            self._canvas_photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self._canvas_item, image=self._canvas_photo)
        else:
            self._canvas_photo.paste(img)

    def _get_28x28(self):
        mask = self.paint_array > 8
        if not mask.any():
            return np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        digit = self.paint_array[y0:y1 + 1, x0:x1 + 1]
        h, w = digit.shape
        side = max(w, h)
        margin = int(side * 0.25)
        side += margin * 2
        square = np.zeros((side, side), dtype=np.float32)
        oy, ox = (side - h) // 2, (side - w) // 2
        square[oy:oy + h, ox:ox + w] = digit
        img = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8), mode="L")
        small = img.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)
        arr = np.asarray(small, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / arr.max() * 255.0
        return arr.astype(np.uint8)

    def _update_preview(self):
        arr = self._get_28x28()
        img = Image.fromarray(arr, mode="L").resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_photo)

    def _show_mnist_sample(self):
        if self.mnist_images is None:
            self.mnist_caption.config(text="Brak pliku mnist_samples.npz")
            return
        idx = np.random.randint(len(self.mnist_images))
        arr = self.mnist_images[idx]
        label = self.mnist_labels[idx]
        img = Image.fromarray(arr, mode="L").resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
        self._mnist_photo = ImageTk.PhotoImage(img)
        self.mnist_label.config(image=self._mnist_photo)
        self.mnist_caption.config(text=f"cyfra: {label}")


if __name__ == "__main__":
    root = tk.Tk()
    app = Calibrator(root)
    root.mainloop()