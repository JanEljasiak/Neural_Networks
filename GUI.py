"""
GUI do rysowania cyfr i testowania modelu (softmax regression / perceptron)
wytrenowanego w Twoim notatniku MNIST.

WAZNE: to jest zwykly skrypt Python, NIE komorka notatnika - uruchamiasz go
z terminala/konsoli:

    python gui_rozpoznawanie_cyfr.py

Wymaga zapisanego wczesniej modelu przez Twoja funkcje save_model()
(plik .npz), np. w folderze "models/".

Zaleznosci (jesli czegos brakuje):
    pip install pillow numpy

Sterowanie:
    F11     - przelacz tryb pelnoekranowy
    Escape  - wyjdz z trybu pelnoekranowego
"""

import glob
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Paleta kolorow - ciemny, spojny motyw
# ---------------------------------------------------------------------------

COL_BG = "#12141c"
COL_PANEL = "#1b1e2b"
COL_PANEL_BORDER = "#2a2e42"
COL_CANVAS = "#05050a"
COL_TEXT = "#e8e9f0"
COL_TEXT_MUTED = "#7d84a3"
COL_ACCENT = "#4fd1c5"
COL_ACCENT_HOT = "#ff7a59"
COL_BAR_BG = "#2a2e42"
COL_BUTTON = "#2a2e42"
COL_BUTTON_HOVER = "#374063"

FONT_FAMILY = "Segoe UI"


# ---------------------------------------------------------------------------
# Klasa modelu i funkcje pomocnicze - zgodne 1:1 z Twoim notatnikiem
# ---------------------------------------------------------------------------

def softmax(z):
    c = np.max(z)
    exp_list = np.array([np.exp(z_i - c) for z_i in z])
    return exp_list / sum(exp_list)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.biases = [np.random.randn(layer) for layer in layer_sizes[1:]]
        self.weights = [np.random.randn(layer_sizes[i + 1], layer_sizes[i])
                        for i in range(self.L)]

    def z(self, input):
        return [np.dot(self.weights[layer], input) + self.biases[layer]
                for layer in range(self.L)]

    def activation_output(self, input, layer, activation_function):
        return activation_function(self.z(input)[layer])


def load_model(filepath: str) -> "NeuralNetwork":
    data = np.load(filepath)
    layer_sizes = data["layer_sizes"].tolist()
    NN_loaded = NeuralNetwork(layer_sizes=layer_sizes)
    L = len(layer_sizes) - 1
    NN_loaded.weights = [data[f"weight_{i}"] for i in range(L)]
    NN_loaded.biases = [data[f"bias_{i}"] for i in range(L)]
    return NN_loaded


# ---------------------------------------------------------------------------
# MODEL PEDZLA - dopasowany do prawdziwych danych MNIST
#
# Zmierzylem (na 3000 obrazkach z prawdziwego zbioru MNIST) profil zaniku
# intensywnosci pikseli w poprzek pojedynczej kreski, wzgledem odleglosci
# od jej centrum, a nastepnie dopasowalem model:
#
#     profile(d) = 1.0                                   dla d <= core_radius
#     profile(d) = exp(-(d - core_radius)^2 / (2*sigma^2)) dla d > core_radius
#
# Wynik dopasowania (w skali obrazka 28x28):
#     core_radius = 0.24 px   (bardzo cienki, "twardy" rdzen kreski)
#     sigma       = 1.17 px   (gaussowski zanik na krawedziach)
#     mediana pelnej grubosci kreski (przy progu 0.3) = 4.0 px  <- zgadza sie
#       z policzonym niezaleznie promieniem modelu (2*(core+sigma*1.55)=4.1px)
#
# Ponizej skaluje te wartosci na piksele platna (CANVAS_SIZE), zeby pojedynczy
# "stempel" pedzla mial realistyczny, miekki profil - dokladnie tak jak
# prosiles: gradient wpisany w sam pedzel, a nie globalne rozmycie na koniec.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MODEL PEDZLA
#
# Pierwotne wartosci wyliczone z dopasowania do zmierzonego profilu MNIST
# (core=0.24, sigma=1.17) okazaly sie dawac zbyt mocne rozmycie w praktyce -
# uzytkownik dostroil parametry recznie w kalibrator_pedzla.py, porownujac
# na biezaco z prawdziwymi przykladami MNIST. Finalne, zweryfikowane wizualnie
# wartosci:
#     core_radius = 0.25 px  (skala 28x28)
#     sigma       = 0.25 px  (skala 28x28)
# ---------------------------------------------------------------------------

CANVAS_SIZE = 320
MNIST_SIZE = 28
_SCALE = CANVAS_SIZE / MNIST_SIZE  # zalozenie: rysowana cyfra wypelnia cale platno

BRUSH_CORE_RADIUS = 0.25 * _SCALE   # ~2.9 px na canvasie
BRUSH_SIGMA = 0.25 * _SCALE         # ~2.9 px na canvasie
BRUSH_KERNEL_RADIUS = int(np.ceil(BRUSH_CORE_RADIUS + 3.2 * BRUSH_SIGMA))  # obciecie ogona

PREVIEW_DISPLAY_SIZE = 140
COLLECTED_DATA_DIR = "collected_data"  # jawnie widoczny folder, zero ukrywania


def _make_brush_kernel(core_radius, sigma, kernel_radius):
    """Precomputed stempel pedzla jako macierz (2R+1)x(2R+1) wartosci 0..1."""
    size = 2 * kernel_radius + 1
    yy, xx = np.mgrid[-kernel_radius:kernel_radius + 1, -kernel_radius:kernel_radius + 1]
    dist = np.sqrt(xx ** 2 + yy ** 2)
    kernel = np.where(
        dist <= core_radius,
        1.0,
        np.exp(-((dist - core_radius) ** 2) / (2 * sigma ** 2)),
    )
    return kernel.astype(np.float32)


BRUSH_KERNEL = _make_brush_kernel(BRUSH_CORE_RADIUS, BRUSH_SIGMA, BRUSH_KERNEL_RADIUS)


# ---------------------------------------------------------------------------
# Przetwarzanie narysowanego obrazu na wejscie modelu (784 wartosci)
# ---------------------------------------------------------------------------

def preprocess_array(paint_array: np.ndarray):
    """
    paint_array: (CANVAS_SIZE, CANVAS_SIZE) float32, wartosci 0..255 -
    juz zawiera realistyczny, miekki profil krawedzi (wpisany w sam pedzel
    podczas rysowania), wiec NIE robimy tu juz zadnego dodatkowego rozmycia -
    tylko przyciecie do bounding boxa, centrowanie i przeskalowanie do 28x28.

    Zwraca (vector_784, small_28x28_uint8) - to drugie do podgladu w GUI.
    """
    threshold = 8.0  # ponizej tego uznajemy piksel za "tlo" przy liczeniu bboxa
    mask = paint_array > threshold
    if not mask.any():
        empty = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8)
        return np.zeros(MNIST_SIZE * MNIST_SIZE, dtype=np.float32), empty

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    digit = paint_array[y0:y1 + 1, x0:x1 + 1]

    h, w = digit.shape
    side = max(w, h)
    margin = int(side * 0.25)
    side += margin * 2

    square = np.zeros((side, side), dtype=np.float32)
    off_y = (side - h) // 2
    off_x = (side - w) // 2
    square[off_y:off_y + h, off_x:off_x + w] = digit

    square_img = Image.fromarray(np.clip(square, 0, 255).astype(np.uint8), mode="L")
    small = square_img.resize((MNIST_SIZE, MNIST_SIZE), Image.LANCZOS)

    arr = np.asarray(small, dtype=np.float32) / 255.0
    if arr.max() > 0:
        arr = arr / arr.max()

    display_arr = (arr * 255.0).astype(np.uint8)
    return arr.reshape(-1), display_arr


# ---------------------------------------------------------------------------
# Male pomocnicze widgety w spojnym stylu
# ---------------------------------------------------------------------------

def styled_button(parent, text, command):
    return tk.Button(
        parent, text=text, command=command,
        bg=COL_BUTTON, fg=COL_TEXT, activebackground=COL_BUTTON_HOVER,
        activeforeground=COL_TEXT, relief="flat", bd=0,
        font=(FONT_FAMILY, 10), padx=14, pady=8, cursor="hand2",
        highlightthickness=0,
    )


def panel_frame(parent):
    return tk.Frame(parent, bg=COL_PANEL, highlightbackground=COL_PANEL_BORDER,
                     highlightthickness=1, bd=0)


# ---------------------------------------------------------------------------
# Aplikacja GUI
# ---------------------------------------------------------------------------

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznawanie cyfr - MNIST perceptron")
        self.root.configure(bg=COL_BG)
        self.root.geometry("1280x820")
        self.root.minsize(980, 560)

        self.is_fullscreen = False
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)

        self.NN = None
        self.model_path = None
        self._redraw_job = None
        self._last_probs = None

        # bufor rysunku - float32, akumulowany operacja max (jak realny pedzel)
        self.paint_array = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=2)
        self.root.rowconfigure(1, weight=1)

        self._build_header()
        self._build_drawing_panel()
        self._build_result_panel()
        self._build_probability_panel()

        self.try_autoload_model()
        self._redraw_bars(None)
        self._update_preview(np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8))
        self._refresh_canvas_display()

    # ------------------------------------------------------------------
    # Budowa UI
    # ------------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self.root, bg=COL_BG)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=20, pady=(16, 8))
        tk.Label(header, text="Rozpoznawanie cyfr", font=(FONT_FAMILY, 20, "bold"),
                 bg=COL_BG, fg=COL_TEXT).pack(side=tk.LEFT)
        tk.Label(header, text="  ·  softmax regression  ·  pedzel wzorowany na profilu MNIST",
                 font=(FONT_FAMILY, 11), bg=COL_BG, fg=COL_TEXT_MUTED).pack(side=tk.LEFT, pady=(6, 0))

    def _build_drawing_panel(self):
        wrapper = panel_frame(self.root)
        wrapper.grid(row=1, column=0, sticky="ns", padx=(20, 10), pady=(0, 20))
        wrapper.rowconfigure(0, weight=1)

        # Canvas + scrollbar - gwarantuje ze cala zawartosc (wlacznie z sekcja
        # zapisu danych na samym dole) jest dostepna, niezaleznie od
        # rozdzielczosci ekranu / wysokosci okna
        scroll_canvas = tk.Canvas(wrapper, bg=COL_PANEL, highlightthickness=0,
                                   width=CANVAS_SIZE + 36)
        scrollbar = tk.Scrollbar(wrapper, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")

        inner = tk.Frame(scroll_canvas, bg=COL_PANEL)
        inner_window = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            scroll_canvas.itemconfig(inner_window, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows / macOS
        scroll_canvas.bind_all("<Button-4>", lambda e: scroll_canvas.yview_scroll(-1, "units"))  # Linux
        scroll_canvas.bind_all("<Button-5>", lambda e: scroll_canvas.yview_scroll(1, "units"))

        inner.configure(padx=0)
        inner_pad = tk.Frame(inner, bg=COL_PANEL)
        inner_pad.pack(padx=18, pady=18, fill="both")
        inner = inner_pad  # od tego miejsca dalej budujemy zawartosc jak wczesniej

        tk.Label(inner, text="RYSUJ TUTAJ", font=(FONT_FAMILY, 10, "bold"),
                 bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(anchor="w", pady=(0, 8))

        self.canvas = tk.Canvas(inner, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                 bg=COL_CANVAS, highlightthickness=2,
                                 highlightbackground=COL_ACCENT, cursor="cross")
        self.canvas.pack()
        self._canvas_image_item = self.canvas.create_image(0, 0, anchor="nw")
        self._canvas_photo = None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_stroke)
        self.last_x, self.last_y = None, None

        btn_row = tk.Frame(inner, bg=COL_PANEL)
        btn_row.pack(fill="x", pady=(14, 0))
        styled_button(btn_row, "Wyczysc", self.clear_canvas).pack(side=tk.LEFT)
        styled_button(btn_row, "Wczytaj model...", self.load_model_dialog).pack(side=tk.LEFT, padx=(8, 0))

        self.status_label = tk.Label(inner, text="Nie wczytano modelu",
                                      font=(FONT_FAMILY, 9), bg=COL_PANEL, fg="#e85d5d")
        self.status_label.pack(anchor="w", pady=(10, 0))

        preview_row = tk.Frame(inner, bg=COL_PANEL)
        preview_row.pack(fill="x", pady=(16, 0))
        tk.Label(preview_row, text="PODGLAD (28x28, wejscie modelu)",
                 font=(FONT_FAMILY, 9, "bold"), bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(anchor="w")
        self.preview_label = tk.Label(preview_row, bg=COL_CANVAS)
        self.preview_label.pack(anchor="w", pady=(4, 0))
        self._preview_photo = None

        # --- jawne, widoczne zbieranie danych treningowych ---
        collect_row = tk.Frame(inner, bg=COL_PANEL)
        collect_row.pack(fill="x", pady=(18, 0))

        tk.Label(collect_row, text="DODAJ TEN RYSUNEK DO ZBIORU DANYCH",
                 font=(FONT_FAMILY, 9, "bold"), bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(anchor="w")
        tk.Label(collect_row,
                 text="Rysunek zostanie zapisany lokalnie na tym komputerze\n"
                      "(folder collected_data/) wraz z wybrana ponizej etykieta.",
                 font=(FONT_FAMILY, 8), bg=COL_PANEL, fg=COL_TEXT_MUTED,
                 justify="left").pack(anchor="w", pady=(2, 6))

        self.selected_digit = tk.IntVar(value=0)
        digit_row = tk.Frame(collect_row, bg=COL_PANEL)
        digit_row.pack(anchor="w")
        self._digit_buttons = []
        for d in range(10):
            b = tk.Button(digit_row, text=str(d), width=2,
                          command=lambda d=d: self._select_digit(d),
                          bg=COL_BUTTON, fg=COL_TEXT, relief="flat", bd=0,
                          font=(FONT_FAMILY, 10), cursor="hand2")
            b.pack(side=tk.LEFT, padx=1)
            self._digit_buttons.append(b)

        self.save_data_button = styled_button(
            collect_row, "Zapisz jako cyfra 0", self._save_training_sample
        )
        self.save_data_button.pack(anchor="w", pady=(8, 0))

        self.collect_status_label = tk.Label(collect_row, text="Zebrano: 0 przykladow",
                                              font=(FONT_FAMILY, 8), bg=COL_PANEL, fg=COL_TEXT_MUTED)
        self.collect_status_label.pack(anchor="w", pady=(4, 0))
        self._refresh_collect_status()

        self._select_digit(0)  # musi byc PO utworzeniu save_data_button (odwoluje sie do niego)

    def _build_result_panel(self):
        wrapper = panel_frame(self.root)
        wrapper.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 20))
        inner = tk.Frame(wrapper, bg=COL_PANEL)
        inner.pack(expand=True, fill="both", padx=18, pady=18)

        tk.Label(inner, text="PREDYKCJA", font=(FONT_FAMILY, 10, "bold"),
                 bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(anchor="w")

        self.result_digit_label = tk.Label(inner, text="-", font=(FONT_FAMILY, 96, "bold"),
                                            bg=COL_PANEL, fg=COL_ACCENT_HOT)
        self.result_digit_label.pack(expand=True)

        self.result_conf_label = tk.Label(inner, text="Narysuj cyfre po lewej",
                                           font=(FONT_FAMILY, 13), bg=COL_PANEL, fg=COL_TEXT_MUTED)
        self.result_conf_label.pack(pady=(0, 6))

    def _build_probability_panel(self):
        wrapper = panel_frame(self.root)
        wrapper.grid(row=1, column=2, sticky="nsew", padx=(10, 20), pady=(0, 20))
        inner = tk.Frame(wrapper, bg=COL_PANEL)
        inner.pack(expand=True, fill="both", padx=18, pady=18)

        tk.Label(inner, text="ROZKLAD PRAWDOPODOBIENSTWA (softmax)",
                 font=(FONT_FAMILY, 10, "bold"), bg=COL_PANEL, fg=COL_TEXT_MUTED).pack(anchor="w", pady=(0, 6))

        self.prob_canvas = tk.Canvas(inner, bg=COL_PANEL, highlightthickness=0)
        self.prob_canvas.pack(expand=True, fill="both")
        self.prob_canvas.bind("<Configure>", lambda e: self._redraw_bars(self._last_probs))

    # ------------------------------------------------------------------
    # Rysowanie - stempel pedzla nakladany operacja max wzdluz ruchu myszy
    # ------------------------------------------------------------------
    def _stamp_at(self, x, y):
        r = BRUSH_KERNEL_RADIUS
        ix, iy = int(round(x)), int(round(y))

        x0, x1 = ix - r, ix + r + 1
        y0, y1 = iy - r, iy + r + 1

        kx0, ky0 = 0, 0
        kx1, ky1 = BRUSH_KERNEL.shape[1], BRUSH_KERNEL.shape[0]

        if x0 < 0:
            kx0 = -x0
            x0 = 0
        if y0 < 0:
            ky0 = -y0
            y0 = 0
        if x1 > CANVAS_SIZE:
            kx1 -= (x1 - CANVAS_SIZE)
            x1 = CANVAS_SIZE
        if y1 > CANVAS_SIZE:
            ky1 -= (y1 - CANVAS_SIZE)
            y1 = CANVAS_SIZE

        if x0 >= x1 or y0 >= y1:
            return

        region = self.paint_array[y0:y1, x0:x1]
        kernel_region = BRUSH_KERNEL[ky0:ky1, kx0:kx1] * 255.0
        np.maximum(region, kernel_region, out=region)

    def _stamp_line(self, x0, y0, x1, y1):
        dist = np.hypot(x1 - x0, y1 - y0)
        step = max(BRUSH_KERNEL_RADIUS / 5.0, 1.5)
        steps = max(1, int(dist / step))
        for i in range(steps + 1):
            t = i / steps
            self._stamp_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self._stamp_line(self.last_x, self.last_y, x, y)
        else:
            self._stamp_at(x, y)
        self.last_x, self.last_y = x, y

        self._refresh_canvas_display()
        self.predict_live()

    def reset_stroke(self, event):
        self.last_x, self.last_y = None, None
        self.predict_live(force_redraw=True)

    def clear_canvas(self):
        self.paint_array[:] = 0
        self._refresh_canvas_display()
        self.result_digit_label.config(text="-")
        self.result_conf_label.config(text="Narysuj cyfre po lewej")
        self._redraw_bars(None)
        self._update_preview(np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.uint8))

    def _refresh_canvas_display(self):
        img = Image.fromarray(self.paint_array.astype(np.uint8), mode="L")
        if self._canvas_photo is None:
            self._canvas_photo = ImageTk.PhotoImage(img)
            self.canvas.itemconfig(self._canvas_image_item, image=self._canvas_photo)
        else:
            self._canvas_photo.paste(img)

    # ------------------------------------------------------------------
    # Jawne zbieranie danych treningowych - widoczne dla osoby rysujacej,
    # etykieta wybierana przez nia recznie, nic nie dzieje sie w tle
    # ------------------------------------------------------------------
    def _select_digit(self, digit):
        self.selected_digit.set(digit)
        for i, b in enumerate(self._digit_buttons):
            b.config(bg=COL_ACCENT if i == digit else COL_BUTTON,
                     fg="#05050a" if i == digit else COL_TEXT)
        self.save_data_button.config(text=f"Zapisz jako cyfra {digit}")

    def _count_collected_samples(self):
        dataset_path = os.path.join(COLLECTED_DATA_DIR, "collected_dataset.npz")
        if not os.path.exists(dataset_path):
            return 0
        data = np.load(dataset_path)
        return len(data["labels"])

    def _refresh_collect_status(self):
        n = self._count_collected_samples()
        self.collect_status_label.config(
            text=f"Zebrano dotychczas: {n} przykladow (folder: {COLLECTED_DATA_DIR}/)"
        )

    def _save_training_sample(self):
        vector, preview_arr = preprocess_array(self.paint_array)
        if not vector.any():
            messagebox.showinfo("Pusty rysunek", "Najpierw narysuj cyfre.")
            return

        digit = self.selected_digit.get()
        os.makedirs(COLLECTED_DATA_DIR, exist_ok=True)

        # (1) czytelny podglad PNG - zeby latwo mozna bylo przejrzec/wyczyscic zbior recznie
        digit_dir = os.path.join(COLLECTED_DATA_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        png_path = os.path.join(digit_dir, f"{timestamp}.png")
        Image.fromarray(preview_arr, mode="L").save(png_path)

        # (2) skonsolidowany zbior .npz (wektor 784 + etykieta), gotowy do wczytania w notatniku
        dataset_path = os.path.join(COLLECTED_DATA_DIR, "collected_dataset.npz")
        if os.path.exists(dataset_path):
            existing = np.load(dataset_path)
            images = np.vstack([existing["images"], vector.reshape(1, -1)])
            labels = np.append(existing["labels"], digit)
        else:
            images = vector.reshape(1, -1)
            labels = np.array([digit])
        np.savez(dataset_path, images=images, labels=labels)

        self._refresh_collect_status()
        messagebox.showinfo("Zapisano",
                            f"Rysunek zapisany jako cyfra {digit}.\n"
                            f"Plik: {png_path}\n"
                            f"Lacznie w zbiorze: {len(labels)} przykladow.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def try_autoload_model(self):
        candidates = sorted(glob.glob("models/*.npz"))
        if candidates:
            self._load_model(candidates[-1])

    def load_model_dialog(self):
        filepath = filedialog.askopenfilename(
            title="Wybierz plik modelu (.npz)",
            filetypes=[("Model NPZ", "*.npz")]
        )
        if filepath:
            self._load_model(filepath)

    def _load_model(self, filepath):
        try:
            self.NN = load_model(filepath)
            self.model_path = filepath
            short_name = filepath.replace("\\", "/").split("/")[-1]
            self.status_label.config(text=f"Model: {short_name}", fg=COL_ACCENT)
        except Exception as e:
            messagebox.showerror("Blad wczytywania modelu", str(e))
            self.status_label.config(text="Blad wczytywania modelu", fg="#e85d5d")

    # ------------------------------------------------------------------
    # Predykcja na biezaco
    # ------------------------------------------------------------------
    def predict_live(self, force_redraw=False):
        if self.NN is None:
            return

        x, preview_arr = preprocess_array(self.paint_array)
        z = self.NN.z(x)[0]
        probs = softmax(z)
        self._last_probs = probs

        self._update_preview(preview_arr)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        self.result_digit_label.config(text=str(pred_class))
        self.result_conf_label.config(text=f"pewnosc: {confidence * 100:.1f}%")

        if force_redraw:
            if self._redraw_job is not None:
                self.root.after_cancel(self._redraw_job)
                self._redraw_job = None
            self._redraw_bars(probs)
        else:
            if self._redraw_job is not None:
                self.root.after_cancel(self._redraw_job)
            self._redraw_job = self.root.after(
                40, lambda: self._redraw_bars(probs)
            )

    def _update_preview(self, arr_28x28_uint8):
        img = Image.fromarray(arr_28x28_uint8, mode="L")
        img = img.resize((PREVIEW_DISPLAY_SIZE, PREVIEW_DISPLAY_SIZE), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_photo)

    # ------------------------------------------------------------------
    # Wlasny, rysowany recznie panel poziomych paskow (bez matplotlib)
    # ------------------------------------------------------------------
    def _redraw_bars(self, probs):
        self._redraw_job = None
        c = self.prob_canvas
        c.delete("all")

        width = c.winfo_width()
        height = c.winfo_height()
        if width < 10 or height < 10:
            return

        n = 10
        row_h = height / n
        label_w = 34
        value_w = 74
        pad_x = 8
        bar_x0 = label_w + pad_x
        bar_x1 = width - value_w - pad_x
        max_bar_w = max(bar_x1 - bar_x0, 1)

        pred_class = int(np.argmax(probs)) if probs is not None else None

        for digit in range(n):
            y0 = digit * row_h
            y1 = y0 + row_h
            y_mid = (y0 + y1) / 2
            bar_h = max(row_h * 0.5, 6)
            bar_y0 = y_mid - bar_h / 2
            bar_y1 = y_mid + bar_h / 2

            is_pred = probs is not None and digit == pred_class
            digit_color = COL_ACCENT_HOT if is_pred else COL_TEXT_MUTED

            c.create_text(label_w / 2, y_mid, text=str(digit),
                          fill=digit_color, font=(FONT_FAMILY, 12, "bold" if is_pred else "normal"))

            c.create_rectangle(bar_x0, bar_y0, bar_x0 + max_bar_w, bar_y1,
                               fill=COL_BAR_BG, outline="")

            prob = float(probs[digit]) if probs is not None else 0.0
            fill_w = max_bar_w * prob
            bar_color = COL_ACCENT_HOT if is_pred else COL_ACCENT

            if fill_w > 0:
                c.create_rectangle(bar_x0, bar_y0, bar_x0 + fill_w, bar_y1,
                                   fill=bar_color, outline="")

            value_text = f"{prob * 100:6.2f}%" if probs is not None else "  -   "
            c.create_text(bar_x1 + pad_x + value_w / 2, y_mid, text=value_text,
                          fill=COL_TEXT if is_pred else COL_TEXT_MUTED,
                          font=(FONT_FAMILY, 11, "bold" if is_pred else "normal"))

    # ------------------------------------------------------------------
    # Pelny ekran
    # ------------------------------------------------------------------
    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def exit_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()