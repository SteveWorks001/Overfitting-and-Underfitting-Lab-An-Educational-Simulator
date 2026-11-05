"""Overfitting & Underfitting Lab — учебный симулятор переобучения.

Десятый проект серии образовательных симуляторов:
    1. One Artificial Neuron   2. Perceptron           3. Mini Neural Network
    4. Gradient Descent        5. Backpropagation      6. Neural Network Training Lab
    7. Activation Functions    8. Loss Functions Lab   9. Classification Lab
   10. Overfitting & Underfitting Lab

Приложение показывает одну цепочку:
    сложность модели -> ошибка на обучении -> ошибка на тесте -> обобщение.

Модель: полиномиальная регрессия
    y = b0 + b1*x + b2*x^2 + ... + bn*x^n
Коэффициенты подбираются по обучающим точкам методом наименьших квадратов.
Тестовые точки в подборе не участвуют: по ним только измеряется ошибка.

Все кривые, числа и выводы считаются по фактической подгонке. Никаких
заранее нарисованных кривых и никаких заготовленных вердиктов.

Только стандартная библиотека: Python 3, tkinter, tkinter.ttk, math, random.
Внешние пакеты (NumPy, matplotlib и другие) не используются.

Автор: SteveWorks001

Куда смотреть
-------------
Раздел 1   Области ввода, палитра, обозначения
Раздел 2   Учебные наборы данных
Раздел 3   Модель: разбиение, подгонка, MSE, вывод
Раздел 4   Форматирование чисел и разбор ввода
Раздел 5   Локализация: весь текст живёт в TEXTS (русский по умолчанию)
Раздел 6   Учебные эксперименты
Раздел 7   Справка: заголовки и тексты разделов
Раздел 8   Виджеты: параметры, основной график, кривая сложности
Раздел 9   Главное окно
Раздел 10  Точка входа
"""

import math
import random
import tkinter as tk
from tkinter import ttk, font as tkfont

# ----------------------------------------------------------------------
# Раздел 1. Области ввода, палитра, обозначения
# ----------------------------------------------------------------------

X_MIN, X_MAX = -3.0, 3.0                     # область признака
N_POINTS = 26                                # точек в каждом наборе
X_SCALE = 3.0                                # подгонка идёт в t = x / X_SCALE

DEG_MIN, DEG_MAX = 1, 9                      # степени: от линейной до девятой
SIGMA_MIN, SIGMA_MAX = 0.0, 0.9              # уровень шума
SIGMA_STEP = 0.05
SHARE_MIN, SHARE_MAX = 0.35, 0.90            # доля обучающих точек
SHARE_STEP = 0.05

DEFAULT_SHAPE = "wave"
DEFAULT_DEGREE = 3
DEFAULT_SIGMA = 0.35
DEFAULT_SHARE = 0.70

PAD_LEFT, PAD_TOP, PAD_RIGHT, PAD_BOTTOM = 58, 18, 18, 44
PAD2_LEFT, PAD2_TOP, PAD2_RIGHT, PAD2_BOTTOM = 62, 34, 16, 44
MIN_WIDTH = 1288                             # минимальная ширина окна
CX_PLOT_HEIGHT = 330                         # высота графика «ошибка против сложности»
CURVE_SAMPLES = 240                          # плотность основной кривой
TEXT_WIDTH = 38                              # ширина панелей текста, символы
TABLE_KEYS = ("deg", "train", "test", "verdict")
TABLE_WIDTHS = (54, 92, 92, 132)

# Пороги вывода. Они не всеобщие и не научные константы: это правила,
# по которым приложение читает конкретный эксперимент. Пороги показаны
# в интерфейсе и объяснены в справке.
TH_CLOSE = 2.0                               # тест не хуже удвоенного минимума
TH_GAP = 2.0                                 # тест хотя бы вдвое хуже обучения
TH_FLOOR = 0.01                              # 1% от ошибки «бесполезной» модели

# Учебная отметка о размере теста, а не граница пригодности расчёта: при
# m <= 5 одна точка весит не меньше 20% тестовой MSE. К guard'у подгонки
# (число ОБУЧАЮЩИХ точек) отношения не имеет.
SMALL_TEST_WARN = 5

# Обучающие точки — синий, тестовые — оранжевый, подобранная кривая —
# тёмно-синяя. Тот же язык, что и в предыдущих проектах серии: цвет
# метрики повторяет цвет точек, к которым метрика относится.
COL_TRAIN = "#2f6f9f"
COL_TEST = "#c8651b"
COL_FIT = "#1b3a55"
COL_TRUTH = "#8a8a82"
COL_FILL_TRAIN = "#e8f1f7"
COL_FILL_TEST = "#fbeee0"
COL_BEST = "#b07d10"
COL_CURRENT = "#5b3a86"
COL_GRID = "#dfe3e8"
COL_AXIS = "#4a4a4a"
COL_TEXT = "#242424"
COL_BG = "#fbfbf7"
COL_PAPER = "#fdfdf9"
COL_FRAME = "#c8c8c0"
COL_DIM = "#8a8a82"
COL_HEAD = "#31566f"
COL_KEY = "#b07d10"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"

SUBS = "₀₁₂₃₄₅₆₇₈₉"
SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def sub(number):
    """x -> x с цифрой в нижнем индексе (для b₀, b₁ и так далее)."""
    return "".join(SUBS[int(c)] if c.isdigit() else c for c in str(number))


def sup(number):
    """x -> x с цифрой в верхнем индексе (для x², x⁹ и так далее)."""
    return "".join(SUPS[int(c)] if c.isdigit() else c for c in str(number))


# ----------------------------------------------------------------------
# Раздел 2. Учебные наборы данных
# ----------------------------------------------------------------------
# Набор строится по одному и тому же правилу: абсциссы стоят ровно и
# равномерно, а отклонение берётся из фиксированного ряда случайных чисел,
# который только масштабируется ползунком шума. Поэтому:
#   * при одних и тех же настройках набор одинаков при любом запуске;
#   * ползунок шума не пересобирает точки заново, а плавно увеличивает уже
#     существующие отклонения — шум видно как рост разброса, а не как
#     новый случайный рисунок.
# «Истинная зависимость» (функция shape.fn) известна: это та самая
# закономерность, которую модель пытается восстановить. В подгонке она не
# участвует и нарисована только для того, чтобы глаз видел ошибку формы.

SHAPES = (
    {
        "key": "wave",
        "seed": 7001,
        "n": N_POINTS,
        "fn": lambda x: 1.7 * math.sin(x),
        "formula": "y = 1.7·sin(x)",
    },
    {
        "key": "cubic",
        "seed": 7002,
        "n": N_POINTS,
        "fn": lambda x: 0.18 * x ** 3 - 0.7 * x,
        "formula": "y = 0.18·x³ − 0.7·x",
    },
    {
        "key": "ripple",
        "seed": 7003,
        "n": N_POINTS,
        "fn": lambda x: 2.4 * math.exp(-0.12 * x * x) * math.cos(1.7 * x),
        "formula": "y = 2.4·e^(−0.12·x²)·cos(1.7·x)",
    },
)

SHAPE_BY_KEY = {entry["key"]: entry for entry in SHAPES}


def deviates(shape_key):
    """Фиксированный ряд стандартных случайных отклонений для набора."""
    entry = SHAPE_BY_KEY[shape_key]
    rng = random.Random(entry["seed"])
    return tuple(rng.gauss(0.0, 1.0) for _ in range(entry["n"]))


def build_dataset(shape_key, sigma):
    """Точки (x, y) и чистые значения y без шума — детерминированно."""
    entry = SHAPE_BY_KEY[shape_key]
    noise = deviates(shape_key)
    n = entry["n"]
    xs, ys, clean = [], [], []
    for i in range(n):
        x = X_MIN + (X_MAX - X_MIN) * i / (n - 1)
        base = entry["fn"](x)
        xs.append(x)
        clean.append(base)
        ys.append(base + sigma * noise[i])
    return tuple(zip(xs, ys)), tuple(zip(xs, clean))


# ----------------------------------------------------------------------
# Раздел 3. Модель: разбиение, подгонка, MSE, вывод
# ----------------------------------------------------------------------

def split_points(points, share):
    """Разбиение на обучение и тест.

    Тестовые точки берутся равномерно по всему диапазону (правило
    «каждая k-я»), а не подряд: иначе тест лежал бы только на краю
    области, и вывод о переобучении зависел бы от края графика. Точки,
    попавшие в тест, из подгонки исключены полностью.
    """
    n = len(points)
    k_test = int(round(n * (1.0 - share)))
    k_test = max(1, min(k_test, n - 2))
    train, test = [], []
    for i, p in enumerate(points):
        if math.floor((i + 1) * k_test / n) > math.floor(i * k_test / n):
            test.append((i, p))
        else:
            train.append((i, p))
    return train, test


def horner(coeffs, x):
    """Значение многочлена по коэффициентам при степенях x."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _dot(a, b):
    return sum(u * v for u, v in zip(a, b))


def fit_tcoef(xs, ys, degree):
    """Метод наименьших квадратов в ортогональном базисе.

    Возвращает коэффициенты при степенях t = x / X_SCALE или None, если
    базис вырождается. Ортогонализация (модифицированный процесс
    Грэма-Шмидта с двойным проходом) вместо нормальных уравнений: для
    степеней до девятой нормальные уравнения теряют точность, а здесь
    каждый коэффициент находится отдельным скалярным делением и остаётся
    устойчивым.
    """
    n = len(xs)
    if n == 0 or n < degree + 2:
        return None
    ts = [x / X_SCALE for x in xs]
    cols, dens, betas, powers = [], [], [], []
    for k in range(degree + 1):
        vals = [t ** k for t in ts]
        pv = [0.0] * (degree + 1)
        pv[k] = 1.0
        norm0 = math.sqrt(_dot(vals, vals))
        if not math.isfinite(norm0) or norm0 <= 0.0:
            return None
        for _ in range(2):
            for j in range(k):
                c = _dot(vals, cols[j]) / dens[j]
                vals = [v - c * cv for v, cv in zip(vals, cols[j])]
                pv = [p - c * q for p, q in zip(pv, powers[j])]
        d = _dot(vals, vals)
        if not math.isfinite(d) or d <= 1e-12 * norm0 * norm0:
            return None
        cols.append(vals)
        powers.append(pv)
        dens.append(d)
        betas.append(_dot(vals, ys) / d)
    out = [0.0] * (degree + 1)
    for k, beta in enumerate(betas):
        for i, p in enumerate(powers[k]):
            out[i] += beta * p
    if not all(math.isfinite(c) for c in out):
        return None
    return out


def t_to_x_coeffs(tcoef):
    """Переход от базиса t к базису x.

    Подгонка идёт по степеням t = x / X_SCALE, поэтому член при x^i равен
    коэффициенту при t^i, делённому на X_SCALE в степени i.
    """
    return [c / (X_SCALE ** i) for i, c in enumerate(tcoef)]


def x_to_t_coeffs(coefs):
    """Обратный переход — нужен для проверки независимым расчётом."""
    return [c * (X_SCALE ** i) for i, c in enumerate(coefs)]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def mse(points, coeffs):
    """Средняя квадратичная ошибка по списку точек (x, y)."""
    if not points:
        return None
    s = 0.0
    for x, y in points:
        r = y - horner(coeffs, x)
        s += r * r
    return s / len(points)


def sse_pair(coeffs, train_pts, test_pts):
    """Суммы квадратов отклонений: они стоят за обеими метриками MSE."""
    def total(points):
        s = 0.0
        for x, y in points:
            r = y - horner(coeffs, x)
            s += r * r
        return s
    return total(train_pts), total(test_pts)


def no_skill_mse(train_pts, test_pts):
    """Ошибка константной модели: среднее обучающих точек на тесте.

    Это естественная шкала «модель ничего не предсказывает». С ней
    сравнивается абсолютная величина ошибки, чтобы не объявывать
    переобучением маленькую ошибку на гладких данных.
    """
    if not train_pts or not test_pts:
        return None
    base = mean([y for x, y in train_pts])
    s = 0.0
    for x, y in test_pts:
        s += (y - base) ** 2
    return s / len(test_pts)


def fit_degree(train_pts, test_pts, degree):
    """Подгонка по обучающим точкам: коэффициенты и обе ошибки.

    Строка всегда возвращается: ok=False с указанием причины. Так интерфейс
    может объяснить отказ вместо молчаливого рисования сомнительной кривой.
    """
    xs = [x for x, y in train_pts]
    ys = [y for x, y in train_pts]
    if len(xs) < degree + 2:
        return {"degree": degree, "ok": False, "reason": "few",
                "coeffs": None, "train_mse": None, "test_mse": None}
    tcoef = fit_tcoef(xs, ys, degree)
    if tcoef is None:
        return {"degree": degree, "ok": False, "reason": "unstable",
                "coeffs": None, "train_mse": None, "test_mse": None}
    coeffs = t_to_x_coeffs(tcoef)
    return {
        "degree": degree,
        "ok": True,
        "reason": None,
        "coeffs": coeffs,
        "train_mse": mse(train_pts, coeffs),
        "test_mse": mse(test_pts, coeffs),
    }


def sweep_complexity(train_pts, test_pts, max_degree=DEG_MAX):
    """Подгонка для каждой степени: из неё строится кривая сложности."""
    return [fit_degree(train_pts, test_pts, degree)
            for degree in range(DEG_MIN, max_degree + 1)]


def guard_min_points(degree):
    """Число точек guard'а подгонки, переписанное словами для подписей.

    Само условие отказа живёт в fit_degree и не меняется: эта функция нужна
    только чтобы текст называл то же число, что и проверка. Внешняя проверка
    сверяет их на всей сетке степеней и долей.
    """
    return degree + 2


def analyse(points, share, degree, max_degree=DEG_MAX):
    """Полный разбор текущей настройки.

    Возвращает разбиение, подгонку выбранной степени, перебор степеней,
    лучшую степень по тесту и текстовый вывод. Всё посчитано один раз: оттуда
    же берутся и график, и метрики, и кривая сложности.
    """
    train, test = split_points(points, share)
    train_pts = [p for i, p in train]
    test_pts = [p for i, p in test]
    base = no_skill_mse(train_pts, test_pts)
    rows = sweep_complexity(train_pts, test_pts, max_degree)
    fitted = [r for r in rows if r["ok"]]
    best_test = min((r["test_mse"] for r in fitted), default=None)
    best_train = min((r["train_mse"] for r in fitted), default=None)
    best_degree = None
    for r in fitted:
        if r["test_mse"] == best_test:
            best_degree = r["degree"]
            break
    current = rows[degree - DEG_MIN]
    tol = None
    tol_close_term = None
    tol_floor_term = None
    if best_test is not None:
        tol_close_term = TH_CLOSE * best_test
        tol_floor_term = TH_FLOOR * (base or 0.0)
        tol = max(tol_close_term, tol_floor_term)
    verdict = interpret(degree, current, best_test, best_degree, base, tol)
    return {
        "train": train,
        "test": test,
        "train_pts": train_pts,
        "test_pts": test_pts,
        "rows": rows,
        "fitted_count": len(fitted),
        "best_test": best_test,
        "best_train": best_train,
        "best_degree": best_degree,
        "base": base,
        "tol": tol,
        "tol_close_term": tol_close_term,
        "tol_floor_term": tol_floor_term,
        "current": current,
        "verdict": verdict,
    }


def interpret(degree, current, best_test, best_degree, base, tol):
    """Вывод по фактическим ошибкам, а не по значению степени.

    Правило (оно же открыто показывается в интерфейсе):
      1. тестовая ошибка не хуже порога tol -> удачное соответствие;
      2. иначе, если тест заметно хуже обучения и степень не левее лучшей
         по тесту -> переобучение;
      3. иначе, если степень левее лучшей и разрыва нет -> недообучение;
      4. иначе признаки смотрят в разные стороны -> переходный случай.
    """
    if not current["ok"]:
        return {"key": "nofit", "degree": degree, "reason": current["reason"]}
    train_mse = current["train_mse"]
    test_mse = current["test_mse"]
    if tol is None:
        return {"key": "nofit", "degree": degree, "reason": "unstable"}
    gap = test_mse >= TH_GAP * max(train_mse, 1e-15)
    left = best_degree is not None and degree < best_degree
    if test_mse <= tol:
        key = "good"
    elif gap and not left:
        key = "over"
    elif left and not gap:
        key = "under"
    else:
        key = "trans"
    return {
        "key": key,
        "degree": degree,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "gap": gap,
        "left": left,
        "tol": tol,
        "reason": None,
    }


def curve_outside(coeffs, y_low, y_high, x_lo, x_hi, samples=64):
    """Выходит ли кривая за видимую область — честная отметка на графике."""
    for i in range(samples + 1):
        x = x_lo + (x_hi - x_lo) * i / samples
        y = horner(coeffs, x)
        if not math.isfinite(y) or y < y_low or y > y_high:
            return True
    return False


# ----------------------------------------------------------------------
# Раздел 4. Форматирование чисел и разбор ввода
# ----------------------------------------------------------------------

def clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def fmt(value, digits=2):
    """Обычное число: без экспоненты и без лишних знаков."""
    if value is None or not math.isfinite(value):
        return "—"
    text = "%.*f" % (digits, value)
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def fmt_error(value):
    """Ошибка MSE: на маленьком масштабе переходим к значащим цифрам."""
    if value is None or not math.isfinite(value):
        return "—"
    a = abs(value)
    if a >= 1000.0:
        return "%.0f" % value
    if a >= 1.0:
        return "%.3f" % value
    if a >= 0.001:
        return "%.4f" % value
    if a == 0.0:
        return "0"
    return "%.1e" % value


def fmt_coef(value):
    """Коэффициент многочлена: четыре значащих цифры, без хвостовых нулей."""
    if value is None or not math.isfinite(value):
        return "—"
    if value == 0.0:
        return "0"
    a = abs(value)
    if a >= 1e5 or a < 1e-3:
        return "%.3e" % value
    text = "%.4g" % value
    return text


def fmt_share(value):
    return "%d%%" % int(round(value * 100.0))


def parse_number(text):
    """Разбор поля ввода: запятая и точка оба принимаются."""
    raw = text.strip().replace(",", ".").replace(" ", "")
    if raw in ("", "-", "+"):
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_int(text):
    value = parse_number(text)
    if value is None:
        return None
    return int(round(value))


def polynomial_text(coeffs):
    """Собирает читаемый вид многочлена: пропущены нулевые слагаемые."""
    if not coeffs:
        return "—"
    parts = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-10:
            continue
        sign = "−" if c < 0 else "+"
        mag = abs(c)
        if i == 0:
            term = fmt_coef(mag)
        else:
            tail = "x" if i == 1 else "x" + sup(i)
            term = "" if abs(mag - 1.0) < 1e-10 else "%s·%s" % (fmt_coef(mag), tail)
            if not term:
                term = tail
        if not parts:
            parts.append(("-" if c < 0 else "") + term)
        else:
            parts.append(" %s %s" % (sign, term))
    if not parts:
        return "0"
    return "ŷ = " + "".join(parts)


# ----------------------------------------------------------------------
# Раздел 5. Локализация
# ----------------------------------------------------------------------

LANG = "ru"


def set_language(code):
    global LANG
    LANG = code


def get_language():
    return LANG


# Весь переводимый текст живёт здесь. Формулы и числа язык не меняют: они
# подписываются ключами ниже, а сами выражения нейтральны.

TEXTS = {
    "ru": {
        "app_title": "Overfitting & Underfitting Lab — где модели хватает",
        "subtitle": ("Сложность модели → ошибка на обучении → ошибка на тесте "
                     "→ обобщение. Полином подбирается по обучающим точкам; "
                     "тестовые служат только для измерения ошибки."),
        # Левая колонка
        "box_dataset": "Учебный набор",
        "ds_wave_name": "Плавная волна (синус)",
        "ds_wave_desc": "Полный виток синуса: прямая не описывает его вовсе, "
                        "кубика уже близка, а высокая степень начинает "
                        "колебаться между точками.",
        "ds_cubic_name": "Кубическая S-образная",
        "ds_cubic_desc": "Форма — ровно многочлен третьей степени: без шума "
                         "степень 3 восстанавливает её точно, а степень выше "
                         "лишняя.",
        "ds_ripple_name": "Затухающая рябь",
        "ds_ripple_desc": "Колебание, сжатое гауссовой оболочкой: требует "
                          "высокой степени, и именно на краях переобучение "
                          "видно сильнее всего.",
        "ds_formula": "Истинная зависимость: {f}",
        "ds_note": "Точки лежат на этой кривой с добавлением шума. Набор один и "
                   "тот же при каждом запуске.",
        "box_setup": "Данные и сложность модели",
        "lbl_degree": "Степень полинома",
        "lbl_sigma": "Уровень шума",
        "lbl_share": "Доля обучающих точек",
        "hint_degree": "Степень 1 — прямая, 2 — парабола, 3 — кубическая. "
                       "Коэффициентов всегда на один больше, чем степень.",
        "hint_step": "Слайдер и кнопки меняют значение сразу: пересчёт идёт на "
                     "каждом движении. Запятая и точка принимаются оба.",
        "btn_reset": "Сбросить всё",
        "box_split": "Разбиение на обучение и тест",
        "split_train": "Обучающих точек: {n}",
        "split_test": "Тестовых точек: {n}",
        "split_note": "Разбиение равномерное: каждая k-я точка уходит в тест, "
                      "поэтому тест лежит по всему диапазону, а не только на "
                      "краю. Эти точки модель не видит.",
        "box_legend": "Обозначения на графике",
        "legend_train": "Синие кружки — обучающие точки",
        "legend_test": "Оранжевые квадраты — тестовые точки",
        "legend_fit": "Тёмно-синяя линия — подобранная модель",
        "legend_truth": "Серый пунктир — истинная зависимость",
        "box_watch": "Что сейчас видно",
        "box_language": "Язык интерфейса",
        "lang_ru": "Русский",
        "lang_en": "English",
        # Вкладки
        "tab_model": "Модель",
        "tab_complexity": "Сложность и ошибка",
        "tab_exp": "Опыты",
        "tab_help": "Справка",
        # Вкладка «Модель»
        "mb_box_state": "Текущая модель",
        "mb_line_degree": "Сложность: степень {d} ({k} коэффициентов)",
        "mb_line_split": "Обучение: {n} точек · Тест: {m} точек",
        "mb_box_math": "Модель и её коэффициенты",
        "mb_general": "ŷ = b₀ + b₁x + b₂x² + … + bₙxⁿ",
        "mb_here": "Здесь n = {d}, коэффициентов {k}: b₀ … b{ds}",
        "mb_fitted": "Подобранное уравнение:",
        "mb_coeff_head": "Коэффициенты по степеням x:",
        "mb_learn_note": "Коэффициенты получены минимизацией суммы квадратов "
                         "отклонений по обучающим точкам. Тестовые точки в этом "
                         "расчёте не участвуют.",
        "mb_box_errors": "Ошибка обучения и ошибка теста",
        "mb_mse_formula": "MSE = (1 / N) · Σ (yᵢ − ŷᵢ)²",
        "mb_train_mse": "MSE обучения: {v}",
        "mb_test_mse": "MSE теста: {v}",
        "mb_sum_train": "Обучение: {s} / {n} = {v}",
        "mb_sum_test": "Тест: {s} / {m} = {v}",
        "mb_ratio": "Отношение теста к обучению: {r}",
        "mb_box_bars": "Сравнение ошибок",
        "bars_train": "обучение",
        "bars_test": "тест",
        "bars_note": "Полоски в одной шкале: чем длиннее, тем больше ошибка.",
        "mb_box_verdict": "Вывод по фактическим числам",
        "verdict_under": "Недообучение",
        "verdict_good": "Удачное соответствие",
        "verdict_over": "Переобучение",
        "verdict_trans": "Переходный случай",
        "verdict_nofit": "Подгонка невозможна",
        "expl_under": ("Ошибка велика и на обучении ({tm}), и на тесте ({em}): "
                       "полином степени {d} слишком прост, чтобы повторить "
                       "форму данных. Модель недосмотрела закономерность, а не "
                       "выучила шум."),
        "expl_good": ("Тестовая ошибка {em} не больше порога {tol}. Порог — "
                      "большее из двух чисел: {c1} · лучшей тестовой ошибки "
                      "{bt} и {c2} · ошибки бесполезной модели {sk}. {dt} "
                      "Модель повторяет закономерность и не гонится за "
                      "отдельными точками."),
        "expl_good_tol_bt": "В этом опыте порог задаёт первое число — лучшая "
                            "тестовая ошибка.",
        "expl_good_tol_sk": "В этом опыте порог задаёт второе число — один "
                            "процент ошибки бесполезной модели.",
        "expl_over": ("На обучении ошибка {tm}, а на тесте {em} — заметно "
                      "больше. Свободы у модели (степень {d}) больше, чем "
                      "нужно: часть коэффициентов работает на шум."),
        "expl_trans": ("Признаки смотрят в разные стороны: разрыв между тестом "
                       "({em}) и обучением ({tm}) уже есть, но степень {d} ещё "
                       "не перешла лучшую по тесту степень {bd}. Так выглядит "
                       "область, где удачное соответствие переходит в "
                       "переобучение."),
        "expl_nofit_few": ("Полином степени {d} требует {k} коэффициентов, а "
                           "обучающих точек всего {n}. Нужно не меньше {need}: "
                           "точек должно быть хотя бы на одну больше, чем "
                           "неизвестных. Пока это не так, подгонка не "
                           "определена, поэтому приложение её не выполняет."),
        "expl_nofit_unstable": ("При степени {d} вычисление потеряло "
                                "устойчивость: очередной член базиса оказался "
                                "слишком близок к уже учтённым. Приложение не "
                                "показывает подозрительные коэффициенты — "
                                "уменьшите степень или увеличьте долю "
                                "обучающих точек."),
        "mb_box_rule": "Как приложение получает вывод",
        "rule_intro": "Правило одно и то же для любого набора; числа ниже — из "
                      "текущего опыта.",
        "rule_1": "1. Лучшая тестовая ошибка среди степеней {lo}…{hi}: {bt}.",
        "rule_2": "2. Порог «всё ещё хорошо» = max({c1} · лучшей ошибки, "
                  "{c2} · ошибки бесполезной модели) = {tol}.",
        "rule_3": "3. Тест ≤ порога → удачное соответствие.",
        "rule_4": "4. Иначе тест ≥ {c3} · обучения и степень правее лучшей → "
                  "переобучение.",
        "rule_5": "5. Иначе степень левее лучшей и разрыва нет → "
                  "недообучение.",
        "rule_6": "6. Иначе — переходный случай.",
        "rule_base": "Ошибкой бесполезной модели считается MSE константы "
                     "«среднее по обучению» на тесте: {v}.",
        "rule_best": "Лучшая тестовая ошибка — у степени {d}; она отмечена "
                     "золотым кружком на вкладке «Сложность и ошибка».",
        "hint_share": "Доля {p}: {n} точек модель видит, {m} точек она не "
                      "видит и по ним только проверяет.",
        "split_coef": "Коэффициентов в модели: {k} при степени {d}. Точек "
                      "обучения {n}: нужно не меньше {need} — хотя бы на одну "
                      "больше, чем коэффициентов.",
        "split_small": "Тестовых точек всего {m}: одна неудачная точка весит в "
                       "этой ошибке {w}%. Отметка учебная — это не граница "
                       "пригодности расчёта, а напоминание, насколько оценка "
                       "колеблется на таком малом тесте.",
        "watch_line": "Степень {d}: обучение {tm}, тест {em}. Вывод: {v}.",
        "mb_box_bias": "Интуиция: смещение и разброс",
        "bias_under": "Модель слишком стеснена — большое смещение: она не "
                      "может описать форму даже в принципе.",
        "bias_over": "Модель слишком чувствительна к удачной выборке — большой "
                     "разброс: стоило бы точке сдвинуться, и кривая ушла бы за "
                     "ней.",
        "mb_curve_out": "Кривая уходит за пределы окна: на видимом диапазоне "
                        "признака значения доходят до {v}. Это тоже следствие "
                        "избыточной свободы.",
        "mb_test_lower": "Тестовая ошибка оказалась не больше обучающей — при "
                         "маленьком тесте это нормально и переобучением не "
                         "считается.",
        # Вкладка «Сложность и ошибка»
        "cx_box_chart": "Ошибка против сложности",
        "cx_axis_note": ("Вертикаль в сжатой шкале √MSE, подписи делений — "
                         "настоящие значения MSE. Кривые посчитаны подгоном "
                         "каждой степени на этом же разбиении."),
        "cx_marks_note": ("Отметки на графике: фиолетовый пунктир — текущая "
                          "степень, золотое кольцо — лучшая по тесту ошибка, "
                          "серые засечки внизу — степени, для которых не "
                          "хватило обучающих точек."),
        "cx_legend_train": "MSE обучения",
        "cx_legend_test": "MSE теста",
        "cx_best": "Лучшая тестовая ошибка — степень {d}.",
        "cx_current": "Текущая степень — {d}, она отмечена на графике.",
        "cx_lowest_train": "Самая малая ошибка обучения — степень {d}: "
                           "наименьшая ошибка на обучении ещё не означает "
                           "лучшую модель.",
        "cx_box_table": "Таблица по всем степеням",
        "cx_note": "Строки «подгонки нет» — степени, которым не хватает "
                   "обучающих точек.",
        "cx_rough": "Кривая теста не обязана быть гладкой: на {n} точках одна "
                    "неудачная точка двигает всю линию.",
        "cx_reuse_note": "Лучшую степень приложение выбирает по тестовой "
                         "ошибке, поэтому тест здесь работает не только как "
                         "одна финальная независимая проверка. Выбирая степень "
                         "по тесту много раз, вы подстраиваете её под этот же "
                         "тест, и показанная тестовая ошибка становится "
                         "оптимистичнее.",
        # Вкладка «Опыты»
        "exp_box_list": "Готовые опыты",
        "btn_run_exp": "Применить выбранный опыт",
        "exp_box_goal": "Зачем этот опыт",
        "exp_none": "Выберите опыт из списка — он переставит набор, степень, "
                    "шум и разбиение, а числа пересчитаются по-настоящему.",
        "exp_hint": "Опыт меняет только настройки. Никакие выводы и кривые не "
                    "подставлены: всё считается заново.",
        # Вкладка «Справка»
        "help_box": "Разделы справки",
        "help_none": "Выберите раздел слева.",
        # Столбцы таблицы
        "deg": "Степень",
        "train": "MSE обуч.",
        "test": "MSE тест",
        "verdict": "Вывод",
        "vd_under": "недообучение",
        "vd_good": "хорошо",
        "vd_over": "переобучение",
        "vd_trans": "переход",
        "vd_nofit": "нет подгонки",
        # Строка состояния
        "status_start": "Набор «{name}», степень {d}, шум {s}, обучение {p}. "
                        "Обучающих точек {n}, тестовых {m}.",
        "status_params": "Степень {d}, шум {s}, обучение {p}: MSE обучения "
                         "{tm}, MSE теста {em}.",
        "status_invalid": "В поле «{f}» не число. Прежнее значение сохранено.",
        "status_clamped": "Значение {raw} подправлено до {value}: диапазон "
                          "{lo}…{hi}.",
        "status_int": "Степень полинома — целое число: {raw} превращено в "
                      "{value}.",
        "status_lang": "Интерфейс переключён на {name}. Настройка, данные и "
                       "результат не изменились.",
        "status_reset": "Сброшено к исходной настройке: набор «{name}», "
                        "степень {d}.",
        "status_exp": "Опыт «{name}» применён: {tm} против {em}.",
        "status_nofit_few": "Степень {d}: коэффициенты не определены — "
                            "обучающих точек {n}, а нужно не меньше {need}.",
        "status_nofit_unstable": "Степень {d}: подгонка неустойчива на этих "
                                 "данных, расчёт остановлен.",
        "status_no_exp": "Сначала выберите опыт в списке.",
        "bars_nofit": "Подгонки нет: сравнивать ошибки не по чему.",
        "cx_nofit": "Ни одна степень не подогнана: обучающих точек слишком "
                    "мало.",
        "cx_axis_x": "степень полинома",
        "cx_axis_y": "MSE, шкала √",
        "axis_x": "признак x",
        "axis_y": "значение y",
        "cx_mark_current": "текущая степень",
        "cx_mark_best": "лучшая по тесту",
        "vd_reason_few": "мало точек",
        "vd_reason_unstable": "неустойчиво",
    },
    "en": {
        "app_title": "Overfitting & Underfitting Lab — when a model is enough",
        "subtitle": ("Model complexity → training error → test error → "
                     "generalization. The polynomial is fitted on training "
                     "points; test points are only measured against."),
        "box_dataset": "Dataset",
        "ds_wave_name": "Smooth wave (sine)",
        "ds_wave_desc": "One full turn of a sine: a line cannot describe it at "
                        "all, a cubic is already close, and a high degree "
                        "starts to wiggle between the points.",
        "ds_cubic_name": "Cubic S-shape",
        "ds_cubic_desc": "The shape is exactly a third-degree polynomial: "
                         "without noise degree 3 restores it precisely, and "
                         "any higher degree is surplus.",
        "ds_ripple_name": "Damped ripple",
        "ds_ripple_desc": "An oscillation squeezed by a Gaussian envelope: it "
                          "needs a high degree, and overfitting shows earliest "
                          "near the edges.",
        "ds_formula": "Underlying pattern: {f}",
        "ds_note": "Points sit on this curve plus noise. The dataset is "
                   "identical at every launch.",
        "box_setup": "Data and model complexity",
        "lbl_degree": "Polynomial degree",
        "lbl_sigma": "Noise level",
        "lbl_share": "Training share",
        "hint_degree": "Degree 1 is a line, 2 a parabola, 3 a cubic. There is "
                       "always one more coefficient than the degree.",
        "hint_step": "The slider and the buttons recalculate immediately: every "
                     "movement redraws. Both comma and dot are accepted.",
        "btn_reset": "Reset everything",
        "box_split": "Train and test split",
        "split_train": "Training points: {n}",
        "split_test": "Test points: {n}",
        "split_note": "The split is even: every k-th point goes to the test, so "
                      "the test spans the whole range instead of only the edge. "
                      "The model never sees those points.",
        "box_legend": "Graph legend",
        "legend_train": "Blue circles — training points",
        "legend_test": "Orange squares — test points",
        "legend_fit": "Dark blue line — the fitted model",
        "legend_truth": "Grey dashes — the underlying pattern",
        "box_watch": "What is on screen",
        "box_language": "Interface language",
        "lang_ru": "Русский",
        "lang_en": "English",
        "tab_model": "Model",
        "tab_complexity": "Complexity",
        "tab_exp": "Experiments",
        "tab_help": "Help",
        "mb_box_state": "Current model",
        "mb_line_degree": "Complexity: degree {d} ({k} coefficients)",
        "mb_line_split": "Training: {n} points · Test: {m} points",
        "mb_box_math": "Model and its coefficients",
        "mb_general": "ŷ = b₀ + b₁x + b₂x² + … + bₙxⁿ",
        "mb_here": "Here n = {d}, so {k} coefficients: b₀ … b{ds}",
        "mb_fitted": "Fitted equation:",
        "mb_coeff_head": "Coefficients by powers of x:",
        "mb_learn_note": "The coefficients minimize the sum of squared "
                         "residuals over the training points only. Test points "
                         "take no part in that calculation.",
        "mb_box_errors": "Training error and test error",
        "mb_mse_formula": "MSE = (1 / N) · Σ (yᵢ − ŷᵢ)²",
        "mb_train_mse": "Training MSE: {v}",
        "mb_test_mse": "Test MSE: {v}",
        "mb_sum_train": "Training: {s} / {n} = {v}",
        "mb_sum_test": "Test: {s} / {m} = {v}",
        "mb_ratio": "Test over training: {r}",
        "mb_box_bars": "Error comparison",
        "bars_train": "training",
        "bars_test": "test",
        "bars_note": "Both bars share one scale: longer means larger error.",
        "mb_box_verdict": "Verdict from the actual numbers",
        "verdict_under": "Underfitting",
        "verdict_good": "Good fit",
        "verdict_over": "Overfitting",
        "verdict_trans": "Transition case",
        "verdict_nofit": "Fitting impossible",
        "expl_under": ("Error is large both on training ({tm}) and on test "
                       "({em}): a degree-{d} polynomial is too simple to "
                       "repeat the shape of the data. The model missed the "
                       "pattern instead of learning the noise."),
        "expl_good": ("The test error {em} stays within the threshold {tol}. "
                      "That threshold is the larger of two numbers: {c1} · the "
                      "best test error {bt}, and {c2} · the no-skill error "
                      "{sk}. {dt} The model follows the pattern without chasing "
                      "individual points."),
        "expl_good_tol_bt": "In this experiment the threshold is set by the "
                            "first number, the best test error.",
        "expl_good_tol_sk": "In this experiment the threshold is set by the "
                            "second number, the one-percent floor taken from "
                            "the no-skill error.",
        "expl_over": ("Training error is {tm} while test error is {em} — "
                      "clearly larger. Degree {d} gives the model more freedom "
                      "than the data need: some coefficients serve the noise."),
        "expl_trans": ("The signs point different ways: a gap between test "
                       "({em}) and training ({tm}) is already there, yet degree "
                       "{d} has not passed the best test degree {bd}. This is "
                       "where a good fit turns into overfitting."),
        "expl_nofit_few": ("A degree-{d} polynomial needs {k} coefficients, "
                           "but there are only {n} training points. At least "
                           "{need} are required: there must be more points "
                           "than unknowns, by at least one. Until that holds "
                           "the fit is undetermined, so the application does "
                           "not perform it."),
        "expl_nofit_unstable": ("At degree {d} the calculation lost stability: "
                                "a new basis term came out almost parallel to "
                                "the ones already used. The application does "
                                "not print doubtful coefficients — lower the "
                                "degree or raise the training share."),
        "mb_box_rule": "How the verdict is produced",
        "rule_intro": "One rule for every dataset; the numbers below come from "
                      "the current experiment.",
        "rule_1": "1. Best test error among degrees {lo}…{hi}: {bt}.",
        "rule_2": "2. Threshold for still good = max({c1} · best error, "
                  "{c2} · no-skill error) = {tol}.",
        "rule_3": "3. Test ≤ threshold → good fit.",
        "rule_4": "4. Otherwise test ≥ {c3} · training and degree right of the "
                  "best → overfitting.",
        "rule_5": "5. Otherwise degree left of the best and no gap → "
                  "underfitting.",
        "rule_6": "6. Otherwise — transition case.",
        "rule_base": "The no-skill error is the test MSE of the constant "
                     "«mean of training»: {v}.",
        "rule_best": "The best test error belongs to degree {d}; it is marked "
                     "with a gold circle on the Complexity tab.",
        "hint_share": "Share {p}: the model sees {n} points and never sees the "
                      "other {m}, which are used only for checking.",
        "split_coef": "The model has {k} coefficients at degree {d}, and {n} "
                      "training points — at least {need} are required, one "
                      "more than the number of coefficients.",
        "split_small": "Only {m} test points: one unlucky point carries {w}% of "
                       "this error. The marker is educational, not a validity "
                       "boundary — it only says how much the estimate wobbles on "
                       "so small a test.",
        "watch_line": "Degree {d}: training {tm}, test {em}. Verdict: {v}.",
        "mb_box_bias": "Intuition: bias and variance",
        "bias_under": "The model is too constrained — high bias: it cannot "
                      "describe the shape even in principle.",
        "bias_over": "The model is too sensitive to this particular sample — "
                     "high variance: were one point to move, the curve would "
                     "move with it.",
        "mb_curve_out": "The curve leaves the window: over the visible range of "
                        "the feature it reaches {v}. That is another sign of "
                        "too much freedom.",
        "mb_test_lower": "The test error turned out no larger than the training "
                         "error — with a small test that is normal and is not "
                         "called overfitting.",
        "cx_box_chart": "Error against complexity",
        "cx_axis_note": ("The vertical axis is the compressed √MSE scale, tick "
                         "labels show real MSE values. Both curves come from "
                         "fitting every degree on this same split."),
        "cx_marks_note": ("Marks on the graph: the violet dashed line is the "
                          "current degree, the gold ring is the best test "
                          "error, and the grey ticks along the bottom are "
                          "degrees that ran out of training points."),
        "cx_legend_train": "Training MSE",
        "cx_legend_test": "Test MSE",
        "cx_best": "Best test error — degree {d}.",
        "cx_current": "Current degree — {d}, marked on the graph.",
        "cx_lowest_train": "Smallest training error — degree {d}: the lowest "
                           "training error is not yet the best model.",
        "cx_box_table": "Table for all degrees",
        "cx_note": "Rows marked «no fit» are degrees with too few training "
                   "points.",
        "cx_rough": "The test curve need not be smooth: with {n} points one "
                    "unlucky point moves the whole line.",
        "cx_reuse_note": "The application picks the best degree by the test "
                         "error, so the test here is not used only as one final "
                         "independent check. Choosing the degree over the test "
                         "again and again tunes the choice to that same test, "
                         "and the test error on screen turns optimistic.",
        "exp_box_list": "Prepared experiments",
        "btn_run_exp": "Run the selected experiment",
        "exp_box_goal": "Why this experiment",
        "exp_none": "Pick an experiment from the list — it will set the "
                    "dataset, degree, noise and split, and every number is "
                    "recomputed for real.",
        "exp_hint": "An experiment only moves the settings. No verdict and no "
                    "curve is substituted: everything is calculated again.",
        "help_box": "Help sections",
        "help_none": "Select a section on the left.",
        "deg": "Degree",
        "train": "Train MSE",
        "test": "Test MSE",
        "verdict": "Verdict",
        "vd_under": "underfitting",
        "vd_good": "good fit",
        "vd_over": "overfitting",
        "vd_trans": "transition",
        "vd_nofit": "no fit",
        "status_start": "Dataset “{name}”, degree {d}, noise {s}, training "
                        "{p}. Training points {n}, test points {m}.",
        "status_params": "Degree {d}, noise {s}, training {p}: training MSE "
                         "{tm}, test MSE {em}.",
        "status_invalid": "The “{f}” field is not a number. Previous value kept.",
        "status_clamped": "{raw} adjusted to {value}: the range is {lo}…{hi}.",
        "status_int": "A polynomial degree is a whole number: {raw} became "
                      "{value}.",
        "status_lang": "Interface switched to {name}. Settings, data and "
                       "results are unchanged.",
        "status_reset": "Reset to the starting configuration: dataset “{name}”, "
                        "degree {d}.",
        "status_exp": "Experiment “{name}” applied: {tm} against {em}.",
        "status_nofit_few": "Degree {d}: coefficients are undetermined — {n} "
                            "training points where at least {need} are "
                            "required.",
        "status_nofit_unstable": "Degree {d}: the fit is unstable on this "
                                 "data, the calculation stopped.",
        "status_no_exp": "Select an experiment from the list first.",
        "bars_nofit": "There is no fit: the two errors have nothing to compare.",
        "cx_nofit": "No degree could be fitted: too few training points.",
        "cx_axis_x": "polynomial degree",
        "cx_axis_y": "MSE, √ scale",
        "axis_x": "feature x",
        "axis_y": "value y",
        "cx_mark_current": "current degree",
        "cx_mark_best": "best on test",
        "vd_reason_few": "too few points",
        "vd_reason_unstable": "unstable",
    },
}


def T(key, **kwargs):
    """Строка текущего языка; пропущенный ключ виден как [key]."""
    table = TEXTS[get_language()]
    if key not in table:
        table = TEXTS["ru"]
    if key not in table:
        return "[" + key + "]"
    text = table[key]
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return text
    return text


# ----------------------------------------------------------------------
# Раздел 6. Учебные эксперименты
# ----------------------------------------------------------------------
# Опыт — это состояние приложения: набор, шум, разбиение и степень. Числа
# подобраны так, чтобы каждый сценарий учил чему-то одному, но сам вывод
# приложение читает по рассчитанным ошибкам, а не по имени опыта.

EXPERIMENTS = (
    {"key": "under_wave", "shape": "wave", "sigma": 0.35, "share": 0.70,
     "degree": 1},
    {"key": "good_wave", "shape": "wave", "sigma": 0.35, "share": 0.70,
     "degree": 4},
    {"key": "over_wave", "shape": "wave", "sigma": 0.35, "share": 0.70,
     "degree": 9},
    {"key": "cubic_exact", "shape": "cubic", "sigma": 0.35, "share": 0.70,
     "degree": 3},
    {"key": "cubic_under", "shape": "cubic", "sigma": 0.35, "share": 0.70,
     "degree": 1},
    {"key": "ripple_high", "shape": "ripple", "sigma": 0.35, "share": 0.70,
     "degree": 9},
    {"key": "ripple_low", "shape": "ripple", "sigma": 0.35, "share": 0.70,
     "degree": 2},
    {"key": "noisy_over", "shape": "wave", "sigma": 0.80, "share": 0.70,
     "degree": 9},
    {"key": "clean_good", "shape": "wave", "sigma": 0.0, "share": 0.70,
     "degree": 5},
    {"key": "few_train", "shape": "cubic", "sigma": 0.40, "share": 0.35,
     "degree": 7},
    {"key": "no_fit", "shape": "cubic", "sigma": 0.35, "share": 0.35,
     "degree": 9},
)

EXPERIMENT_BY_KEY = {entry["key"]: entry for entry in EXPERIMENTS}

EXP_TEXTS = {
    "ru": {
        "under_wave": (
            "Прямая против синуса: недообучение",
            "Степень 1, шум 0.35, обучение 70%. Ошибка велика с обеих сторон: "
            "0.6675 на обучении и 0.5997 на тесте — отношение около единицы. "
            "Прямой линии не хватает, чтобы повторить виток синуса, и тест "
            "здесь не мстит: он просто показывает ту же нехватку ёмкости.",
        ),
        "good_wave": (
            "Степень 4: ошибка теста почти лучшая",
            "Тот же набор, степень 4. Обучение 0.108, тест 0.038 — тестовая "
            "ошибка оказалась меньше обучающей, и это нормально: тестовых "
            "точек всего восемь, и они удачно легли на кривую. Лучшая тестовая "
            "ошибка всего диапазона принадлежит этой же степени.",
        ),
        "over_wave": (
            "Степень 9: обучение идеально, тест плохой",
            "Шум тот же. Обучение падает до 0.069, а тест растёт до 0.757 — "
            "в одиннадцать раз хуже. Кривая начинает колебаться между точками, "
            "и именно эти колебания дают большие ошибки на тестовых абсциссах.",
        ),
        "cubic_exact": (
            "Форма данных — кубика: степень 3 попадает",
            "Набор построен как 0.18·x³ − 0.7·x, поэтому степени 3 хватает: "
            "0.078 на обучении и 0.172 на тесте, обе ошибки держатся около "
            "уровня шума. Дальнейший рост степени улучшения не приносит.",
        ),
        "cubic_under": (
            "Даже знакомая форма слишком сложна для прямой",
            "Та же кубическая форма, степень 1: 0.584 и 1.001. S-образный "
            "изгиб прямо не вытягивается, и тест оказывается хуже обучения — "
            "не потому что модель запомнила, а потому что форма не влезла.",
        ),
        "ripple_high": (
            "Высокая степень иногда нужна: затухающая рябь",
            "Набор с быстрыми колебаниями, степень 9: обучение 0.120, тест "
            "0.279 — и это лучшая тестовая ошибка всего диапазона. Припомните: "
            "большая степень сама по себе не порок, порок — избыток свободы "
            "относительно формы данных.",
        ),
        "ripple_low": (
            "Степень 2 на ряби: обе ошибки велики",
            "Тот же набор при степени 2: обучение 1.490, тест 1.936. Ни одна "
            "сторона не улучшается — типичная картина недосмотра закономерности.",
        ),
        "noisy_over": (
            "Много шума — переобучение дороже",
            "Шум 0.80, степень 9: 0.359 против 3.956. Чем сильнее разброс, тем "
            "заметнее свободная модель тратит коэффициенты на шум, и тем "
            "сильнее расхождение двух ошибок.",
        ),
        "clean_good": (
            "Шум выключен: кривая почти совпадает с истиной",
            "Шум 0.00, степень 5: 0.00003 против 0.00015. Без шума высокая "
            "свобода почти безвредна — переобучению нужно что-то запоминать, а "
            "запоминать нечего. Двигайте ползунок шума и смотрите, как "
            "появляется разрыв.",
        ),
        "few_train": (
            "Мало обучающих точек — кривая улетает",
            "Обучение 35% (9 точек), степень 7: обучение 0.016 против теста "
            "2.245. За пределами окна кривая уходит далеко: сравните синий "
            "кружок и оранжевый квадрат там, где точек мало.",
        ),
        "no_fit": (
            "Степень 9 на девяти точках: подгонки нет",
            "Коэффициентов нужно десять, а точек девять: решение не "
            "единственное. Приложение честно отказывается считать и объясняет "
            "причину — вместо того чтобы рисовать случайную кривую.",
        ),
    },
    "en": {
        "under_wave": (
            "A line against a sine: underfitting",
            "Degree 1, noise 0.35, 70% training. The error is large on both "
            "sides: 0.6675 on training and 0.5997 on test, a ratio near one. A "
            "straight line simply cannot repeat a full turn of the sine, and "
            "the test is not punishing the model — it reports the same "
            "shortcoming.",
        ),
        "good_wave": (
            "Degree 4: the test error is close to the best",
            "Same dataset, degree 4. Training 0.108, test 0.038 — the test "
            "error came out smaller, which is normal: there are only eight "
            "test points and they happened to sit on the curve. The best test "
            "error of the whole range belongs to this same degree.",
        ),
        "over_wave": (
            "Degree 9: perfect training, poor test",
            "Same noise. Training drops to 0.069 while the test rises to 0.757 "
            "— eleven times worse. The curve starts to wiggle between points, "
            "and those wiggles produce the large errors at the test abscissas.",
        ),
        "cubic_exact": (
            "The shape is a cubic: degree 3 hits it",
            "The dataset is 0.18·x³ − 0.7·x, so degree 3 is enough: 0.078 on "
            "training and 0.172 on test, both sitting near the noise level. "
            "Raising the degree further brings no improvement.",
        ),
        "cubic_under": (
            "Even a familiar shape is too much for a line",
            "Same cubic shape, degree 1: 0.584 and 1.001. An S-bend cannot be "
            "stretched into a straight line, and the test comes out worse — not "
            "because the model memorised, but because the shape did not fit.",
        ),
        "ripple_high": (
            "Sometimes a high degree is what you need",
            "Fast oscillations, degree 9: training 0.120, test 0.279 — the best "
            "test error in the whole range. A large degree is not a defect by "
            "itself; surplus freedom relative to the shape is.",
        ),
        "ripple_low": (
            "Degree 2 on the ripple: both errors are large",
            "Same dataset at degree 2: training 1.490, test 1.936. Neither side "
            "improves — the classic picture of a missed pattern.",
        ),
        "noisy_over": (
            "More noise makes overfitting expensive",
            "Noise 0.80, degree 9: 0.359 against 3.956. The wider the scatter, "
            "the more coefficients a flexible model spends on noise, and the "
            "wider the gap between the two errors.",
        ),
        "clean_good": (
            "Noise switched off: the curve nearly is the pattern",
            "Noise 0.00, degree 5: 0.00003 against 0.00015. Without noise extra "
            "freedom is almost harmless — overfitting needs something to "
            "memorise. Move the noise slider and watch the gap appear.",
        ),
        "few_train": (
            "Few training points — the curve flies away",
            "35% training (9 points), degree 7: 0.016 against a test error of "
            "2.245. Outside the window the curve runs far away: compare a blue "
            "circle with an orange square where points are scarce.",
        ),
        "no_fit": (
            "Degree 9 on nine points: no fit at all",
            "Ten coefficients against nine points leaves the solution "
            "undetermined. The application refuses to compute and explains why, "
            "instead of drawing a random curve.",
        ),
    },
}


# ----------------------------------------------------------------------
# Раздел 7. Справка
# ----------------------------------------------------------------------
# Формат раздела: (заголовок, строки). Строка, начинающаяся с "»", — ключевая
# формула; с "!" — важная оговорка. Остальные строки — обычный текст.

HELP_KEYS = (
    "idea", "polynomial", "fitting", "train", "test", "mse", "complexity",
    "under", "good", "over", "general", "biasvar", "verdict", "howto",
    "limits",
)

HELP = {
    "ru": {
        "idea": (
            "1. Что показывает приложение",
            (
                "Один и тот же набор точек, одна и та же модель — многочлен, и "
                "только сложность модели меняется. На экране видно, как ошибка "
                "на обучении падает, а ошибка на тесте сначала падает, а потом "
                "растёт.",
                "Цепочка: сложность → ошибка обучения → ошибка теста → "
                "обобщение.",
                "! Здесь нет обучения в смысле градиентного спуска: коэффициенты "
                "находятся одним решением метода наименьших квадратов, а не "
                "итерациями.",
            ),
        ),
        "polynomial": (
            "2. Модель: многочлен от одной переменной",
            (
                "Модель предсказывает значение по одной формуле:",
                "»ŷ = b₀ + b₁x + b₂x² + … + bₙxⁿ",
                "Степень n задаёт число коэффициентов: их всегда n + 1. "
                "Степень 1 — прямая, 2 — парабола, 3 — кубическая кривая. "
                "Каждый следующий коэффициент добавляет кривой ещё один "
                "возможный изгиб.",
                "Коэффициенты приложение показывает в явном виде на вкладке "
                "«Модель»: предсказание можно пересчитать вручную по той же "
                "формуле.",
            ),
        ),
        "fitting": (
            "3. Как подбираются коэффициенты",
            (
                "Перебираются не коэффициенты, а ищется минимум суммы "
                "квадратов отклонений по обучающим точкам:",
                "»S = Σ (yᵢ − ŷᵢ)²  — только по точкам обучения",
                "Это классическая задача наименьших квадратов: при нехватке "
                "точек её решение не единственно. Приложение требует, чтобы "
                "обучающих точек было хотя бы на одну больше, чем "
                "коэффициентов, и решает задачу в ортогональном базисе, "
                "поэтому высокие степени не превращают расчёт в мусор.",
                "! Тестовые точки в эту сумму не входят: ни одна из них не "
                "влияет на коэффициенты.",
            ),
        ),
        "train": (
            "4. Обучающие точки",
            (
                "Обучающие точки — те, по которым модель строит кривую. На "
                "графике это синие кружки.",
                "Разбиение равномерное: каждая k-я точка уходит в тест, "
                "поэтому обучение покрыто всей областью, а не одним её "
                "куском.",
                "Ползунок «доля обучающих точек» меняет только это: набор "
                "остаётся прежним, меняется лишь то, что модель видит.",
            ),
        ),
        "test": (
            "5. Тестовые точки",
            (
                "Тестовые точки — оранжевые квадраты. Модель их не видела: они "
                "исключены из подгонки до того, как найден первый коэффициент.",
                "По ним считается вторая ошибка — та, которая отвечает на "
                "вопрос «насколько кривая годится для данных, которых модель не "
                "смотрела».",
                "! Если бы тестовые точки попали в подгонку, вторая ошибка "
                "перестала бы быть проверкой: приложение так не делает ни при "
                "каких настройках.",
            ),
        ),
        "mse": (
            "6. Средняя квадратичная ошибка (MSE)",
            (
                "MSE — среднее значение квадратов отклонений предсказания от "
                "ответа:",
                "»MSE = (1 / N) · Σ (yᵢ − ŷᵢ)²",
                "Квадрат делает большие ошибки дороже малых и убирает знак, "
                "поэтому ошибка не компенсируется с разных сторон кривой.",
                "MSE обучения и MSE теста считаются одной и той же формулой, но "
                "на разных наборах точек и с разным числом N. Сравнивать их "
                "законно: метрика одна и та же, и на этом сравнении держатся и "
                "отношение теста к обучению, и вывод режима. Неравны не сами "
                "числа, а их точность: за каждой оценкой стоит своё число "
                "точек, поэтому при малом тесте тестовая ошибка колеблется "
                "сильнее. Смотрите, сколько точек стоит за каждой цифрой.",
                "Эта же метрика вводилась в проекте «Loss Functions Lab».",
            ),
        ),
        "complexity": (
            "7. Сложность модели",
            (
                "Сложность здесь — степень многочлена, то есть число "
                "коэффициентов. Чем она больше, тем более изогнутую кривую "
                "модель умеет воспроизвести.",
                "На вкладке «Сложность и ошибка» приложение перебирает все "
                "степени от 1 до 9 на текущем разбиении и строит две кривые: "
                "ошибка обучения и ошибка теста.",
                "Кривая обучения убывает почти всегда: дополнительная свобода "
                "всегда помогает подстроиться под те точки, которые модель "
                "видит. Кривая теста имеет минимум — правее него добавочная "
                "свобода начинает работать против модели.",
            ),
        ),
        "under": (
            "8. Недообучение",
            (
                "Модели не хватает ёмкости: она не описывает даже ту "
                "закономерность, которая видна глазами.",
                "Признак: обе ошибки велики и близки друг к другу. Прямая "
                "линия, протянутая через виток синуса, ошибается и на своих "
                "точках, и на чужих одинаково сильно.",
                "Язык интуиции: большое смещение (high bias). Модель слишком "
                "стеснена, её класс функций не содержит нужной формы.",
            ),
        ),
        "good": (
            "9. Удачное соответствие",
            (
                "Модель повторяет закономерность, не пытаясь пройти через каждую "
                "точку. Ошибка обучения небольшая, ошибка теста — около лучшей, "
                "какую эти данные вообще дают.",
                "Заметьте: «удачное» не значит «самая малая ошибка обучения». "
                "Минимум всегда у более свободной модели, и он не является "
                "целью.",
                "Именно эта точка на кривой сложности и есть цель подбора "
                "модели: не максимум свободы, а достаточная.",
            ),
        ),
        "over": (
            "10. Переобучение",
            (
                "Модели хватает свободы, и она использует её на шум: кривая "
                "изгибается так, чтобы пройти через удачные отклонения "
                "конкретных точек.",
                "Признак: ошибка обучения мала, а ошибка теста заметно больше. "
                "На графике видно, как кривая колышется между синими кружками.",
                "Язык интуиции: большой разброс (high variance). Мысленно "
                "сдвиньте одну обучающую точку — кривая изменится сильнее, чем "
                "должна. Перетаскивания точек в окне нет: двигаются только "
                "настройки — набор, шум, степень и доля обучения, и вместе с "
                "ними уезжает вся кривая.",
                "! Переобучение — свойство не «степени 9», а соотношения "
                "свободы модели, шума и числа обучающих точек.",
            ),
        ),
        "general": (
            "11. Обобщение",
            (
                "Обобщение — способность модели сохранять точность на данных, "
                "которые в неё не входили.",
                "Тестовая ошибка — самая честная оценка обобщения из того, что "
                "в этом проекте есть. Именно поэтому за ней следят, а не за "
                "ошибкой обучения.",
                "Практический вывод: сравнивайте две модели по тесту, а не по "
                "тому, какая лучше прошла свои точки.",
            ),
        ),
        "biasvar": (
            "12. Смещение и разброс: только интуиция",
            (
                "Смещение — систематическая ошибка слишком простой модели: она "
                "всегда тянет предсказание к упрощённой форме.",
                "Разброс — чувствительность к конкретной выборке: другая "
                "выборка тех же данных даст другую кривую.",
                "Недообучению соответствует большое смещение, переобучению — "
                "большой разброс. Между ними и лежит удачное соответствие.",
                "! Строгого разложения ошибки на смещение, разброс и шум здесь "
                "нет: для него нужен многократный перезапуск на разных "
                "выборках, а этот проект показывает одну настройку за раз.",
            ),
        ),
        "verdict": (
            "13. Как приложение подписывает режим",
            (
                "Подпись ставится по рассчитанным ошибкам, а не по значению "
                "степени. Правила те же для любого набора и открыто показаны на "
                "вкладке «Модель»:",
                "»1. тестовая ошибка ≤ порога → удачное соответствие",
                "»2. иначе тест ≥ 2 · обучения и степень правее лучшей → "
                "переобучение",
                "»3. иначе степень левее лучшей и разрыва нет → недообучение",
                "»4. иначе → переходный случай",
                "Порог «всё ещё хорошо» — большее из двух чисел: двойная лучшая "
                "тестовая ошибка этого опыта и один процент ошибки "
                "бесполезной модели (константы, равной среднему по обучению).",
                "! Это учебные пороги, а не научная теорема. Числа, по которым "
                " идёт вывод, видны всегда — спорить с подписью можно по ним.",
            ),
        ),
        "howto": (
            "14. Как работать с симулятором",
            (
                "1. Выберите набор и посмотрите на серый пунктир — это форма, "
                "которую модель пытается угадать.",
                "2. Поставьте степень 1 и шум 0.35: обе ошибки велики и почти "
                "равны. Это недообучение.",
                "3. Поднимайте степень по одной: ошибка обучения будет падать, "
                "а тестовая сначала уменьшится. Найдите момент, когда тест "
                "перестал улучшаться.",
                "4. Продолжайте поднимать степень: кривая начинает колыхаться "
                "между точками, тестовая ошибка растёт. Это переобучение.",
                "5. Откройте вкладку «Сложность и ошибка» — там весь диапазон "
                "степеней сразу, с минимумом тестовой кривой.",
                "6. Уменьшите долю обучающих точек: переобучение при тех же "
                "степенях станет резче.",
                "7. Поставьте шум 0.00 и высокую степень: без шума лишняя "
                "свобода почти не вредит — вывод о переобучении зависит от "
                "данных, а не от степени.",
            ),
        ),
        "limits": (
            "15. Границы метода и чего здесь нет",
            (
                "Данные синтетические: форма известна заранее, поэтому «истинная "
                "закономерность» нарисована. В реальных задачах её никто не "
                "показывает.",
                "Один признак, регрессия, одна метрика MSE. Классификация, "
                "многомерность и другие функции потерь — отдельные проекты "
                "серии.",
                "Нет регуляризации, перекрёстной проверки, ансамблей и "
                "градиентного спуска: они начинаются там, где этот проект "
                "заканчивает объяснение причины.",
                "Выбор степени по тестовой ошибке использует тест дважды: как "
                "критерий выбора и как оценку обобщения. Поэтому показанная "
                "тестовая ошибка при многократном выборе оптимистична; для "
                "честной оценки нужны повторы и резервная выборка, а их в этом "
                "проекте нет.",
                "Степень ограничена девятой: при более высоких степенях "
                "многочлен на коротком интервале становится численно "
                "неустойчивым, и учебный вывод тонет в ошибках округления.",
            ),
        ),
    },
    "en": {
        "idea": (
            "1. What the application shows",
            (
                "One dataset, one kind of model — a polynomial — and only the "
                "complexity moves. On screen the training error falls while the "
                "test error first falls and then rises.",
                "The chain: complexity → training error → test error → "
                "generalization.",
                "! Nothing is trained by gradient descent: the coefficients "
                "come from one least-squares solve, not from iterations.",
            ),
        ),
        "polynomial": (
            "2. The model: a polynomial of one variable",
            (
                "The model predicts with a single formula:",
                "»ŷ = b₀ + b₁x + b₂x² + … + bₙxⁿ",
                "The degree fixes the number of coefficients: always n + 1. "
                "Degree 1 is a line, 2 a parabola, 3 a cubic curve. Each new "
                "coefficient gives the curve one more possible bend.",
                "The Model tab shows every fitted coefficient, so any "
                "prediction can be recomputed by hand from the same formula.",
            ),
        ),
        "fitting": (
            "3. How the coefficients are chosen",
            (
                "Coefficients are not searched one by one; the minimum of the "
                "sum of squared residuals over the training points is found:",
                "»S = Σ (yᵢ − ŷᵢ)²  — over training points only",
                "That is the classic least-squares problem: with too few "
                "points its solution is not unique. The application requires "
                "at least one more training point than there are coefficients "
                "and solves the system in an orthogonal basis, so high degrees "
                "do not turn the arithmetic into junk.",
                "! Test points never enter this sum: none of them influences a "
                "single coefficient.",
            ),
        ),
        "train": (
            "4. Training points",
            (
                "Training points are the ones the model builds its curve from — "
                "the blue circles.",
                "The split is even: every k-th point goes to the test, so the "
                "training set covers the whole range instead of one piece of "
                "it.",
                "The training-share slider changes only that. The dataset "
                "stays; what changes is how much of it the model is allowed to "
                "see.",
            ),
        ),
        "test": (
            "5. Test points",
            (
                "Test points are the orange squares. The model never saw them: "
                "they are taken out of the fit before the first coefficient is "
                "found.",
                "They produce the second error, the one that answers “how good "
                "is this curve on data the model did not look at”.",
                "! If test points entered the fit, the second error would stop "
                "being a check. The application never does that, at any "
                "setting.",
            ),
        ),
        "mse": (
            "6. Mean squared error (MSE)",
            (
                "MSE is the mean of the squared differences between prediction "
                "and answer:",
                "»MSE = (1 / N) · Σ (yᵢ − ŷᵢ)²",
                "Squaring makes large errors cost more than small ones and "
                "removes the sign, so mistakes on opposite sides of the curve "
                "do not cancel.",
                "Both errors use the same formula but different N. Comparing "
                "them is legitimate: the metric is one and the same, and both "
                "the test-to-training ratio and the verdict rest on that "
                "comparison. What is not equal is the precision — each estimate "
                "stands on its own number of points, so a small test makes the "
                "test error wobble more. Check how many points stand behind "
                "each number.",
                "The same metric was introduced in the Loss Functions Lab.",
            ),
        ),
        "complexity": (
            "7. Model complexity",
            (
                "Complexity here means the polynomial degree, that is the "
                "number of coefficients. More of them, and the model can "
                "reproduce a more wiggly line.",
                "The Complexity tab fits every degree from 1 to 9 on the "
                "current split and draws two curves: training error and test "
                "error.",
                "The training curve keeps falling: extra freedom always helps "
                "with the points the model can see. The test curve has a "
                "minimum — right of it extra freedom starts working against the "
                "model.",
            ),
        ),
        "under": (
            "8. Underfitting",
            (
                "The model has too little capacity: it misses even the pattern "
                "an eye sees at once.",
                "Sign: both errors are large and close to each other. A "
                "straight line drawn through a turn of a sine makes comparable "
                "mistakes on its own and on foreign points.",
                "Intuitive language: high bias. The model is too constrained — "
                "its class of functions does not contain the shape.",
            ),
        ),
        "good": (
            "9. Good fit",
            (
                "The model follows the pattern without trying to pass through "
                "every point. The training error is small, the test error is "
                "close to the best these data produce.",
                "Note that “good” does not mean “smallest training error”. The "
                "smallest training error always belongs to the most flexible "
                "model, and that is not the goal.",
                "This point on the complexity curve is what model selection "
                "looks for: not maximum freedom, but enough freedom.",
            ),
        ),
        "over": (
            "10. Overfitting",
            (
                "The model has freedom and spends it on noise: the curve bends "
                "so as to pass through the lucky deviations of particular "
                "points.",
                "Sign: the training error is small while the test error is "
                "clearly larger. On the graph the curve wiggles between the "
                "blue circles.",
                "Intuitive language: high variance. Shift one training point "
                "in a thought experiment — the curve changes more than it "
                "should. There is no point-dragging in the window: only the "
                "settings move, that is the dataset, the noise, the degree and "
                "the training share, and with them the whole curve moves.",
                "! Overfitting is not a property of “degree 9” but of the "
                "relation between freedom, noise and the number of training "
                "points.",
            ),
        ),
        "general": (
            "11. Generalization",
            (
                "Generalization is the model staying accurate on data that "
                "never entered it.",
                "The test error is the most honest estimate of it available "
                "here, which is why it is watched instead of the training "
                "error.",
                "Practical conclusion: compare two models by their test error, "
                "not by which one passed its own points better.",
            ),
        ),
        "biasvar": (
            "12. Bias and variance: intuition only",
            (
                "Bias is the systematic error of an overly simple model: it "
                "always pulls predictions toward a simplified shape.",
                "Variance is sensitivity to the particular sample: another "
                "sample of the same data would give another curve.",
                "Underfitting corresponds to high bias, overfitting to high "
                "variance, and a good fit sits between them.",
                "! There is no strict decomposition of the error here: it needs "
                "many repeated samples, and this project shows one "
                "configuration at a time.",
            ),
        ),
        "verdict": (
            "13. How the application labels the regime",
            (
                "The label comes from the computed errors, not from the degree. "
                "One rule for every dataset, and it is printed on the Model "
                "tab:",
                "»1. test error ≤ threshold → good fit",
                "»2. otherwise test ≥ 2 · training and degree right of the best "
                "→ overfitting",
                "»3. otherwise degree left of the best and no gap → "
                "underfitting",
                "»4. otherwise → transition case",
                "The “still good” threshold is the larger of two numbers: twice "
                "the best test error of this very experiment, and one percent "
                "of the no-skill error (a constant equal to the training mean).",
                "! These are teaching thresholds, not a theorem. Every number "
                "behind the label stays on screen, so the label can be argued "
                "with.",
            ),
        ),
        "howto": (
            "14. How to work with the simulator",
            (
                "1. Pick a dataset and look at the grey dashes — that is the "
                "shape the model tries to guess.",
                "2. Set degree 1 and noise 0.35: both errors are large and "
                "nearly equal. That is underfitting.",
                "3. Raise the degree one by one: training error keeps falling, "
                "the test error first falls. Find the point where it stops.",
                "4. Keep raising it: the curve starts to waver between points "
                "and the test error grows. That is overfitting.",
                "5. Open the Complexity tab to see the whole degree range at "
                "once, with the minimum of the test curve marked.",
                "6. Shrink the training share: overfitting at the same degrees "
                "becomes sharper.",
                "7. Set noise 0.00 with a high degree: without noise surplus "
                "freedom barely harms — the verdict depends on the data, not on "
                "the degree alone.",
            ),
        ),
        "limits": (
            "15. Limits of the method and what is not here",
            (
                "The data are synthetic: the shape is known in advance, so the "
                "underlying pattern can be drawn. Real problems never show it.",
                "One feature, regression, one metric. Classification, "
                "multidimensional data and other loss functions are separate "
                "projects of the series.",
                "No regularization, cross-validation, ensembles or gradient "
                "descent: they begin where this project stops explaining the "
                "cause.",
                "Choosing the degree by test error uses the test twice: as the "
                "selection criterion and as the estimate of generalization. The "
                "test error on screen is therefore optimistic under repeated "
                "selection; an honest estimate needs repeats and a held-out "
                "set, and this project has neither.",
                "The degree is capped at nine: on a short interval higher "
                "polynomials become numerically unstable, and rounding noise "
                "would drown the lesson.",
            ),
        ),
    },
}


# ----------------------------------------------------------------------
# Раздел 8. Виджеты: параметры, основной график, кривая сложности
# ----------------------------------------------------------------------

def to_text(value):
    return fmt(value, 2)


def _trim_number(value):
    """Подпись шага на кнопке: 0.05 -> '0.05', 1.0 -> '1'."""
    text = ("%.2f" % value).rstrip("0").rstrip(".")
    return text or "0"


def fill_text(widget, lines):
    """Заполняет read-only текстовое окно парами (тег, строка)."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for tag, text in lines:
        widget.insert("end", text + "\n", tag)
    widget.configure(state="disabled")


def nice_step(raw):
    """Круглый шаг осевой шкалы: 1, 2 или 5 в какой-то десятичной степени."""
    if raw <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw))
    base = 10.0 ** exponent
    for factor in (1.0, 2.0, 5.0, 10.0):
        if factor * base >= raw:
            return factor * base
    return 10.0 * base


def tick_values(low, high, wanted=6):
    """Метки оси внутри диапазона, включая границы диапазона."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [low]
    step = nice_step((high - low) / max(1, wanted - 1))
    first = math.ceil(low / step) * step
    values = []
    v = first
    while v <= high + step * 1e-6 and len(values) < 40:
        values.append(round(v, 10))
        v += step
    return values


NICE_MSE = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
            1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
            2000.0, 5000.0)


def mse_ticks(vmax, wanted=5):
    """Метки для шкалы √MSE: настоящие значения MSE, расположенные по √."""
    if vmax is None or vmax <= 0 or not math.isfinite(vmax):
        return [0.0]
    picks = [v for v in NICE_MSE if v >= vmax * 0.13 and v <= vmax * 1.05]
    if not picks:
        picks = [vmax]
    while len(picks) > wanted:
        picks = picks[1::2]
    if not picks or picks[-1] < vmax:
        picks.append(vmax)
    return picks


class ScrollFrame(ttk.Frame):
    """Вертикально прокручиваемая колонка: левый столбец выше экрана."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=COL_BG,
                                borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical",
                                 command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner,
                                                anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.bind("<Enter>", lambda e: self.canvas.bind(
            "<MouseWheel>", self._on_wheel))
        self.bind("<Leave>", lambda e: self.canvas.unbind("<MouseWheel>"))

    def _on_inner(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 90), "units")


class SliderControl(ttk.Frame):
    """Один параметр: подпись, слайдер, поле ввода и кнопки шага.

    Режим integer используется для степени: шаг целый, ошибки дробного
    ввода объясняются отдельно, а слайдер не показывает промежуточных
    значений.
    """

    def __init__(self, master, app, label_key, low, high, value, step, nudge,
                 integer=False, fmt_fun=None, state_attr=None,
                 nudge_text=None):
        super().__init__(master)
        self.app = app
        self.label_key = label_key
        self.low = low
        self.high = high
        self.step = step
        self.nudge = nudge
        self.integer = integer
        self.state_attr = state_attr
        self.nudge_text = nudge_text
        self.fmt_fun = fmt_fun or (lambda v: fmt(v, 2))
        self.value = self._round(value)
        self._echo = False

        self.label = ttk.Label(self, text=T(label_key),
                               font=app.fonts["smallb"])
        self.label.grid(row=0, column=0, columnspan=4, sticky="w")

        self.var_num = tk.DoubleVar(value=self.value)
        self.var_text = tk.StringVar(value=self._text(self.value))
        # Цвет бегунка tk.Scale берётся из bg: на светлом фоне ползунок не
        # виден, поэтому bg — акцентный, а дорожка — «бумажная».
        self.scale = tk.Scale(
            self, from_=low, to=high, resolution=step, orient="horizontal",
            showvalue=False, length=140, sliderrelief="flat", sliderlength=22,
            bd=0, highlightthickness=1, highlightbackground=COL_FRAME,
            troughcolor=COL_PAPER, bg=COL_HEAD, fg=COL_TEXT,
            variable=self.var_num, command=self._from_slider)
        self.scale.grid(row=1, column=0, sticky="w")

        self.entry = ttk.Entry(self, textvariable=self.var_text, width=7,
                               justify="right", font=app.fonts["mono"])
        self.entry.grid(row=1, column=1, padx=(6, 0), sticky="w")
        self.entry.bind("<Return>", self._commit)
        self.entry.bind("<FocusOut>", self._commit)

        step_text = self.nudge_text or _trim_number(self.nudge)
        self.down = ttk.Button(self, text="-" + step_text, width=5,
                               command=lambda: self._nudge(-self.nudge))
        self.up = ttk.Button(self, text="+" + step_text, width=5,
                             command=lambda: self._nudge(self.nudge))
        self.down.grid(row=1, column=2, padx=(5, 0), sticky="w")
        self.up.grid(row=1, column=3, padx=(3, 0), sticky="w")

    def _text(self, value):
        if self.integer:
            return str(int(value))
        return self.fmt_fun(value)

    def _push(self):
        """Переносит значение в состояние приложения перед перерасчётом."""
        if self.state_attr is not None:
            setattr(self.app, self.state_attr, self.value)

    def _round(self, value):
        value = clamp(value, self.low, self.high)
        if self.integer:
            return int(round(value))
        return round(round(value / self.step) * self.step, 6)

    def _from_slider(self, _raw):
        if self._echo:
            return
        self.value = self._round(self.var_num.get())
        self._echo = True
        self.var_text.set(self._text(self.value))
        self.var_num.set(self.value)
        self._echo = False
        self._mark(False)
        self._push()
        self.app.refresh()
        self.app.announce_params()

    def _commit(self, _event=None):
        raw = self.var_text.get()
        parsed = parse_int(raw) if self.integer else parse_number(raw)
        if parsed is None:
            self._mark(True)
            self.app.set_status("status_invalid", f=T(self.label_key))
            return
        value = self._round(parsed)
        clamped = value != parsed
        if clamped:
            if self.integer and int(round(parsed)) == value:
                self.app.set_status("status_int", raw=fmt(parsed, 2),
                                    value=str(value))
            else:
                self.app.set_status("status_clamped", raw=fmt(parsed, 3),
                                    value=self._text(value),
                                    lo=self._text(self.low),
                                    hi=self._text(self.high))
        self.value = value
        self._mark(False)
        self.var_text.set(self._text(value))
        self._echo = True
        self.var_num.set(value)
        self._echo = False
        self._push()
        self.app.refresh()
        if not clamped:
            self.app.announce_params()

    def _nudge(self, delta):
        # В поле кладём число, а не отображаемый текст: доля показана как
        # «70%», и _commit отверг бы такую запись как не число.
        self.var_text.set(str(self._round(self.value + delta)))
        self._commit()

    def set_value(self, value):
        self.value = self._round(value)
        self.var_text.set(self._text(self.value))
        self._echo = True
        self.var_num.set(self.value)
        self._echo = False
        self._mark(False)
        self._push()

    def _mark(self, bad):
        self.entry.configure(style="BadEntry.TEntry" if bad else
                             "GoodEntry.TEntry")

    def retranslate(self):
        self.label.configure(text=T(self.label_key))


class MainPlot(tk.Canvas):
    """Основной график: точки, подобранная кривая, истинная зависимость."""

    def __init__(self, master, app):
        super().__init__(master, bg="white", highlightthickness=2,
                         highlightbackground=COL_FRAME)
        self.app = app
        self.bind("<Configure>", lambda _e: self.redraw())

    def _frame(self, y_low, y_high):
        width = max(self.winfo_width(), 240)
        height = max(self.winfo_height(), 200)
        self.plot_left = PAD_LEFT
        self.plot_top = PAD_TOP
        self.plot_right = width - PAD_RIGHT
        self.plot_bottom = height - PAD_BOTTOM
        x_span = (X_MAX - X_MIN) + 0.4
        y_span = max(y_high - y_low, 1e-6)
        self.unit_x = (self.plot_right - self.plot_left) / x_span
        self.unit_y = (self.plot_bottom - self.plot_top) / y_span
        self.x_low = X_MIN - 0.2
        self.x_high = X_MAX + 0.2
        self.y_low = y_low
        self.y_high = y_high

    def px(self, x):
        return self.plot_left + (x - self.x_low) * self.unit_x

    def py(self, y):
        return self.plot_bottom - (y - self.y_low) * self.unit_y

    def redraw(self):
        self.delete("all")
        state = self.app.result
        if state is None:
            return
        y_low, y_high = self.app.y_bounds
        self._frame(y_low, y_high)
        self._draw_grid()
        self._draw_truth()
        self._draw_curve()
        self._draw_points()
        self._draw_legend()

    def _draw_grid(self):
        for x in tick_values(self.x_low, self.x_high, 8):
            px = self.px(x)
            self.create_line(px, self.plot_top, px, self.plot_bottom,
                             fill=COL_GRID)
            self.create_text(px, self.plot_bottom + 12, text=fmt(x, 1),
                             fill=COL_AXIS, font=self.app.fonts["tiny"])
        for y in tick_values(self.y_low, self.y_high, 7):
            py = self.py(y)
            self.create_line(self.plot_left, py, self.plot_right, py,
                             fill=COL_GRID)
            self.create_text(self.plot_left - 8, py, text=fmt(y, 1),
                             fill=COL_AXIS, font=self.app.fonts["tiny"],
                             anchor="e")
        self.create_rectangle(self.plot_left, self.plot_top, self.plot_right,
                              self.plot_bottom, outline=COL_AXIS, width=1)
        self.create_text((self.plot_left + self.plot_right) // 2,
                         self.plot_bottom + 30,
                         text=T("axis_x"), fill=COL_AXIS,
                         font=self.app.fonts["tiny"])
        # Подпись оси значений — вертикально слева: горизонтальная надпись не
        # умещается в левом поле графика и обрезалась бы.
        self.create_text(14, (self.plot_top + self.plot_bottom) // 2,
                         text=T("axis_y"), fill=COL_AXIS, angle=90,
                         font=self.app.fonts["tiny"])

    def _polyline(self, pairs, color, width, dash=None, cap="round"):
        flat = []
        run = []
        for x, y in pairs:
            if y is None or not math.isfinite(y) or y < self.y_low or \
                    y > self.y_high:
                if len(run) >= 2:
                    flat.append(run)
                run = []
                continue
            run.append(self.px(x))
            run.append(self.py(y))
        if len(run) >= 2:
            flat.append(run)
        for segment in flat:
            self.create_line(*segment, fill=color, width=width, dash=dash,
                             capstyle=cap, smooth=False)

    def _draw_truth(self):
        entry = SHAPE_BY_KEY[self.app.shape_key]
        pairs = []
        for i in range(161):
            x = X_MIN + (X_MAX - X_MIN) * i / 160.0
            pairs.append((x, entry["fn"](x)))
        self._polyline(pairs, COL_TRUTH, 2, dash=(5, 4))

    def _draw_curve(self):
        current = self.app.result.get("current")
        if current is None or not current["ok"]:
            return
        coeffs = current["coeffs"]
        pairs = []
        for i in range(CURVE_SAMPLES + 1):
            x = self.x_low + (self.x_high - self.x_low) * i / CURVE_SAMPLES
            pairs.append((x, horner(coeffs, x)))
        self._polyline(pairs, COL_FIT, 3)

    def _draw_points(self):
        state = self.app.result
        for i, (x, y) in state["test"]:
            px, py = self.px(x), self.py(y)
            self.create_rectangle(px - 5, py - 5, px + 5, py + 5,
                                  fill=COL_FILL_TEST, outline=COL_TEST,
                                  width=2)
        for i, (x, y) in state["train"]:
            px, py = self.px(x), self.py(y)
            self.create_oval(px - 5, py - 5, px + 5, py + 5,
                             fill=COL_TRAIN, outline="#1b3a55", width=1)

    def _draw_legend(self):
        items = ((COL_TRAIN, "legend_train", "dot"),
                 (COL_TEST, "legend_test", "square"),
                 (COL_FIT, "legend_fit", "line"),
                 (COL_TRUTH, "legend_truth", "dash"))
        pad = 7
        x0 = self.plot_left + 10
        y0 = self.plot_top + 8
        rows = []
        for color, key, mark in items:
            text = T(key)
            width = self.app.fonts["tiny"].measure(text)
            rows.append((color, text, mark, width))
        box_w = max(r[3] for r in rows) + 46
        box_h = len(rows) * 16 + 2 * pad - 4
        self.create_rectangle(x0, y0, x0 + box_w, y0 + box_h, fill="white",
                              outline=COL_FRAME)
        for index, (color, text, mark, _w) in enumerate(rows):
            cy = y0 + pad + index * 16 + 4
            cx = x0 + 14
            if mark == "dot":
                self.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill=color,
                                 outline="#1b3a55")
            elif mark == "square":
                self.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4,
                                      fill=COL_FILL_TEST, outline=color,
                                      width=2)
            else:
                dash = None if mark == "line" else (4, 3)
                self.create_line(cx - 11, cy, cx + 11, cy, fill=color,
                                 width=3 if mark == "line" else 2, dash=dash)
            self.create_text(cx + 18, cy, text=text, anchor="w",
                             fill=COL_TEXT, font=self.app.fonts["tiny"])


class ErrorBars(tk.Canvas):
    """Компактное сравнение двух ошибок в одной шкале."""

    def __init__(self, master, app, height=96):
        super().__init__(master, height=height, bg=COL_PAPER,
                         highlightthickness=1,
                         highlightbackground=COL_FRAME)
        self.app = app
        self.fixed_height = height
        self.bind("<Configure>", lambda _e: self.redraw())

    def redraw(self):
        self.delete("all")
        current = self.app.result.get("current") if self.app.result else None
        width = max(self.winfo_width(), 200)
        # Полоски рисуются в собственной шкале: если виджет ещё не разложен,
        # берётся запрошенная высота, иначе фактическая.
        height = max(self.fixed_height, self.winfo_height())
        if current is None or not current["ok"]:
            self.create_text(width // 2, height // 2, text=T("bars_nofit"),
                             fill=COL_DIM, font=self.app.fonts["small"],
                             width=width - 30)
            return
        rows = ((T("bars_train"), current["train_mse"], COL_TRAIN,
                 COL_FILL_TRAIN),
                (T("bars_test"), current["test_mse"], COL_TEST,
                 COL_FILL_TEST))
        top = 12
        bar_h = 18
        label_w = 74
        value_w = 78
        left = label_w + 6
        right = width - value_w - 8
        scale_max = max(r[1] for r in rows) or 1.0
        for index, (name, value, color, fill) in enumerate(rows):
            y = top + index * (bar_h + 12)
            self.create_text(4, y + bar_h // 2, text=name, anchor="w",
                             fill=COL_TEXT, font=self.app.fonts["tiny"])
            length = max(2.0, (right - left) * (value / scale_max))
            self.create_rectangle(left, y, max(left + 2, right), y + bar_h,
                                  fill=COL_PAPER, outline=COL_FRAME)
            self.create_rectangle(left, y, left + length, y + bar_h,
                                  fill=fill, outline=color, width=2)
            self.create_text(right + 8, y + bar_h // 2,
                             text=fmt_error(value), anchor="w",
                             fill=color, font=self.app.fonts["monob"])
        note_y = top + 2 * (bar_h + 12) + 2
        if note_y < height - 6:
            self.create_text(4, note_y, text=T("bars_note"), anchor="w",
                             fill=COL_DIM, font=self.app.fonts["tiny"],
                             width=width - 12)


class ComplexityPlot(tk.Canvas):
    """Ошибка против сложности: две кривые по всем степеням."""

    def __init__(self, master, app):
        super().__init__(master, height=CX_PLOT_HEIGHT, bg=COL_PAPER,
                         highlightthickness=1,
                         highlightbackground=COL_FRAME)
        self.app = app
        self.bind("<Configure>", lambda _e: self.redraw())

    def redraw(self):
        self.delete("all")
        result = self.app.result
        if result is None:
            return
        width = max(self.winfo_width(), 260)
        height = max(self.winfo_height(), CX_PLOT_HEIGHT)
        self.left = PAD2_LEFT
        self.top = PAD2_TOP
        self.right = width - PAD2_RIGHT
        self.bottom = height - PAD2_BOTTOM
        rows = result["rows"]
        values = []
        for row in rows:
            if row["ok"]:
                values.append(row["train_mse"])
                values.append(row["test_mse"])
        if not values:
            self.create_text(width // 2, height // 2, text=T("cx_nofit"),
                             fill=COL_DIM, font=self.app.fonts["small"])
            return
        self.vmax = max(values)
        self.sqrt_max = math.sqrt(self.vmax)
        self._draw_frame(rows)
        self._draw_series(rows, "train_mse", COL_TRAIN)
        self._draw_series(rows, "test_mse", COL_TEST)
        self._draw_marks()
        # Легенда рисуется последней: она на белом фоне и должна перекрывать
        # линии, а не наоборот.
        self._draw_legend()

    def _deg_x(self, degree):
        span = DEG_MAX - DEG_MIN
        if span <= 0:
            return self.left
        return self.left + (self.right - self.left) * (degree - DEG_MIN) / span

    def _val_y(self, value):
        if value is None or not math.isfinite(value) or value < 0:
            return self.bottom
        ratio = math.sqrt(value) / self.sqrt_max if self.sqrt_max > 0 else 0.0
        return self.bottom - (self.bottom - self.top) * ratio

    def _draw_frame(self, rows):
        self.create_rectangle(self.left, self.top, self.right, self.bottom,
                              outline=COL_AXIS)
        for value in mse_ticks(self.vmax):
            y = self._val_y(value)
            self.create_line(self.left, y, self.right, y, fill=COL_GRID)
            self.create_text(self.left - 6, y, text=fmt_error(value),
                             anchor="e", fill=COL_AXIS,
                             font=self.app.fonts["tiny"])
        for row_index, row in enumerate(rows):
            degree = DEG_MIN + row_index
            x = self._deg_x(degree)
            self.create_text(x, self.bottom + 11, text=str(degree),
                             fill=COL_AXIS, font=self.app.fonts["tiny"])
            if not row["ok"]:
                self.create_line(x, self.bottom, x, self.bottom - 6,
                                 fill=COL_DIM)
        self.create_text((self.left + self.right) // 2, self.bottom + 27,
                         text=T("cx_axis_x"), fill=COL_AXIS,
                         font=self.app.fonts["tiny"])
        self.create_text(14, (self.top + self.bottom) // 2,
                         text=T("cx_axis_y"), fill=COL_AXIS, angle=90,
                         font=self.app.fonts["tiny"])

    def _draw_series(self, rows, field, color):
        points = []
        for row in rows:
            if not row["ok"]:
                continue
            points.append((self._deg_x(row["degree"]), self._val_y(row[field])))
        if len(points) >= 2:
            flat = [c for pair in points for c in pair]
            self.create_line(*flat, fill=color, width=2)
        for x, y in points:
            self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color,
                             outline=color)

    def _draw_legend(self):
        rows = ((COL_TRAIN, "line", "cx_legend_train"),
                (COL_TEST, "line", "cx_legend_test"))
        x = self.left + 8
        y = self.top + 6
        widths = [self.app.fonts["tiny"].measure(T(key))
                  for _c, _mark, key in rows]
        box_w = max(widths) + 40
        box_h = len(rows) * 15 + 8
        self.create_rectangle(x, y, x + box_w, y + box_h, fill="white",
                              outline=COL_FRAME)
        for index, (color, _mark, key) in enumerate(rows):
            cy = y + 8 + index * 15
            self.create_line(x + 8, cy, x + 24, cy, fill=color, width=2)
            self.create_text(x + 28, cy, text=T(key), anchor="w",
                             fill=COL_TEXT, font=self.app.fonts["tiny"])

    def _draw_marks(self):
        """Отметки без подписей: что они значат, написано под графиком."""
        result = self.app.result
        degree = self.app.degree
        x = self._deg_x(degree)
        self.create_line(x, self.top, x, self.bottom, fill=COL_CURRENT,
                         width=2, dash=(4, 3))
        best = result.get("best_degree")
        if best is not None:
            bx = self._deg_x(best)
            row = result["rows"][best - DEG_MIN]
            if row["ok"]:
                by = self._val_y(row["test_mse"])
                self.create_oval(bx - 7, by - 7, bx + 7, by + 7,
                                 outline=COL_BEST, width=2)


# ----------------------------------------------------------------------
# Раздел 9. Главное окно
# ----------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title(T("app_title"))
        self.configure(bg=COL_BG)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        # Ширины 1288 требует сам макет: 393 + 420 + 439 пиксела колонок плюс
        # поля. Меньше — виджеты начнут обрезаться, поэтому минимум честный.
        self.minsize(min(MIN_WIDTH, screen_w - 40), min(640, screen_h - 60))
        self.geometry("%dx%d" % (
            min(max(1240, int(screen_w * 0.9)), screen_w - 20),
            min(max(780, int(screen_h * 0.88)), screen_h - 20)))
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        base = tkfont.nametofont("TkDefaultFont").actual("family")
        mono = tkfont.nametofont("TkFixedFont").actual("family")

        def make_font(family, size, weight="normal"):
            return tkfont.Font(family=family, size=size, weight=weight)

        # Шрифты — объекты Font: их передают в font= и измерают линейкой
        # canvas, чтобы подписи не обрезались.
        self.fonts = {
            "title": make_font(base, 16, "bold"),
            "sub": make_font(base, 13),
            "midb": make_font(base, 13, "bold"),
            "small": make_font(base, 12),
            "smallb": make_font(base, 12, "bold"),
            "tiny": make_font(base, 10),
            "mono": make_font(mono, 12),
            "monob": make_font(mono, 12, "bold"),
        }
        self.option_add("*Font", (base, 13))
        self._setup_styles()

        # Состояние: язык его не трогает.
        self.shape_key = DEFAULT_SHAPE
        self.degree = DEFAULT_DEGREE
        self.sigma = DEFAULT_SIGMA
        self.share = DEFAULT_SHARE
        self.experiment_key = None
        self.help_key = HELP_KEYS[0]
        self.points = ()
        self.result = None
        self.y_bounds = (-2.0, 2.0)
        self._text_refs = []
        self._table_heads = []
        self.var_status = tk.StringVar()
        self.var_lang = tk.StringVar(value=LANG)

        self._build()
        self._fill_shape_combo()
        self._fill_exp_list()
        self._fill_help_list()
        self.refresh()
        self._show_help()
        self._show_exp_goal()
        self.set_status("status_start", name=self.shape_name(),
                        d=self.degree, s=fmt(self.sigma),
                        p=fmt_share(self.share),
                        n=len(self.result["train_pts"]),
                        m=len(self.result["test_pts"]))

    # --- оформление ---------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook.Tab", padding=(9, 5))
        style.configure("TLabelframe", background=COL_BG)
        style.configure("TLabelframe.Label", background=COL_BG,
                        foreground=COL_HEAD, font=self.fonts["smallb"])
        style.configure("TFrame", background=COL_BG)
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["title"])
        style.configure("Sub.TLabel", background=COL_BG, foreground="#5a5a54",
                        font=self.fonts["small"])
        style.configure("Key.TLabel", background=COL_BG, foreground=COL_KEY,
                        font=self.fonts["monob"])
        style.configure("Head.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["smallb"])
        style.configure("Muted.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=self.fonts["tiny"])
        style.configure("Under.TLabel", background=COL_BG,
                        foreground=COL_TRAIN, font=self.fonts["smallb"])
        style.configure("Over.TLabel", background=COL_BG, foreground=COL_TEST,
                        font=self.fonts["smallb"])
        style.configure("Good.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=self.fonts["midb"])
        style.configure("Bad.TLabel", background=COL_BG, foreground=COL_BAD_FG,
                        font=self.fonts["smallb"])
        style.configure("Go.TButton", font=self.fonts["smallb"])
        style.map("Go.TButton", foreground=[("active", COL_HEAD)])
        style.configure("GoodEntry.TEntry", fieldbackground=COL_PAPER)
        style.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG,
                        foreground=COL_BAD_FG)

    def _reg(self, widget, key, attr="text", **kwargs):
        self._text_refs.append((widget, key, attr, kwargs))
        widget.configure(**{attr: T(key, **kwargs)})

    def _box(self, parent, title_key):
        box = ttk.LabelFrame(parent, text=T(title_key), padding=(9, 5))
        box.pack(fill="x", pady=(0, 7))
        self._text_refs.append((box, title_key, "text", {}))
        return box

    def _label(self, parent, key, font=None, style=None, **kwargs):
        widget = ttk.Label(parent, font=font, style=style, **kwargs)
        self._reg(widget, key)
        return widget

    # --- сборка -------------------------------------------------------
    def _build(self):
        header = ttk.Frame(self, padding=(12, 8, 12, 4))
        header.grid(row=0, column=0, columnspan=3, sticky="we")
        self._label(header, "app_title", style="Title.TLabel").pack(anchor="w")
        self._label(header, "subtitle", style="Sub.TLabel", wraplength=1120,
                    justify="left").pack(anchor="w")

        body = ttk.Frame(self, padding=(10, 4, 10, 4))
        body.grid(row=1, column=0, columnspan=3, sticky="nsew")
        # Нижняя строка обязана оставаться внизу: без веса средняя строка
        # растягивается по содержимому и панель уезжает за окно.
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)
        self.columnconfigure(1, weight=1)
        body.columnconfigure(0, minsize=336, weight=0)
        body.columnconfigure(1, minsize=420, weight=1)
        body.columnconfigure(2, minsize=436, weight=0)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)
        self._build_bottom()

    def _build_left(self, parent):
        scroll = ScrollFrame(parent)
        scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left = scroll.inner

        box = self._box(left, "box_dataset")
        self.combo_shape = ttk.Combobox(box, state="readonly", width=25)
        self.combo_shape.pack(fill="x")
        self.combo_shape.bind("<<ComboboxSelected>>", self._on_shape)
        self.lbl_ds_desc = ttk.Label(box, wraplength=296, justify="left",
                                     font=self.fonts["small"])
        self.lbl_ds_desc.pack(anchor="w", pady=(5, 0))
        self.lbl_ds_formula = ttk.Label(box, style="Key.TLabel",
                                        wraplength=296, justify="left")
        self.lbl_ds_formula.pack(anchor="w", pady=(4, 0))
        self._label(box, "ds_note", style="Muted.TLabel", wraplength=296,
                    justify="left").pack(anchor="w", pady=(3, 0))

        box = self._box(left, "box_setup")
        self.controls = {}
        self.controls["degree"] = SliderControl(
            box, self, "lbl_degree", DEG_MIN, DEG_MAX, DEFAULT_DEGREE,
            1, 1, integer=True, state_attr="degree")
        self.controls["sigma"] = SliderControl(
            box, self, "lbl_sigma", SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA,
            SIGMA_STEP, SIGMA_STEP, state_attr="sigma")
        self.controls["share"] = SliderControl(
            box, self, "lbl_share", SHARE_MIN, SHARE_MAX, DEFAULT_SHARE,
            SHARE_STEP, SHARE_STEP, fmt_fun=fmt_share, state_attr="share",
            nudge_text="5%")
        for control in self.controls.values():
            control.pack(fill="x", pady=(0, 6))
        self._label(box, "hint_degree", style="Muted.TLabel", wraplength=296,
                    justify="left").pack(anchor="w")
        self.lbl_hint_share = ttk.Label(box, wraplength=296, justify="left",
                                        style="Muted.TLabel")
        self.lbl_hint_share.pack(anchor="w", pady=(3, 0))
        self.btn_reset = ttk.Button(box, text=T("btn_reset"),
                                    command=self.reset_all)
        self._reg(self.btn_reset, "btn_reset")
        self.btn_reset.pack(anchor="w", pady=(5, 2))
        self._label(box, "hint_step", style="Muted.TLabel", wraplength=296,
                    justify="left").pack(anchor="w")

        box = self._box(left, "box_split")
        self.lbl_split_train = ttk.Label(box, font=self.fonts["small"])
        self.lbl_split_train.pack(anchor="w")
        self.lbl_split_test = ttk.Label(box, font=self.fonts["small"])
        self.lbl_split_test.pack(anchor="w", pady=(2, 0))
        self._label(box, "split_note", wraplength=296, justify="left",
                    font=self.fonts["tiny"]).pack(anchor="w", pady=(5, 0))
        self.lbl_split_coef = ttk.Label(box, wraplength=296, justify="left",
                                        font=self.fonts["tiny"])
        self.lbl_split_coef.pack(anchor="w", pady=(3, 0))
        self.lbl_split_small = ttk.Label(box, wraplength=296, justify="left",
                                         font=self.fonts["tiny"],
                                         foreground=COL_TEST)
        self.lbl_split_small.pack(anchor="w", pady=(3, 0))

        box = self._box(left, "box_watch")
        self.lbl_watch = ttk.Label(box, wraplength=296, justify="left",
                                   font=self.fonts["small"])
        self.lbl_watch.pack(anchor="w")
        self.lbl_watch_curve = ttk.Label(box, wraplength=296, justify="left",
                                         font=self.fonts["tiny"],
                                         foreground=COL_TEST)
        self.lbl_watch_curve.pack(anchor="w", pady=(4, 0))

        box = self._box(left, "box_legend")
        for key in ("legend_train", "legend_test", "legend_fit",
                    "legend_truth"):
            self._label(box, key, font=self.fonts["tiny"], wraplength=296,
                        justify="left").pack(anchor="w")

        box = self._box(left, "box_language")
        for code, key in (("ru", "lang_ru"), ("en", "lang_en")):
            button = ttk.Radiobutton(box, text=T(key), value=code,
                                     variable=self.var_lang,
                                     command=self.switch_language)
            button.pack(anchor="w")
            self._reg(button, key)

    def _build_center(self, parent):
        center = ttk.Frame(parent)
        center.grid(row=0, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.plot = MainPlot(center, self)
        self.plot.grid(row=0, column=0, sticky="nsew")

    def _build_right(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.grid(row=0, column=2, sticky="nsew")
        # Каждая вкладка прокручивается: содержимому вкладки нужно больше
        # высоты, чем даёт окно, а обрезать подписи приложение не имеет права.
        self.tab_model = ScrollFrame(self.nb)
        self.tab_complexity = ScrollFrame(self.nb)
        self.tab_exp = ScrollFrame(self.nb)
        self.tab_help = ScrollFrame(self.nb)
        self.body_model = self.tab_model.inner
        self.body_complexity = self.tab_complexity.inner
        self.body_exp = self.tab_exp.inner
        self.body_help = self.tab_help.inner
        for body in (self.body_model, self.body_complexity, self.body_exp,
                     self.body_help):
            body.configure(padding=8)
        self._tabs = ((self.tab_model, "tab_model"),
                      (self.tab_complexity, "tab_complexity"),
                      (self.tab_exp, "tab_exp"),
                      (self.tab_help, "tab_help"))
        for frame, key in self._tabs:
            self.nb.add(frame, text=T(key))
        self._build_tab_model()
        self._build_tab_complexity()
        self._build_tab_exp()
        self._build_tab_help()

    def _build_tab_model(self):
        tab = self.body_model
        box = self._box(tab, "mb_box_state")
        self.lbl_line_degree = ttk.Label(box, wraplength=336, justify="left",
                                         font=self.fonts["smallb"])
        self.lbl_line_degree.pack(anchor="w")
        self.lbl_line_split = ttk.Label(box, wraplength=336, justify="left",
                                         font=self.fonts["small"])
        self.lbl_line_split.pack(anchor="w", pady=(2, 0))

        box = self._box(tab, "mb_box_verdict")
        self.lbl_verdict = ttk.Label(box, style="Good.TLabel")
        self.lbl_verdict.pack(anchor="w")
        self.lbl_expl = ttk.Label(box, wraplength=336, justify="left",
                                  font=self.fonts["small"])
        self.lbl_expl.pack(anchor="w", pady=(4, 0))
        self.lbl_extra = ttk.Label(box, wraplength=336, justify="left",
                                   font=self.fonts["tiny"],
                                   foreground=COL_DIM)
        self.lbl_extra.pack(anchor="w", pady=(4, 0))

        box = self._box(tab, "mb_box_errors")
        self._label(box, "mb_mse_formula", style="Key.TLabel").pack(anchor="w")
        self.lbl_train_mse = ttk.Label(box, font=self.fonts["monob"],
                                       foreground=COL_TRAIN)
        self.lbl_train_mse.pack(anchor="w", pady=(4, 0))
        self.lbl_test_mse = ttk.Label(box, font=self.fonts["monob"],
                                      foreground=COL_TEST)
        self.lbl_test_mse.pack(anchor="w", pady=(2, 0))
        self.lbl_sum_train = ttk.Label(box, wraplength=336, justify="left",
                                       font=self.fonts["mono"])
        self.lbl_sum_train.pack(anchor="w", pady=(4, 0))
        self.lbl_sum_test = ttk.Label(box, wraplength=336, justify="left",
                                      font=self.fonts["mono"])
        self.lbl_sum_test.pack(anchor="w")
        self.lbl_ratio = ttk.Label(box, font=self.fonts["small"])
        self.lbl_ratio.pack(anchor="w", pady=(4, 0))
        self.bars = ErrorBars(box, self)
        self.bars.pack(fill="x", pady=(6, 0))
        self._label(box, "bars_note", style="Muted.TLabel", wraplength=336,
                    justify="left").pack(anchor="w", pady=(2, 0))

        box = self._box(tab, "mb_box_math")
        self._label(box, "mb_general", style="Key.TLabel", wraplength=336,
                    justify="left").pack(anchor="w")
        self.lbl_here = ttk.Label(box, wraplength=336, justify="left",
                                  font=self.fonts["small"])
        self.lbl_here.pack(anchor="w", pady=(3, 0))
        self.lbl_fitted = ttk.Label(box, wraplength=336, justify="left",
                                    font=self.fonts["monob"],
                                    foreground=COL_FIT)
        self.lbl_fitted.pack(anchor="w", pady=(5, 0))
        self._label(box, "mb_coeff_head", style="Head.TLabel").pack(anchor="w",
                                                                    pady=(4, 0))
        self.txt_coeffs = self._make_text(box, 6)

    def _build_tab_complexity(self):
        tab = self.body_complexity
        box = self._box(tab, "cx_box_chart")
        box.pack(fill="x")
        self.cx_plot = ComplexityPlot(box, self)
        self.cx_plot.pack(fill="x")
        self._label(box, "cx_axis_note", wraplength=336, justify="left",
                    font=self.fonts["tiny"]).pack(anchor="w", pady=(4, 0))
        self._label(box, "cx_marks_note", wraplength=336, justify="left",
                    font=self.fonts["tiny"]).pack(anchor="w", pady=(3, 0))
        self.lbl_cx_best = ttk.Label(box, wraplength=336, justify="left",
                                     font=self.fonts["small"])
        self.lbl_cx_best.pack(anchor="w", pady=(4, 0))
        self.lbl_cx_lowest = ttk.Label(box, wraplength=336, justify="left",
                                       font=self.fonts["small"])
        self.lbl_cx_lowest.pack(anchor="w", pady=(2, 0))
        self.lbl_cx_rough = ttk.Label(box, wraplength=336, justify="left",
                                      font=self.fonts["tiny"],
                                      foreground=COL_DIM)
        self.lbl_cx_rough.pack(anchor="w", pady=(3, 0))
        self._label(box, "cx_reuse_note", wraplength=336, justify="left",
                    font=self.fonts["tiny"],
                    foreground=COL_DIM).pack(anchor="w", pady=(3, 0))

        box = self._box(tab, "cx_box_table")
        self.tree_cx = self._make_table(box, TABLE_KEYS, TABLE_WIDTHS)
        self._label(box, "cx_note", wraplength=336, justify="left",
                    font=self.fonts["tiny"]).pack(anchor="w", pady=(4, 0))

        # Правило вывода и интуиция о смещении/разбросе стоят здесь, а не на
        # вкладке модели: они объясняют именно ход двух кривых.
        box = self._box(tab, "mb_box_rule")
        self.lbl_rule = ttk.Label(box, wraplength=336, justify="left",
                                  font=self.fonts["tiny"])
        self.lbl_rule.pack(anchor="w")

        box = self._box(tab, "mb_box_bias")
        self.lbl_bias_under = ttk.Label(box, wraplength=336, justify="left",
                                        font=self.fonts["small"])
        self.lbl_bias_under.pack(anchor="w")
        self.lbl_bias_over = ttk.Label(box, wraplength=336, justify="left",
                                       font=self.fonts["small"])
        self.lbl_bias_over.pack(anchor="w", pady=(4, 0))

    def _make_table(self, parent, keys, widths):
        frame = ttk.Frame(parent)
        frame.pack(fill="x")
        tree = ttk.Treeview(frame, columns=keys, show="headings", height=9)
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        for key, width in zip(keys, widths):
            tree.heading(key, text=T(key))
            tree.column(key, width=width, anchor="center", stretch=False)
        tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self._table_heads.append(tree)
        return tree

    def _build_tab_exp(self):
        tab = self.body_exp
        box = self._box(tab, "exp_box_list")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.exp_list = tk.Listbox(row, height=8, width=42,
                                   exportselection=False, activestyle="none",
                                   background=COL_PAPER, foreground=COL_TEXT,
                                   selectbackground="#d8e4ee",
                                   selectforeground=COL_TEXT, relief="solid",
                                   borderwidth=1, font=self.fonts["small"])
        self.exp_list.pack(side="left", fill="x", expand=True)
        bar = ttk.Scrollbar(row, orient="vertical",
                            command=self.exp_list.yview)
        bar.pack(side="right", fill="y")
        self.exp_list.configure(yscrollcommand=bar.set)
        self.exp_list.bind("<<ListboxSelect>>", self._on_exp_select)
        self.btn_run_exp = ttk.Button(box, text=T("btn_run_exp"),
                                      command=self.run_selected_experiment,
                                      style="Go.TButton")
        self._reg(self.btn_run_exp, "btn_run_exp")
        self.btn_run_exp.pack(anchor="w", pady=(6, 2))
        self._label(box, "exp_hint", style="Muted.TLabel", wraplength=336,
                    justify="left").pack(anchor="w")

        box = self._box(tab, "exp_box_goal")
        self.lbl_exp_goal = ttk.Label(box, wraplength=336, justify="left",
                                      font=self.fonts["small"])
        self.lbl_exp_goal.pack(anchor="w")

    def _build_tab_help(self):
        tab = self.body_help
        box = self._box(tab, "help_box")
        row = ttk.Frame(box)
        row.pack(fill="x")
        self.help_list = tk.Listbox(row, height=7, width=42,
                                    exportselection=False, activestyle="none",
                                    background=COL_PAPER, foreground=COL_TEXT,
                                    selectbackground="#d8e4ee",
                                    selectforeground=COL_TEXT, relief="solid",
                                    borderwidth=1, font=self.fonts["small"])
        self.help_list.pack(side="left", fill="x", expand=True)
        bar = ttk.Scrollbar(row, orient="vertical",
                            command=self.help_list.yview)
        bar.pack(side="right", fill="y")
        self.help_list.configure(yscrollcommand=bar.set)
        self.help_list.bind("<<ListboxSelect>>", self._on_help_select)
        nav = ttk.Frame(box)
        nav.pack(anchor="w", pady=(6, 0))
        prev = ttk.Button(nav, text="<", width=4,
                          command=lambda: self.help_move(-1))
        prev.pack(side="left")
        next_button = ttk.Button(nav, text=">", width=4,
                                 command=lambda: self.help_move(1))
        next_button.pack(side="left", padx=(4, 0))
        self.txt_help = self._make_text(tab, 16)

    def _make_text(self, parent, height):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        # width обязателен: без него tk.Text просит 80 символов, и правая
        # колонка перестаёт помещаться в узком окне.
        widget = tk.Text(frame, width=TEXT_WIDTH, height=height, wrap="word",
                         relief="flat", background=COL_PAPER,
                         foreground=COL_TEXT, borderwidth=0,
                         highlightthickness=1, highlightbackground=COL_FRAME,
                         font=self.fonts["small"], padx=8, pady=6)
        bar = ttk.Scrollbar(frame, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=bar.set, state="disabled")
        widget.tag_configure("h", font=self.fonts["smallb"],
                             foreground=COL_HEAD)
        widget.tag_configure("t", font=self.fonts["small"])
        widget.tag_configure("k", font=self.fonts["monob"], foreground=COL_KEY)
        widget.tag_configure("b", font=self.fonts["small"],
                             foreground=COL_BAD_FG)
        widget.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        return widget

    # --- нижняя строка ------------------------------------------------
    def _build_bottom(self):
        bar = ttk.Frame(self, padding=(12, 3, 16, 3))
        bar.grid(row=2, column=0, columnspan=3, sticky="we")
        status = ttk.Label(bar, textvariable=self.var_status, anchor="w",
                           font=self.fonts["small"], foreground="#3a3a35")
        status.pack(side="left", fill="x", expand=True)
        # Имя автора постоянно, внизу справа: это подпись, а не учебный текст.
        credit = ttk.Label(bar, text="SteveWorks001",
                           font=self.fonts["smallb"], foreground=COL_DIM)
        credit.pack(side="right", padx=(10, 0))
        credit.tkraise()

    # --- служебное ----------------------------------------------------
    def set_status(self, key, **kwargs):
        self.var_status.set(T(key, **kwargs))

    def _nofit_text(self):
        """Объяснение отказа: две причины различаются по-русски и по-английски."""
        current = self.result["current"]
        reason = current.get("reason") or "unstable"
        return T("expl_nofit_" + reason, d=self.degree,
                 k=self.degree + 1, n=len(self.result["train_pts"]),
                 need=guard_min_points(self.degree))

    def announce_failure(self, current):
        reason = (current or {}).get("reason") or "unstable"
        self.set_status("status_nofit_" + reason, d=self.degree,
                        n=len(self.result["train_pts"]),
                        k=self.degree + 1,
                        need=guard_min_points(self.degree))

    def shape_name(self, shape_key=None):
        return T("ds_" + (shape_key or self.shape_key) + "_name")

    def announce_params(self):
        """Строка состояния после изменения параметров: числа и итог."""
        current = self.result.get("current") if self.result else None
        if current is None or not current["ok"]:
            self.announce_failure(current)
            return
        self.set_status("status_params", d=self.degree, s=fmt(self.sigma),
                        p=fmt_share(self.share),
                        tm=fmt_error(current["train_mse"]),
                        em=fmt_error(current["test_mse"]))

    def current_state(self):
        """Состояние для проверок: смена языка ничего в нём не трогает."""
        current = self.result.get("current") if self.result else None
        return {
            "lang": get_language(),
            "shape": self.shape_key,
            "degree": self.degree,
            "sigma": self.sigma,
            "share": self.share,
            "n_train": len(self.result["train_pts"]),
            "n_test": len(self.result["test_pts"]),
            "best_degree": self.result.get("best_degree"),
            "verdict": self.result["verdict"]["key"],
            "train_mse": None if current is None else current["train_mse"],
            "test_mse": None if current is None else current["test_mse"],
        }

    # --- пересчёт ------------------------------------------------------
    def refresh(self):
        points, clean = build_dataset(self.shape_key, self.sigma)
        self.points = points
        self.clean = clean
        self.result = analyse(points, self.share, self.degree)
        self.y_bounds = self._compute_bounds()
        self._refresh_left()
        self._refresh_model_tab()
        self._refresh_complexity_tab()
        self.plot.redraw()
        self.bars.redraw()
        self.cx_plot.redraw()

    def _compute_bounds(self):
        values = [y for x, y in self.points] + [y for x, y in self.clean]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            return (-1.0, 1.0)
        low, high = min(values), max(values)
        pad = max(0.12 * (high - low), 0.1)
        return (low - pad, high + pad)

    def _refresh_left(self):
        result = self.result
        self.lbl_ds_desc.configure(text=T("ds_" + self.shape_key + "_desc"))
        self.lbl_ds_formula.configure(
            text=T("ds_formula", f=SHAPE_BY_KEY[self.shape_key]["formula"]))
        self.lbl_split_train.configure(text=T("split_train",
                                              n=len(result["train_pts"])))
        self.lbl_split_test.configure(text=T("split_test",
                                             n=len(result["test_pts"])))
        self.lbl_hint_share.configure(
            text=T("hint_share", p=fmt_share(self.share),
                   n=len(result["train_pts"]), m=len(result["test_pts"])))
        self.lbl_split_coef.configure(
            text=T("split_coef", k=self.degree + 1, d=self.degree,
                   n=len(result["train_pts"]),
                   need=guard_min_points(self.degree)))
        n_test = len(result["test_pts"])
        if n_test <= SMALL_TEST_WARN:
            self.lbl_split_small.configure(
                text=T("split_small", m=n_test, w=fmt(100.0 / n_test, 1)))
        else:
            self.lbl_split_small.configure(text="")
        current = result["current"]
        if not current["ok"]:
            self.lbl_watch.configure(text=self._nofit_text())
            self.lbl_watch_curve.configure(text="")
        else:
            self.lbl_watch.configure(text=T("watch_line", d=self.degree,
                                            tm=fmt_error(current["train_mse"]),
                                            em=fmt_error(current["test_mse"]),
                                            v=T("verdict_" +
                                                result["verdict"]["key"])))
            coeffs = current["coeffs"]
            outside = curve_outside(coeffs, self.y_bounds[0],
                                    self.y_bounds[1], X_MIN, X_MAX)
            if outside:
                peak = 0.0
                for i in range(201):
                    x = X_MIN + (X_MAX - X_MIN) * i / 200.0
                    value = abs(horner(coeffs, x))
                    if math.isfinite(value):
                        peak = max(peak, value)
                self.lbl_watch_curve.configure(
                    text=T("mb_curve_out", v=fmt_error(peak)))
            else:
                self.lbl_watch_curve.configure(text="")

    def _refresh_model_tab(self):
        result = self.result
        current = result["current"]
        verdict = result["verdict"]
        n_coeffs = self.degree + 1
        self.lbl_line_degree.configure(
            text=T("mb_line_degree", d=self.degree, k=n_coeffs))
        self.lbl_line_split.configure(
            text=T("mb_line_split", n=len(result["train_pts"]),
                   m=len(result["test_pts"])))

        style = {"under": "Under.TLabel", "over": "Over.TLabel",
                 "good": "Good.TLabel", "trans": "Bad.TLabel",
                 "nofit": "Bad.TLabel"}.get(verdict["key"], "Head.TLabel")
        self.lbl_verdict.configure(style=style,
                                   text=T("verdict_" + verdict["key"]))
        self.lbl_expl.configure(text=self._verdict_text(verdict))

        extras = []
        if current is not None and current["ok"]:
            if current["test_mse"] < current["train_mse"]:
                extras.append(T("mb_test_lower"))
            if self.lbl_watch_curve.cget("text"):
                extras.append(self.lbl_watch_curve.cget("text"))
        self.lbl_extra.configure(text="\n".join(extras))

        if not current["ok"]:
            dash = fmt_error(None)
            self.lbl_train_mse.configure(text=T("mb_train_mse", v=dash))
            self.lbl_test_mse.configure(text=T("mb_test_mse", v=dash))
            for widget in (self.lbl_sum_train, self.lbl_sum_test,
                           self.lbl_ratio):
                widget.configure(text="")
            self.lbl_here.configure(text=T("mb_here", d=self.degree,
                                           k=n_coeffs, ds=sub(self.degree)))
            self.lbl_fitted.configure(text="ŷ = —")
            fill_text(self.txt_coeffs, [("t", self._nofit_text())])
        else:
            coeffs = current["coeffs"]
            train_pts = [(x, y) for i, (x, y) in result["train"]]
            test_pts = [(x, y) for i, (x, y) in result["test"]]
            sse_train, sse_test = sse_pair(coeffs, train_pts, test_pts)
            self.lbl_train_mse.configure(
                text=T("mb_train_mse", v=fmt_error(current["train_mse"])))
            self.lbl_test_mse.configure(
                text=T("mb_test_mse", v=fmt_error(current["test_mse"])))
            self.lbl_sum_train.configure(
                text=T("mb_sum_train", s=fmt_error(sse_train),
                       n=len(train_pts), v=fmt_error(current["train_mse"])))
            self.lbl_sum_test.configure(
                text=T("mb_sum_test", s=fmt_error(sse_test),
                       m=len(test_pts), v=fmt_error(current["test_mse"])))
            ratio = current["test_mse"] / max(current["train_mse"], 1e-15)
            self.lbl_ratio.configure(text=T("mb_ratio", r=fmt(ratio, 2)))
            self.lbl_here.configure(text=T("mb_here", d=self.degree,
                                           k=n_coeffs, ds=sub(self.degree)))
            self.lbl_fitted.configure(text=T("mb_fitted") + "  " +
                                      polynomial_text(coeffs))
            fill_text(self.txt_coeffs,
                      [("h", T("mb_coeff_head")),
                       ("t", T("mb_learn_note"))] +
                      [("k", "%s%s = %s" % ("b", sub(i), fmt_coef(c)))
                       for i, c in enumerate(coeffs)])
        self._refresh_rule_box()
        self._refresh_bias_box()

    def _verdict_text(self, verdict):
        if verdict["key"] == "nofit":
            return self._nofit_text()
        best = self.result.get("best_test")
        close_term = self.result.get("tol_close_term")
        floor_term = self.result.get("tol_floor_term")
        tol_key = ("expl_good_tol_bt"
                   if floor_term is None or close_term >= floor_term
                   else "expl_good_tol_sk")
        return T("expl_" + verdict["key"],
                 tm=fmt_error(verdict["train_mse"]),
                 em=fmt_error(verdict["test_mse"]),
                 bt=fmt_error(best),
                 tol=fmt_error(verdict["tol"]),
                 sk=fmt_error(self.result.get("base")),
                 c1=fmt(TH_CLOSE, 1), c2=fmt(TH_FLOOR, 2),
                 dt=T(tol_key),
                 d=verdict["degree"],
                 bd=self.result.get("best_degree") or "—")

    def _refresh_rule_box(self):
        result = self.result
        lines = [T("rule_intro"),
                 T("rule_1", lo=DEG_MIN, hi=DEG_MAX,
                   bt=fmt_error(result.get("best_test"))),
                 T("rule_2", c1=fmt(TH_CLOSE, 1), c2=fmt(TH_FLOOR, 2),
                   tol=fmt_error(result.get("tol"))),
                 T("rule_3"), T("rule_4", c3=fmt(TH_GAP, 1)), T("rule_5"),
                 T("rule_6"),
                 T("rule_base", v=fmt_error(result.get("base")))]
        if result.get("best_degree") is not None:
            lines.append(T("rule_best", d=result["best_degree"]))
        self.lbl_rule.configure(text="\n".join(lines))

    def _refresh_bias_box(self):
        key = self.result["verdict"]["key"]
        self.lbl_bias_under.configure(
            text=T("bias_under"),
            foreground=COL_HEAD if key == "under" else COL_DIM)
        self.lbl_bias_over.configure(
            text=T("bias_over"),
            foreground=COL_HEAD if key == "over" else COL_DIM)

    def _refresh_complexity_tab(self):
        result = self.result
        rows = result["rows"]
        tree = self.tree_cx
        tree.delete(*tree.get_children())
        for index, row in enumerate(rows):
            degree = DEG_MIN + index
            if not row["ok"]:
                tree.insert("", "end", values=(
                    degree, T("vd_nofit"), T("vd_nofit"),
                    T("vd_reason_" + row["reason"])))
                continue
            verdict = interpret(degree, row, result["best_test"],
                                result["best_degree"], result["base"],
                                result["tol"])
            tree.insert("", "end", values=(
                degree, fmt_error(row["train_mse"]),
                fmt_error(row["test_mse"]),
                T("vd_" + verdict["key"])))
        if result.get("best_degree") is not None:
            self.lbl_cx_best.configure(
                text=T("cx_best", d=result["best_degree"]))
        else:
            self.lbl_cx_best.configure(text="—")
        lowest = None
        for row in rows:
            if row["ok"] and (lowest is None or
                              row["train_mse"] < lowest["train_mse"]):
                lowest = row
        if lowest is not None:
            self.lbl_cx_lowest.configure(
                text=T("cx_lowest_train", d=lowest["degree"]))
        self.lbl_cx_rough.configure(
            text=T("cx_rough", n=len(result["test_pts"])))

    # --- выбор настроек ------------------------------------------------
    def _on_shape(self, _event=None):
        index = self.combo_shape.current()
        if index < 0:
            return
        self.shape_key = SHAPES[index]["key"]
        self.experiment_key = None
        self._sync_exp_list()
        self.refresh()
        self.set_status("status_start", name=self.shape_name(), d=self.degree,
                        s=fmt(self.sigma), p=fmt_share(self.share),
                        n=len(self.result["train_pts"]),
                        m=len(self.result["test_pts"]))

    def reset_all(self):
        self.shape_key = DEFAULT_SHAPE
        self.controls["degree"].set_value(DEFAULT_DEGREE)
        self.controls["sigma"].set_value(DEFAULT_SIGMA)
        self.controls["share"].set_value(DEFAULT_SHARE)
        self.experiment_key = None
        self._sync_exp_list()
        self.refresh()
        self.set_status("status_reset", name=self.shape_name(),
                        d=self.degree)

    def _current_tab_key(self):
        try:
            return self.nb.tab(self.nb.select(), "text")
        except (tk.TclError, KeyError):
            return ""

    # --- опыты ----------------------------------------------------------
    def _on_exp_select(self, _event=None):
        self._show_exp_goal()

    def run_selected_experiment(self):
        selection = self.exp_list.curselection()
        if not selection:
            self.set_status("status_no_exp")
            return
        self.run_experiment(EXPERIMENTS[selection[0]]["key"])

    def run_experiment(self, key):
        entry = EXPERIMENT_BY_KEY[key]
        self.shape_key = entry["shape"]
        self.controls["degree"].set_value(entry["degree"])
        self.controls["sigma"].set_value(entry["sigma"])
        self.controls["share"].set_value(entry["share"])
        self.experiment_key = key
        self._fill_shape_combo()
        self._sync_exp_list()
        self.refresh()
        self._show_exp_goal()
        current = self.result["current"]
        if not current["ok"]:
            self.announce_failure(current)
        else:
            self.set_status("status_exp",
                            name=EXP_TEXTS[get_language()][key][0],
                            tm=fmt_error(current["train_mse"]),
                            em=fmt_error(current["test_mse"]))

    def _sync_exp_list(self, key=None):
        target = key or self.experiment_key
        if target is None:
            return
        for index, entry in enumerate(EXPERIMENTS):
            if entry["key"] == target:
                self.exp_list.selection_clear(0, "end")
                self.exp_list.selection_set(index)
                self.exp_list.activate(index)
                break

    def _show_exp_goal(self):
        selection = self.exp_list.curselection()
        if not selection:
            self.lbl_exp_goal.configure(text=T("exp_none"))
            return
        entry = EXPERIMENTS[selection[0]]
        title, goal = EXP_TEXTS[get_language()][entry["key"]]
        self.lbl_exp_goal.configure(text=title + "\n\n" + goal)

    # --- справка -------------------------------------------------------
    def _on_help_select(self, _event=None):
        selection = self.help_list.curselection()
        if selection:
            self.help_key = HELP_KEYS[selection[0]]
        self._show_help()

    def help_move(self, step):
        try:
            index = HELP_KEYS.index(self.help_key)
        except ValueError:
            index = 0
        index = clamp(index + step, 0, len(HELP_KEYS) - 1)
        self.help_key = HELP_KEYS[index]
        self._sync_help_list()
        self._show_help()

    def _sync_help_list(self):
        self.help_list.selection_clear(0, "end")
        self.help_list.selection_set(HELP_KEYS.index(self.help_key))
        self.help_list.activate(HELP_KEYS.index(self.help_key))

    def _show_help(self):
        title, lines = HELP[get_language()][self.help_key]
        rows = [("h", title)]
        for line in lines:
            if line.startswith("»"):
                rows.append(("k", line[1:]))
            elif line.startswith("!"):
                rows.append(("b", line[1:]))
            else:
                rows.append(("t", line))
        fill_text(self.txt_help, rows)

    # --- локализация ----------------------------------------------------
    def switch_language(self):
        code = self.var_lang.get()
        if code == get_language():
            return
        set_language(code)
        self.title(T("app_title"))
        for widget, key, attr, kwargs in self._text_refs:
            try:
                widget.configure(**{attr: T(key, **kwargs)})
            except tk.TclError:
                pass
        for frame, key in self._tabs:
            self.nb.tab(frame, text=T(key))
        for tree in self._table_heads:
            self._retable(tree)
        for control in self.controls.values():
            control.retranslate()
        self._fill_shape_combo()
        self._fill_exp_list()
        self._fill_help_list()
        self.refresh()
        self._show_help()
        self._show_exp_goal()
        name = T("lang_ru") if code == "ru" else T("lang_en")
        self.set_status("status_lang", name=name)

    def _retable(self, tree):
        for column in tree["columns"]:
            tree.heading(column, text=T(column))

    def _fill_shape_combo(self):
        values = [self.shape_name(entry["key"]) for entry in SHAPES]
        keys = [entry["key"] for entry in SHAPES]
        self.combo_shape.configure(values=values)
        self.combo_shape.current(keys.index(self.shape_key))

    def _fill_exp_list(self):
        keep = self.exp_list.curselection()
        self.exp_list.delete(0, "end")
        for entry in EXPERIMENTS:
            self.exp_list.insert("end",
                                 EXP_TEXTS[get_language()][entry["key"]][0])
        if keep:
            self.exp_list.selection_set(keep[0])
            self.exp_list.activate(keep[0])

    def _fill_help_list(self):
        self.help_list.delete(0, "end")
        for key in HELP_KEYS:
            self.help_list.insert("end", HELP[get_language()][key][0])
        self._sync_help_list()


# ----------------------------------------------------------------------
# Раздел 10. Точка входа
# ----------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()




