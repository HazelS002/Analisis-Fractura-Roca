import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from src.visualitation import show_images

def component_from_peak(peak, F_shifted, radius, image_size):
    """
    Reconstruye la imagen correspondiente a un pico de Fourier y su simétrico.

    Parámetros:
    - peak: tupla (peak_u, peak_v) en coordenadas centradas (sin shift).
    - F_shifted: espectro complejo ya centrado (np.fft.fftshift).
    - radius: radio del filtro circular alrededor del pico (en píxeles).
    - image_size: tupla (H, W) dimensiones de la imagen.

    Retorna:
    - comp: imagen real (array 2D) de la componente sinusoidal.
    """
    peak_u, peak_v = peak
    H, W = image_size
    center_y, center_x = H // 2, W // 2

    # Filtro binario (máscara)
    filt = np.zeros_like(F_shifted, dtype=bool)

    # Coordenadas en el espectro centrado
    x_peak, y_peak = center_x + peak_v, center_y + peak_u
    x_peak_sym, y_peak_sym = center_x - peak_v, center_y - peak_u

    # Rellenar círculos alrededor del pico y su simétrico
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                # Pico principal
                yy, xx = y_peak + dy, x_peak + dx
                if 0 <= yy < H and 0 <= xx < W:
                    filt[yy, xx] = True
                # Pico simétrico
                yy, xx = y_peak_sym + dy, x_peak_sym + dx
                if 0 <= yy < H and 0 <= xx < W:
                    filt[yy, xx] = True

    # Filtrar y reconstruir
    F_component = F_shifted * filt
    comp = np.fft.ifft2(np.fft.ifftshift(F_component)).real
    return comp

def reconstruct_from_components(components, dc_value=None):
    """
    Suma todas las componentes individuales (ondas planas) para obtener una imagen.

    Parámetros:
    - components: lista de arrays 2D, cada uno es una componente sinusoidal.
    - dc_value: valor escalar que se añade como componente de frecuencia cero (brillo medio).

    Retorna:
    - Suma de todas las componentes (y opcionalmente el DC).
    """
    reconstructed = np.sum(components, axis=0)
    if dc_value is not None:
        reconstructed = reconstructed + dc_value
    return reconstructed

def fft_analysis(image, use_log=True, threshold_rel=0.3, min_distance=5):
    """
    Analiza y visualiza la transformada de Fourier de una sola imagen.

    Parámetros:
    - image: array 2D (escala de grises).
    - use_log: si es True, muestra el espectro de magnitud en escala logarítmica.
    - threshold_rel: fracción del máximo (excluyendo DC) para considerar un pico.
    - min_distance: distancia mínima (en píxeles de frecuencia) entre picos.

    Retorna:
    - F_shifted: espectro centrado (complejo).
    - magnitude: magnitud del espectro centrado.
    - phase: fase del espectro centrado.
    """
    # calcular fft
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    magnitude, phase = np.abs(F_shifted), np.angle(F_shifted)

    # Visualización de magnitud (con o sin log)
    magnitude_display = np.log1p(magnitude) if use_log else magnitude

    H, W = image.shape
    center_y, center_x = H // 2, W // 2

    # detectar pichos
    mag_detection = magnitude.copy()
    mag_detection[center_y - 2 : center_y + 3, center_x - 2 : center_x + 3] = 0 # excluir centro

    max_val = np.max(mag_detection)
    threshold = threshold_rel * max_val

    # Máximos locales (ventana 3x3)
    local_max = (mag_detection == maximum_filter(mag_detection, size=3))
    peaks = np.argwhere(local_max & (mag_detection > threshold))

    # Convertir a coordenadas centradas (sin shift) y filtrar por distancia mínima
    peaks_original = [(p[0] - H // 2, p[1] - W // 2) for p in peaks]

    filtered_peaks = []
    for p in peaks_original:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_distance for q in filtered_peaks):
            filtered_peaks.append(p)


    components = [
        component_from_peak(p, F_shifted, radius=1, image_size=(H, W))
        for p in filtered_peaks
    ]

    # ------------------ SALIDA ORDENADA EN CONSOLA ------------------
    print("\n" + "=" * 70)
    print("RESULTADOS DEL ANÁLISIS DE FOURIER")
    print("=" * 70)

    print("\n[INFO DETECCIÓN]")
    print(f"\tMáximo del espectro (excluyendo DC) : {max_val:12.2f}")
    print(f"\tUmbral aplicado ({threshold_rel=:<5})   : {threshold:12.4f}")

    print("\n[PICOS DETECTADOS]")
    print(f"\tTotal: {len(filtered_peaks)}")
    if filtered_peaks:
        print("\t #      u         v       frecuencia física (ciclos/píxel)")
        print("\t---   -------    ------   -------------------------------")
        for i, (u, v) in enumerate(filtered_peaks, 1):
            freq_u = u / H
            freq_v = v / W
            print(f"\t{i:2d}     {u:6.2f}    {v:6.2f}     ({freq_u:.4f}, {freq_v:.4f})")
    else:
        print("\t(No se detectaron picos significativos)")

    print("\n[ESTADÍSTICAS DE LA FFT]")
    print(f"\tDimensiones de la imagen        : {H} x {W}")
    print(f"\tValor DC (frecuencia cero)      : {magnitude[center_y, center_x]:12.2f}")
    print(f"\tRango de magnitud               : [{magnitude.min():10.2f}, {magnitude.max():10.2f}]")
    print(f"\tRango de fase (radianes)        : [{phase.min():8.2f}, {phase.max():8.2f}]")
    print("=" * 70 + "\n")


    _, axes = plt.subplots(1, 4)
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Imagen original"); axes[0].axis('off')

    axes[1].imshow(magnitude_display, cmap='gray', origin='lower')
    axes[1].set_xlabel("Frecuencia u"); axes[1].set_ylabel("Frecuencia v")
    axes[1].set_title("Espectro de magnitud")

    # Marcar picos con círculos rojos
    for u, v in filtered_peaks:
        cx = v + W // 2
        cy = u + H // 2
        circle = plt.Circle((cx, cy), radius=3, color='red', fill=False, linewidth=1)
        axes[1].add_patch(circle)

    axes[2].imshow(phase, cmap='twilight', origin='lower', vmin=-np.pi, vmax=np.pi)
    axes[2].set_xlabel("Frecuencia u"); axes[2].set_ylabel("Frecuencia v")
    axes[2].set_title("Espectro de fase")

    reconstructed = reconstruct_from_components(components)
    axes[3].imshow(reconstructed, cmap='gray')
    axes[3].set_title("Reconstructed"); axes[3].axis('off')
    plt.show()

    show_images(components, [f"Componente {i+1}" for i in range(len(components))])

    return F_shifted, magnitude, phase


if __name__ == "__main__":
    from src.load_images import read_images
    from src import PROCESSED_IMAGES_DIR

    images, names = read_images(PROCESSED_IMAGES_DIR + "aligned-images/")

    for img in images[:10]:
        fft_analysis(img, use_log=True, threshold_rel=0.4, min_distance=1)