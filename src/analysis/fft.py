import numpy as np
from scipy.ndimage import maximum_filter
from src.visualitation import fft_visualitation


def component_from_pair(peak, F_shifted, radius, image_size):
    """
    Reconstruye UNA componente real (coseno) a partir de un par de picos conjugados.

    Parámetros:
    - peak: (u, v) en coordenadas centradas
    - F_shifted: FFT centrada
    - radius: radio del vecindario espectral
    - image_size: (H, W)

    Retorna:
    - componente real (onda plana)
    """
    u, v = peak
    H, W = image_size
    cy, cx = H // 2, W // 2

    filt = np.zeros_like(F_shifted, dtype=bool)

    # Coordenadas de ambos picos (conjugados)
    coords = [
        (cy + u, cx + v),
        (cy - u, cx - v)
    ]

    for (py, px) in coords:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy**2 + dx**2 <= radius**2:
                    yy, xx = py + dy, px + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        filt[yy, xx] = True

    F_component = F_shifted * filt

    comp = np.fft.ifft2(np.fft.ifftshift(F_component)).real
    return comp


def group_conjugate_peaks(peaks):
    """
    Agrupa picos conjugados y devuelve solo uno por pareja.

    Criterio: (u,v) y (-u,-v) representan la misma frecuencia.
    """
    unique = []
    visited = set()

    for u, v in peaks:
        if (u, v) in visited or (-u, -v) in visited:
            continue

        unique.append((u, v))
        visited.add((u, v))
        visited.add((-u, -v))

    return unique


def reconstruct_from_components(components, dc_value):
    """
    Reconstrucción completa: suma de componentes + DC (media).
    """
    reconstructed = np.sum(components, axis=0)
    return reconstructed + dc_value


def fft_analysis(image, use_log=True, threshold_rel=0.3, min_distance=5, radius=1):
    """
    Análisis de Fourier con interpretación físicamente consistente.
    """
    H, W = image.shape

    # FFT
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    magnitude, phase = np.abs(F_shifted), np.angle(F_shifted)
    dc_value = F[0, 0].real / (H * W)
    magnitude_display = np.log1p(magnitude) if use_log else magnitude

    cy, cx = H // 2, W // 2

    # Eliminar DC de detección
    mag_detection = magnitude.copy()
    mag_detection[cy-2:cy+3, cx-2:cx+3] = 0

    threshold = threshold_rel * np.max(mag_detection)

    # Máximos locales
    local_max = (mag_detection == maximum_filter(mag_detection, size=3))
    peaks = np.argwhere(local_max & (mag_detection > threshold))

    # Coordenadas centradas
    peaks = [(p[0] - cy, p[1] - cx) for p in peaks]

    # Filtrado por distancia
    filtered = []
    for p in peaks:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_distance for q in filtered):
            filtered.append(p)

    unique_peaks = group_conjugate_peaks(filtered)
    components = [ component_from_pair(p, F_shifted, radius, (H, W))\
                  for p in unique_peaks ]
    reconstructed = reconstruct_from_components(components, dc_value)

    # ------------------ DEBUG ------------------
    print("\n" + "=" * 70)
    print("ANALISIS DE FOURIER")
    print("=" * 70)

    print(f"\n\tTotal picos detectados     : {len(filtered)}")
    print(f"\tPicos únicos (sin duplicados) : {len(unique_peaks)}")

    print("\n\tFrecuencias (ciclos/píxel):")
    for i, (u, v) in enumerate(unique_peaks, 1):
        print(f"\t{i:2d}: ({u/H:.4f}, {v/W:.4f})")

    print(f"\n\tDC (media espacial): {dc_value:.4f}")
    print("=" * 70 + "\n")

    fft_visualitation(image, magnitude_display, phase, reconstructed,
                      unique_peaks, components)

    return F_shifted, magnitude, phase


if __name__ == "__main__":
    from src.load_images import read_images
    from src import PROCESSED_IMAGES_DIR
    from src.transform import clean_images
    from src.utils import get_stage_images

    images, _ = read_images(PROCESSED_IMAGES_DIR + "aligned-images/")
    images = get_stage_images(clean_images(images), "cleaned")

    for img in images[:5]:
        fft_analysis(img, threshold_rel=0.4, min_distance=2, radius=1)