import cv2

################################################################################
#                                manual aligner                                #
################################################################################

angle = 90    # angulo al que se ajustan las lineas marcadas

wa_kwargs = {    # warpAffine kwargs (para aplicar matriz de rotacion)
    "dsize": (2700, 2100),                # tamaño de imagenes de salida
    "borderMode": cv2.BORDER_CONSTANT,
    "borderValue": 255
}

global_center = (wa_kwargs["dsize"][0])//2, (wa_kwargs["dsize"][1])//2 + 200

circle_kwargs = {    # para marcador visual
    "radius": 10,
    "color": (0, 255, 0),
    "thickness": cv2.FILLED
}

line_kwargs = {
    "color": (0, 255, 0),
    "thickness": 5,
    "lineType": cv2.LINE_AA
}
################################################################################


################################################################################
#          Auto aligner  (iterative_average_alignment with logpolar)           #
################################################################################

wp_kwargs = {
    "dsize": (0, 0),
    "flags": cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS
}

iterative_average_alignment_tol = 1e-4

min_angle_response = 0.06
min_desp_response = 0.06
################################################################################


################################################################################
#                                clean images                                  # 
################################################################################

# para eliminar areas pequeñas
cc_kwargs = {    # connectedComponentsWithStats
    "connectivity": 4    # 4 o 8
}

min_area=800

clahe_kwargs = {    # para aplicar contraste
    "clipLimit": 7.0,
    "tileGridSize": (24, 24)
}

mb_kwargs = {    # para aplicar medianBlur
    "ksize": 7
}

thresh_kwargs = {    # para aplicar threshholding
    "thresh": 150,
    "maxval": 255,
    "type": cv2.THRESH_BINARY
}
################################################################################