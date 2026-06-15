import cv2

################################################################################
#                                manual aligner                                #
################################################################################

angle = 90    # angulo al que se ajustan las lineas marcadas

wa_kwargs = {    # warpAffine kwargs (para aplicar matriz de rotacion)
    "dsize": (400, 400),                   # tamaño de imagenes de salida
    "borderMode": cv2.BORDER_CONSTANT,
    "borderValue": 255
}

circle_kwargs = {    # para marcador visual
    "radius": 3,
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