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