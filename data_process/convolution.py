import cv2

def smoothing(images):
    return [ cv2.GaussianBlur(255-cv2.medianBlur(img, 3), (0, 0), sigmaX=60, sigmaY=60)\
            for img in images ]
    # return [ cv2.medianBlur(img, 21) for img in images ]

if __name__ == "__main__": pass