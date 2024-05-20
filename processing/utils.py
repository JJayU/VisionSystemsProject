import numpy as np
import cv2
from copy import copy

def perform_processing(image: np.ndarray) -> str:
    # print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    tablica = None

    imscaled = cv2.resize(image, (800, 600))

    # Wstepne przetworzenie obrazu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 1)
    edged = cv2.Canny(blurred, 20, 250)
    edged = cv2.dilate(edged, np.ones((3,3)), iterations=2)

    # cv2.namedWindow('filtr', cv2.WINDOW_NORMAL)
    # cv2.imshow('filtr', edged)
    # cv2.resizeWindow('filtr', 800, 600)

    # Wykrywanie konturow
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Filtracja konturów
    for contour in sorted_contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 1 < aspect_ratio < 10 and np.all(approx != 0) and 10**8 > cv2.contourArea(contour) > 10**5:
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 5)
                rect = np.zeros((4, 2), dtype="float32")

                pts = approx.reshape(4,2)
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                dst = np.array([
                    [0, 0],
                    [520, 0],
                    [520, 114],
                    [0, 114]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)

                tablica = cv2.warpPerspective(image, M, (520, 114))

                break

    if tablica is not None:
        cv2.imshow('Warped Image', tablica)
        result = ''

        # Wyświetlanie wyniku
        cv2.namedWindow('Detected Plate', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected Plate', image)
        cv2.resizeWindow('Detected Plate', 800, 600)

        print('Odczytano: ' + result)
    else:
        print('Nie znaleziono tablicy!')

    cv2.waitKey(0)
    return 'PO12345'