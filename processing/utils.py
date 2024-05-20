import numpy as np
import cv2
from copy import copy
import pickle

def perform_processing(image: np.ndarray, path) -> str:
    # print(f'image.shape: {image.shape}')
    # TODO: add image processing here

    tablica = None
    result = ''

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
        tablica_rgb = tablica
        tablica = cv2.cvtColor(tablica, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Warped Image', tablica)
        result = ''

        # Wyświetlanie wyniku
        cv2.namedWindow('Detected Plate', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected Plate', image)
        cv2.resizeWindow('Detected Plate', 800, 600)

        _, tablica_filtr = cv2.threshold(tablica, 150, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        tablica_filtr = cv2.erode(tablica_filtr, np.ones((5,5)))
        tablica_filtr = cv2.dilate(tablica_filtr, np.ones((3,3)))

        tablica_filtr = tablica_filtr[10:105, :]

        first_letter = 0
        x_scan = np.zeros((tablica_filtr.shape[1]))

        for i in range(0, tablica_filtr.shape[1]):
            for j in range(0, tablica_filtr.shape[0]):
                if tablica_filtr[j, i] == 255:
                    x_scan[i] = True
                    break
                else:
                    x_scan[i] = False

        znaki = []
        prev = 0
        for i, val in enumerate(x_scan):
            if val == 1 and prev == 0:
                if i-5 < 0:
                    znaki.append((0,0))
                else:
                    znaki.append((i-5, 0))
            if val == 0 and prev == 1:
                znaki[-1] = (znaki[-1][0], i+5)
            prev = val

        #TEMP - przygotowanie zbioru treningowego
        for i in znaki:
            size = i[1] - i[0]

            cv2.line(tablica_rgb, (i[0], 0), (i[0], 114), (255, 0, 0), 1)
            cv2.line(tablica_rgb, (i[1], 0), (i[1], 114), (0, 255, 0), 1)

            litera = tablica_filtr[:, i[0]:i[1]]
            litera_tmp = np.zeros((95,80))
            if size < 80:
                x_position = (80 - size) // 2
                litera_tmp[:, x_position:x_position+size] = litera
            else:
                litera_tmp = litera[:, 0:80]

            cv2.imshow('litera', litera_tmp)

            with open('model.pkl', 'rb') as f:
                clf2 = pickle.load(f)

            odczyt = clf2.predict([litera_tmp.flatten()])[0]

            if odczyt != '_':
                result = result + odczyt

            # Zapis plikow do zbioru treningowego
            # cv2.waitKey(1)
            # znak = input('Podaj znak: ')
            # cv2.destroyWindow('litera')
            # name = 'train_letters/' + str(znak) + '_' + str(np.random.randint(0, 10000)) + '.png'
            # cv2.imwrite(name, litera_tmp)

        cv2.imshow('Tablica filtr', tablica_filtr)
        cv2.imshow('Tablica podzielona', tablica_rgb)

        dlugosc = len(result)
        dobre = 0
        for i in range(0, dlugosc):
            if result[i] == path.name[i]:
                dobre += 1

        procent = (dobre / dlugosc) * 100


        print('Odczytano: ' + result + ' / ' + path.name[:-4] + ' - ' + str(procent) + '%')

    else:
        print('Nie znaleziono tablicy!')

    cv2.waitKey(1)
    return result
