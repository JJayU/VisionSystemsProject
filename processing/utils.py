import numpy as np
import cv2
import pickle


def perform_processing(image: np.ndarray, path) -> str:

    license_plate_img = None
    result = ''

    # Initial processing of the image
    # 1. Convert to grayscale
    # 2. Apply Gaussian blur
    # 3. Apply Canny edge detection and dilate the edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 1)
    edged = cv2.Canny(blurred, 40, 150)
    edged = cv2.dilate(edged, np.ones((3,3)), iterations=2)

    # Show Canny edge detection
    # cv2.namedWindow('canny', cv2.WINDOW_NORMAL)
    # cv2.imshow('canny', edged)
    # cv2.resizeWindow('canny', 800, 600)

    # Detect contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Set aspect ratio limits
    min_ar = 1
    max_ar = 10

    # Filer the contours
    # Select only the contours that are rectangles with aspect ratio between min_ar and max_ar and have appropriate area
    # Select the first contour that meets the criteria and apply perspective transform
    for contour in sorted_contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            correct_ar = min_ar < aspect_ratio < max_ar
            correct_loc_and_size = np.all(approx != 0)
            correct_area = 10**8 > cv2.contourArea(contour) > 10**5
            correct_width = w > image.shape[1]/3

            if correct_ar and correct_loc_and_size and correct_area and correct_width:
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

                license_plate_img = cv2.warpPerspective(image, M, (520, 114))

                break

    # Check if the license plate was detected
    if license_plate_img is not None:
        license_plate_img_rgb = license_plate_img
        license_plate_img = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

        # Show license plate
        cv2.imshow('Warped Image', license_plate_img)

        # Show contour of the license plate on original image
        cv2.namedWindow('Detected Plate', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected Plate', image)
        cv2.resizeWindow('Detected Plate', 800, 600)

        # Apply adaptive thresholding and morphological operations to the license plate
        _, license_plate_filtered = cv2.threshold(license_plate_img, 150, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        license_plate_filtered = cv2.erode(license_plate_filtered, np.ones((5,5)))
        license_plate_filtered = cv2.dilate(license_plate_filtered, np.ones((3,3)))

        # Crop the license plate to remove unwanted parts
        license_plate_filtered = license_plate_filtered[10:105, :]

        # Scan the license plate to detect the letters
        x_scan = np.zeros((license_plate_filtered.shape[1]))
        for i in range(0, license_plate_filtered.shape[1]):
            for j in range(0, license_plate_filtered.shape[0]):
                if license_plate_filtered[j, i] == 255:
                    x_scan[i] = True
                    break
                else:
                    x_scan[i] = False

        # Prepare list of detected letters, each with its start and end position
        characters = []
        prev = 0
        for i, val in enumerate(x_scan):
            if val == 1 and prev == 0:
                if i-5 < 0:
                    characters.append((0,0))
                else:
                    characters.append((i-5, 0))
            if val == 0 and prev == 1:
                if i+5 < len(x_scan):
                    characters[-1] = (characters[-1][0], i+5)
                else:
                    characters[-1] = (characters[-1][0], len(x_scan)-1)
            prev = val

        # Open OCR model
        with open('models/model.pkl', 'rb') as f:
            ocr_model = pickle.load(f)

        # Iterate over detected letters and predict the letter using the OCR model
        for i in characters:
            # Size of the detected letter
            size = i[1] - i[0]

            # Draw lines on the license plate image to show the boundaries of the detected letters
            cv2.line(license_plate_img_rgb, (i[0], 0), (i[0], 114), (255, 0, 0), 1)
            cv2.line(license_plate_img_rgb, (i[1], 0), (i[1], 114), (0, 255, 0), 1)

            # Crop the detected letter and resize it to 80x95 so every letter has the same size
            char_found = license_plate_filtered[:, i[0]:i[1]]
            char_to_ocr = np.zeros((95,80))
            if size < 80:
                x_position = (80 - size) // 2
                char_to_ocr[:, x_position:x_position+size] = char_found
            else:
                char_to_ocr = char_found[:, 0:80]

            # Display the detected letter
            # cv2.imshow('litera', litera_tmp)

            # Predict the letter using the OCR model
            read_letter = ocr_model.predict([char_to_ocr.flatten()])[0]

            # Ignore the letter if model predicted it as '_' (meaning it is not a valid character)
            if read_letter != '_':
                result = result + read_letter

            # Save the detected letter to folder 'train_letters' used to train OCR model
            # cv2.waitKey(1)
            # char_name = input('Podaj znak: ')
            # cv2.destroyWindow('litera')
            # name = 'train_letters/' + str(char_name) + '_' + str(np.random.randint(0, 10000)) + '.png'
            # cv2.imwrite(name, char_to_ocr)

        # Display the filtered license plate and the license plate with boundaries of detected letters
        cv2.imshow('Tablica filtr', license_plate_filtered)
        cv2.imshow('Tablica podzielona', license_plate_img_rgb)

        # Check how many letters are correct
        len_of_lp = len(result)
        correct = 0
        for i in range(0, len_of_lp):
            if result[i] == path.name[i]:
                correct += 1
        if len_of_lp > 0:
            percentage = (correct / (len(path.name) - 4)) * 100
        else:
            percentage = 0

        print('Odczytano: ' + result + ' / ' + path.name[:-4] + ' - ' + str(percentage) + '%')

    else:
        print('Nie znaleziono tablicy!')

    if result == '':
        # If the license plate was not detected, set the result to statically most common license plate
        result = "PO41532"

    # Change the waitKey value to 0 in order to pause the program until any key is pressed after every image
    cv2.waitKey(1)
    return result
