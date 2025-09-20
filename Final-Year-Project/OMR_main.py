import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import utils

########################################################################
pathImage = "a.jpeg"
heightImg = 700
widthImg = 700
questions = 5
choices = 5

# Read correct answers from Excel (robust)
excel_path = "Key.xlsx"  # <-- change if needed

if not os.path.exists(pathImage):
    raise FileNotFoundError(f"Image file not found: {pathImage}")

if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel key file not found: {excel_path}")

df = pd.read_excel(excel_path)
print("Excel columns:", df.columns.tolist())

# Select column that contains answers. Use 'Python' if exists, else first column.
key_col = "Python" if "Python" in df.columns else df.columns[0]
print("Using answer column:", key_col)

# Parse answers robustly
ans = []
for i in range(questions):
    try:
        cell = str(df[key_col].iloc[i])
    except Exception:
        raise IndexError(f"Not enough rows in Excel sheet for {questions} questions. Found {len(df)}")
    # Try to extract a letter a-e from the cell (case-insensitive)
    m = re.search(r'([a-eA-E])', cell)
    if not m:
        # If cell looks like a digit (e.g., 1,2 -> mapped to 0,1), try that
        mnum = re.search(r'(\d+)', cell)
        if mnum:
            val = int(mnum.group(1)) - 1
            if 0 <= val < choices:
                ans.append(val)
                continue
        raise ValueError(f"Could not parse answer from cell '{cell}' (row {i})")
    letter = m.group(1).lower()
    ans.append(ord(letter) - ord('a'))  # a->0, b->1, ...

print("Parsed answer key (0-indexed):", ans)
########################################################################

# Helper for robust findContours with different OpenCV returns
def find_contours_robust(bin_img):
    contours_data = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # OpenCV returns either (contours, hierarchy) or (image, contours, hierarchy)
    if len(contours_data) == 2:
        contours, hierarchy = contours_data
    else:
        _, contours, hierarchy = contours_data
    return contours, hierarchy

# Main loop
while True:
    start = time.time()
    img = cv2.imread(pathImage)
    if img is None:
        print("Failed to read image. Check path:", pathImage)
        break

    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)

    try:
        imgContours = img.copy()
        imgBigContour = img.copy()

        contours, hierarchy = find_contours_robust(imgCanny)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        rectCon = utils.rectContour(contours)  # expected list of rectangle contours sorted by area

        # require at least two rectangular contours (sheet + grade box)
        if len(rectCon) < 2:
            raise ValueError("Could not find two rectangle contours (sheet and grade area). Found: " + str(len(rectCon)))

        biggestPoints = utils.getCornerPoints(rectCon[0])
        gradePoints = utils.getCornerPoints(rectCon[1])

        if biggestPoints.size != 0 and gradePoints.size != 0:
            # BIGGEST RECTANGLE WARPING
            biggestPoints = utils.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, [biggestPoints], -1, (0, 255, 0), 20)

            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # SECOND BIGGEST RECTANGLE WARPING (grade area)
            cv2.drawContours(imgBigContour, [gradePoints], -1, (255, 0, 0), 20)
            gradePoints = utils.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            # SPLIT BOXES
            boxes = utils.splitBoxes(imgThresh)
            if len(boxes) != questions * choices:
                # The splitBoxes implementation might return a different ordering; warn but try to continue
                print(f"Warning: expected {questions*choices} boxes but got {len(boxes)}. Check splitBoxes implementation and the questions/choices values.")
                # Attempt to proceed by reshaping as many boxes as possible
                max_boxes = min(len(boxes), questions * choices)
            else:
                max_boxes = questions * choices

            # Count non-zero per box robustly and fill myPixelVal
            myPixelVal = np.zeros((questions, choices))
            for idx in range(max_boxes):
                r = idx // choices
                c = idx % choices
                box_img = boxes[idx]
                totalPixels = cv2.countNonZero(box_img)
                myPixelVal[r, c] = totalPixels

            # For rows that might not be complete, leave zeros (will choose argmax of available)
            # Find user answers by argmax per row
            myIndex = []
            for r in range(questions):
                row = myPixelVal[r]
                # if all zeros -> treat as unanswered (-1)
                if np.all(row == 0):
                    myIndex.append(-1)
                else:
                    myIndex.append(int(np.argmax(row)))

            # Compare with answer key
            grading = []
            for i_q in range(questions):
                if myIndex[i_q] == -1:
                    grading.append(0)
                elif ans[i_q] == myIndex[i_q]:
                    grading.append(1)
                else:
                    grading.append(0)

            score = (sum(grading) / questions) * 100
            print("SCORE", score)

            # Displaying answers on warp images
            utils.showAnswers(imgWarpColored, myIndex, grading, ans)
            utils.drawGrid(imgWarpColored)
            imgRawDrawings = np.zeros_like(imgWarpColored)
            utils.showAnswers(imgRawDrawings, myIndex, grading, ans)
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            # Display grade in grade box and inverse-warp it back
            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            # Show result windows
            imageArray = ([img, imgGray, imgCanny, imgContours],
                          [imgBigContour, imgThresh, imgWarpColored, imgFinal])
            cv2.imshow("Final Result", imgFinal)

        else:
            raise ValueError("Corner points for biggest/grade rectangles are empty.")

    except Exception as e:
        # Print exception details for debugging
        print("Image cannot be processed. Exception:", repr(e))
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
        cv2.imshow("Final Result", imgBlank)

    end = time.time()
    print("Time Taken: {:.3f}s".format(end - start))

    # LABELS FOR DISPLAY
    labels = [["Original", "Gray", "Edges", "Contours"],
              ["Biggest Contour", "Threshold", "Warpped", "Final"]]

    stackedImage = utils.stackImages(imageArray, 0.5, labels)
    cv2.imshow('Result', stackedImage)

    # Wait for a short keypress; press 'q' to quit, any other key to process image again
    key = cv2.waitKey(1) & 0xFF
    # If you want to pause for 5 seconds each loop, you can uncomment the following:
    # key = cv2.waitKey(5000) & 0xFF
    if key == ord('q'):
        print("Exit requested by user.")
        break

cv2.destroyAllWindows()
