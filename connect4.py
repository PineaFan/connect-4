import math
import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.25,
)
mpDraw = mp.solutions.drawing_utils


class Board:
    def __init__(self, width=7, height=6):
        self.height = height
        self.width = width
        self.board = [[None for x in range(width)] for y in range(height)]
        self.turn = 0
        self.winner = None

    def dropInColumn(self, col, player):
        if self.board[0][col] != None:
            return False
        for row in range(5, -1, -1):
            if self.board[row][col] == None:
                self.board[row][col] = player
                return True
        return False


def render(results, board, columnHighlight, pieceToDrop):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        results = hands.process(imgRGB)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    try:
        print(results.multi_hand_landmarks[0].landmark[8].x)
    except:
        pass

    screen_width = img.shape[1]
    screen_height = img.shape[0]
    columns = board.width
    rows = board.height
    margins = 25

    radius = round(((screen_height / 2) - margins) / (columns + margins / 2))
    spacing_horizontal = ((screen_width - 2 * margins) - (2 * radius * columns)) / (columns - 1)
    spacing_vertical = ((screen_height/2 - 2 * margins) - (2 * radius * rows)) / (rows - 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (screen_width, int(round(screen_height/2))), (255, 255, 255), -1)
    img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    colours = [
        (242, 120, 120),
        (101, 118, 204)
    ]

    if columnHighlight and pieceToDrop != None:
        # Creates a rectangle below the column you will drop the piece in
        # This should be shown under the circles
        # It should have a width of 2 * radius, and a height of the margin
        cv2.rectangle(
            img,
            (
                int(round(margins + (columnHighlight - 1) * (2 * radius + spacing_horizontal))),
                int(round(screen_height/2 - margins / 2))
            ),
            (
                int(round(margins + (columnHighlight - 1) * (2 * radius + spacing_horizontal) + 2 * radius)),
                int(round(screen_height/2))
            ),
            colours[pieceToDrop - 1], -1
        )

    for row in range(rows):
        for col in range(columns):
            center = (round(margins + radius + col * (2 * radius + spacing_horizontal)), round(margins + radius + row * (2 * radius + spacing_vertical)))
            if board.board[row][col] and board.board[row][col] < len(colours):
                cv2.circle(img, center, radius, colours[board.board[row][col]], cv2.FILLED)
            else:
                cv2.circle(img, center, radius, (66, 66, 66), 2)


    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)


def fromList(l, a):
    return [l[i] for i in a]


def getFingerCoords():
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        results = hands.process(imgRGB)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    landmarks = results.multi_hand_landmarks
    if landmarks:
        return landmarks[0].landmark, results
    return landmarks, results


def distance(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def center(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2


precision = 1000
pinchTime = 0
threshold = 10
game = Board()
# Get camera width
width = cap.read()[1].shape[1]
print(width)

holdingPiece = False
nearestColumn = 0

try:
    while True:
        landmarks, results = getFingerCoords()
        if landmarks:
            points = [[p.x, p.y, p.z, 2] for p in landmarks]
            origin = points[0]  # Make this the center
            for index, point in enumerate(points):  # For each point
                points[index] = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]]  # Translate such that point 0 is at (0, 0, 0)
                points[index] = [round(points[index][0] * precision), round(points[index][1] * precision), round(points[index][2] * precision)]  # Scale to a reasonable size

            if distance(points[4], points[8]) < distance(points[2], points[5]):
                pinchTime = max(pinchTime + 1, 0)
            else:
                pinchTime = min(pinchTime - 1, 0)
            if pinchTime == threshold:
                print("Pinch")
                holdingPiece = True
            if pinchTime == -threshold:
                print("Drop")
                holdingPiece = False

            # Find the column the finger is over. The amount of columns is game.width
            pointX = points[5][0] / 2 * 10
            print(pointX)
            nearestColumn = round(pointX / (width / game.width))

        render(results, game, nearestColumn, game.turn if holdingPiece else False)

        cv2.waitKey(1)

except Exception as e:
    print(e)
