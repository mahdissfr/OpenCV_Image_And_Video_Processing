import cv2


def show_orginal_photo(path):
    img = cv2.imread(path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def non_blue_filter(img):
    nblue = img
    nblue[:, :, 0] = 0
    cv2.namedWindow('with out blue', cv2.WINDOW_NORMAL)
    cv2.imshow('with out ', nblue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blue_filter(img):
    blue = img
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0
    cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
    cv2.imshow('blue', blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_scale(img):
    gray = img
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gray


def GaussianBlur(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.namedWindow('smoothed', cv2.WINDOW_NORMAL)
    cv2.imshow('smoothed', blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_detector(img):
    edges = cv2.Canny(img, 100, 200)
    cv2.namedWindow('edge_detector', cv2.WINDOW_NORMAL)
    cv2.imshow("edge_detector", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, w=0.5, h=1):
    width = int(img.shape[1] * w)
    height = int(img.shape[0] * h)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, cv2.INTER_AREA)  # Resize image
    cv2.namedWindow('resize', cv2.WINDOW_NORMAL)
    cv2.imshow("resize", resized)  # Show image
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate(image, angle=90, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    cv2.namedWindow('rotate', cv2.WINDOW_NORMAL)
    cv2.imshow('rotate', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentation(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # initialize the list of threshold methods
    methods = [
        ("THRESH_BINARY", cv2.THRESH_BINARY),
        ("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
        ("THRESH_TRUNC", cv2.THRESH_TRUNC),
        ("THRESH_TOZERO", cv2.THRESH_TOZERO),
        ("THRESH_TOZERO INV", cv2.THRESH_TOZERO_INV)]

    for (threshName, threshMethod) in methods:
        (T, thresh) = cv2.threshold(gray, threshold, 255, threshMethod)
        cv2.namedWindow(threshName, cv2.WINDOW_NORMAL)
        cv2.imshow(threshName, thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def face_detection(image):
    face_cascade_def = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    face_cascade_ext = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
    face_cascade_a = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_cascade_a2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    face_cascade_at = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_def = face_cascade_def.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_def:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    face_a2 = face_cascade_a2.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_a2:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    face_a = face_cascade_a.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_a:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    face_ext = face_cascade_ext.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_ext:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    face_at = face_cascade_at.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face_at:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.namedWindow('face_detection', cv2.WINDOW_NORMAL)
    cv2.imshow('face_detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_capture_frame(path):
    vidcap = cv2.VideoCapture(path)
    counter = 0
    while (counter < 5):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (counter + 1) * 500)  # just cue to 20 sec. position
        success, image = vidcap.read()
        if success:
            counter += 1
            cv2.imwrite(str(counter) + "th sec.jpg", image)
            cv2.imshow("20sec", image)
            cv2.waitKey()
    print("done")


decision = input("1.video capture frame\n"
                 "2.show input photo\n"
                 "3.face detection\n"
                 "4.segmentation\n"
                 "5.blue filter\n"
                 "6.non blue filter\n"
                 "7.gray scale\n"
                 "8.gaussian filter\n"
                 "9.edge detector\n"
                 "10.resize\n"
                 "11.rotate\n"
                 "12.end\n")

while decision != "12":
    if decision == "1":
        video_capture_frame("test.avi")
    elif decision == "2":
        show_orginal_photo("test.jpg")
    elif decision == "3":
        img = cv2.imread("test.jpg")
        face_detection(img)
    elif decision == "4":
        img = cv2.imread("test.jpg")
        segmentation(img)
    elif decision == "5":
        img = cv2.imread("test.jpg")
        blue_filter(img)
    elif decision == "6":
        img = cv2.imread("test.jpg")
        non_blue_filter(img)
    elif decision == "7":
        img = cv2.imread("test.jpg")
        gray_scale(img)
    elif decision == "8":
        img = cv2.imread("test.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GaussianBlur(gray)
    elif decision == "9":
        img = cv2.imread("test.jpg")
        edge_detector(img)
    elif decision == "10":
        img = cv2.imread("test.jpg")
        resize(img, 0.5, 1)
    elif decision == "11":
        img = cv2.imread("test.jpg")
        rotate(img, 90)
    else:
        print("wrong input")

    decision = input("1.video capture frame\n"
                     "2.show input photo\n"
                     "3.face detection\n"
                     "4.segmentation\n"
                     "5.blue filter\n"
                     "6.non blue filter\n"
                     "7.gray scale\n"
                     "8.gaussian filter\n"
                     "9.edge detector\n"
                     "10.resize\n"
                     "11.rotate\n"
                     "12.end\n")
