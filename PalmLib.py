# palm tracking and detection library
# created by omar aboulubdeh for multimedia controller project
import cv2
import numpy as np
from classifier import  Geture
from datetime import datetime
from DMImage import DMImagePreprocessor

hist_mask = None

tmp = None
time = 0

traverse_point = []
classifier = Geture(1)

total_rectangle = 9
view_channels = False
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

def hand(right= False):
    if right:
        classifier.right_hand = True
    else:
        classifier.right_hand = False

def imshow(title, frame):
    global view_channels
    if view_channels:
        cv2.imshow(title, frame)

def toggle_channels():
    global view_channels
    if view_channels:
        view_channels = False
        cv2.destroyAllWindows()
    else:
        view_channels = True

def draw_palm(frame):
    palm = cv2.imread('imgs/palm_edges.png', cv2.IMREAD_UNCHANGED)

    indx = frame.shape[0] > frame.shape
    scale = frame.shape[0] / palm.shape[0]

    width = int(palm.shape[1] * scale)
    height = int(palm.shape[0] * scale)

    dim = (width, height)

    resized_palm = cv2.resize(palm, dim, interpolation=cv2.INTER_AREA)

def draw_one_rect(frame):
    rows, cols, _ = frame.shape
    y1 = int(6 * rows / 20)
    y2 = int(12 * rows / 20) + 10

    x1 = int(8 * cols / 20)
    x2 = int(12 * cols / 20) + 10
    # cv2.imshow('test',frame[x:x+10, y:y+10, :])
    total_rectangle = 1
    cv2.rectangle(frame, (x1, y1),(x2, y2),(0, 255, 0), 1)
    _ = one_rect_hand_histogram(frame)
    return frame

def roi_to_square(img):
    h,w = img.shape[0:2]
    if w > h :
        diff = (w-h)/2
        img = cv2.copyMakeBorder(img, int(diff),int(diff),0,0,cv2.BORDER_CONSTANT, value=0)
    else :
        diff = (h-w)/2
        img = cv2.copyMakeBorder(img, 0,0,int(diff),int(diff),cv2.BORDER_CONSTANT, value=0)

    img = cv2.resize(img, (33,33))
    return img

def activate_palm(palm):
    global hist_mask
    hist_mask = palm

def create_palm(name, camera):
    global hand_hist
    while camera.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = camera.read()
        cv2.putText(frame, "Create Your "+name+" Palm:", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (82, 234, 201,255), lineType=cv2.LINE_AA )
        cv2.putText(frame, "Place your fist above the green rectangle then press z",(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 240, 201), lineType=cv2.LINE_AA )
        if pressed_key & 0xFF == ord('z'):
            # hand_hist = hand_histogram(frame)
            hand_hist = one_rect_hand_histogram(frame)
            np.save('./palms/'+name+'.palm.npy',hand_hist)
            break
        else:
            frame = draw_rect(frame)
            frame = draw_one_rect(frame)
        cv2.imshow("Create "+name+" Palm", rescale_frame(frame))
    cv2.destroyAllWindows()

# def roi(frame):
def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
         9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    x = int(6 * rows / 20)
    y = int(9 * cols / 20)
    # imshow('test',frame[x:x+10, y:y+10, :])
    hsv = cv2.cvtColor(frame[x:x + 10, y:y + 10, :], cv2.COLOR_BGR2HSV)
    imshow('test', hsv)

    print(hsv[5, 5])
    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    ret, im_bw = cv2.threshold(thresh, 200, 255, cv2.THRESH_BINARY)
    _, cont, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def max_contour(contour_list):
    max_i = 0
    max_area = 0

    lenth = len(contour_list)
    for i in range(lenth):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def one_rect_hand_histogram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rows, cols, _ = frame.shape

    y1 = int(6 * rows / 20)
    y2 = int(12 * rows / 20) + 10

    x1 = int(8 * cols / 20)
    x2 = int(12 * cols / 20) + 10

    roi = np.zeros([y2-y1, x2-x1, 3], dtype=hsv_frame.dtype)
    roi[:,:] = hsv_frame[y1:y2,x1:x2]
    cv2.imshow('roi',roi)
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    imshow('roi',roi)
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def img_fill(im_in, n):  # n = binary image threshold
    th, im_th = cv2.threshold(im_in, n, 255, cv2.THRESH_BINARY);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv

    return fill_image

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    imshow('#2 hsv filtering', dst)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,20))
    cv2.filter2D(dst, -1, disc, dst)
    imshow('#3 filter2D image', dst)

    ret, thresh = cv2.threshold(dst, 30, 255, cv2.THRESH_BINARY)
    imshow('#4 threshold grayscale image',thresh)

    thresh = cv2.medianBlur(thresh, 9)
    thresh = cv2.medianBlur(thresh, 9)
    imshow('#5 median blur',thresh)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.erode(thresh, disc, iterations=2)
    imshow('#6 apply erosion', thresh)
    # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 3))
    # thresh = cv2.erode(thresh,disc,iterations=2)
    # imshow('#6 apply erosion',thresh)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.dilate(thresh,disc,iterations=1)
    imshow('#7 apply dilation',thresh)

    thresh = cv2.merge((thresh, thresh, thresh))
    processor = DMImagePreprocessor();
    try:
        flood = processor.select_largest_obj(img_bin=thresh[:, :, 0], fill_holes=True)
        imshow('#8 floodfill to select largest object', flood)
        r = cv2.bitwise_and(frame[:, :, 0], flood)
        g = cv2.bitwise_and(frame[:, :, 1], flood)
        b = cv2.bitwise_and(frame[:, :, 2], flood)
        thresh = cv2.merge((r, g, b))

        imshow('#9 merging frame with our final mask', thresh)
        return flood, thresh
    except:
        return thresh*0, thresh*0

def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def max_peaks(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)
        print("peak points: "+ len(s).__str__())
        # global prev_button
        # if len(s) == 2 and prev_button == 'down':
        #     controller.left_up()
        #     prev_button = 'up'
        # if len(s) == 3 and prev_button == 'up' :
        #     controller.left_down()
        #     prev_button = 'down'
        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            print("farest", farthest_point)
            return farthest_point,
        else:
            return None

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        jtest = defects[:,0]

        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp,     yp))

        dist_max_i = np.argmax(dist)
        # global prev_button
        # if len(s) == 2 and prev_button == 'down':
        #     controller.left_up()
        #     prev_button = 'up'
        # if len(s) == 3 and prev_button == 'up' :
        #     controller.left_down()
        #     prev_button = 'down'
        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point, s.size
        else:
            return None, 0
    return None, 0

# def fingers_count(defects, contour, centroid):
    # if defects is not

def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [217, 66, 244], -1)


def palm_info(frame):
    global hist_mask
    mask, hist_mask_image = hist_masking(frame, hist_mask)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)
    if max_cont is not None:
        epsilon = 0.01 * cv2.arcLength(max_cont, True)
        max_cont = cv2.approxPolyDP(max_cont, epsilon, True)

        cnt_centroid = centroid(max_cont)
        cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)
        hull = cv2.convexHull(max_cont, returnPoints=False)
        cv2.drawContours(frame, [max_cont], 0,(0,0,255),2)
        imshow('hull1', frame)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point, peak_points = farthest_point(defects, max_cont, cnt_centroid)
        # max_peaks(defects,max_cont,cnt_centroid)
        global cursor
        if (far_point):
            # new_cursor = np.array([(600-cnt_centroid[0])*2, cnt_centroid[1]*2])
            # controller.set_cursor(new_cursor)
            # new_cursor = np.array([cnt_centroid[0]],cnt_centroid[1])
            cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
            if len(traverse_point) < 10:
                traverse_point.append(far_point)
            else:
                traverse_point.pop(0)
                traverse_point.append(far_point)
        draw_circles(frame, traverse_point)
        imshow('#10 track farthest point',frame)
        x,y,w,h = cv2.boundingRect(max_cont)
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 1)
        roi = mask[y:y+h,x:x+w]
        model_input = roi_to_square(roi)
        gesture = classifier.classify(model_input,peak_points)
        return cnt_centroid, gesture
    return None, None
