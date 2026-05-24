import cv2
import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = "videos/garts1.MOV"
TEMPLATE_DIR = "templates"
FONT_PATH = "DejaVuSans.ttf"

SAVE_DIR = "detected_frames"

RATIO_THRESH = 0.68
MIN_MATCH_COUNT = 12
RANSAC_THRESH = 2.5
MIN_INLIER_RATIO = 0.40
MIN_INLIERS = 8

USE_TRACKER = True
TRACKER_TYPE = "CSRT"

REAL_SIGN_WIDTH_M = 0.60
FOCAL_LENGTH_PX = 620.0

SAVE_DETECTED_FRAMES = True
SAVE_EVERY_DETECTION = False
SHOW_MATCH_WINDOW = False

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

NMS_IOU_THRESH = 0.35
MAX_TRACK_FAIL = 8
DETECT_EVERY_N_FRAMES = 15


os.makedirs(SAVE_DIR, exist_ok=True)

SIGN_CATEGORIES = {
    "50.png": "Хурдны хязгаарлалтын",
    "zogs.png": "Зогс",
    "garts.png": "Явган хүний гарц",
}


def get_sign_category(template_name):
    return SIGN_CATEGORIES.get(template_name, "Замын тэмдэг")


def sign_color_valid(frame, bbox, template_name):
    """
    False positive багасгах өнгөний шалгалт.
    SIFT заримдаа бичигтэй цаас, ханын texture дээр буруу таарах тул
    template-ийн үндсэн өнгө crop дотор байгаа эсэхийг шалгана.
    """
    x, y, w, h = [int(v) for v in bbox]
    H, W = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # red: hue 0 орчим болон 180 орчим хоёр мужтай
    red1 = cv2.inRange(hsv, (0, 45, 45), (12, 255, 255))
    red2 = cv2.inRange(hsv, (165, 45, 45), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # blue/cyan signs
    blue_mask = cv2.inRange(hsv, (85, 35, 45), (135, 255, 255))

    total = crop.shape[0] * crop.shape[1]
    red_ratio = cv2.countNonZero(red_mask) / max(total, 1)
    blue_ratio = cv2.countNonZero(blue_mask) / max(total, 1)

    name = template_name.lower()

    # Явган хүний гарц, сургууль орчмын бүс зэрэг цэнхэр суурьтай тэмдэг
    if name in ["garts.png", "sur.png"]:
        return blue_ratio >= 0.025

    # Улаан хүрээ/улаан суурьтай тэмдэг
    if name in ["zogs.png", "hvvhd.png", "50.png", "hurd.png"]:
        return red_ratio >= 0.020

    # Ерөнхий fallback: аль нэг замын тэмдгийн өнгө тодорхой хэмжээнд байвал зөвшөөрнө
    return (red_ratio + blue_ratio) >= 0.020


def fit_to_screen(frame, max_w=1280, max_h=720):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)

    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame


def create_tracker():
    t = TRACKER_TYPE.upper()

    if t == "CSRT":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
    elif t == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    else:
        raise RuntimeError("Tracker create function OpenCV дээр олдсонгүй.")


def estimate_distance(
    box_width_px, real_width_m=REAL_SIGN_WIDTH_M, focal_px=FOCAL_LENGTH_PX
):
    if box_width_px <= 0:
        return None
    return (real_width_m * focal_px) / box_width_px


def load_mongolian_font(size=28):
    if not os.path.isfile(FONT_PATH):
        raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
    return ImageFont.truetype(FONT_PATH, size)


def draw_text_pil(
    frame, text, position, font, text_color=(255, 255, 255), bg_color=None, padding=10
):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if bg_color is not None:
        draw.rectangle(
            [x - padding, y - padding, x + text_w + padding, y + text_h + padding],
            fill=bg_color,
        )

    draw.text((x, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_top_center_alert(frame, detections):
    if not detections:
        return frame

    dets = []
    for det in detections:
        x, y, w, h = det["bbox"]
        dist = estimate_distance(w)
        if dist is not None:
            dets.append((dist, det))

    if not dets:
        return frame

    nearest_dist, nearest_det = min(dets, key=lambda x: x[0])
    category = get_sign_category(nearest_det["template_name"])
    message = f"{category} тэмдэг {nearest_dist:.1f} м зайд байна!"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    w_frame, h_frame = pil_img.size

    title_font_size = max(24, int(h_frame * 0.028))
    body_font_size = max(24, int(h_frame * 0.026))

    title_font = ImageFont.truetype(FONT_PATH, title_font_size)
    body_font = ImageFont.truetype(FONT_PATH, body_font_size)

    title = "АНХААР !!!"

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]

    msg_bbox = draw.textbbox((0, 0), message, font=body_font)
    msg_w = msg_bbox[2] - msg_bbox[0]
    msg_h = msg_bbox[3] - msg_bbox[1]

    box_padding_x = max(18, int(w_frame * 0.02))
    box_padding_y = max(12, int(h_frame * 0.012))

    box_w = msg_w + box_padding_x * 2
    box_h = msg_h + box_padding_y * 2

    top_margin = max(45, int(h_frame * 0.06))
    box_x = (w_frame - box_w) // 2
    box_y = top_margin

    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(255, 255, 255))
    draw.rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h], outline=(180, 180, 180), width=2
    )

    title_y = max(8, int(h_frame * 0.015))
    title_x = (w_frame - title_w) // 2
    draw.text((title_x, title_y), title, font=title_font, fill=(255, 0, 0))

    msg_x = (w_frame - msg_w) // 2
    msg_y = box_y + box_padding_y
    draw.text((msg_x, msg_y), message, font=body_font, fill=(40, 40, 40))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_templates(template_dir):
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template folder not found: {template_dir}")

    sift = cv2.SIFT_create()
    templates = []

    allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    for fname in sorted(os.listdir(template_dir)):
        if not fname.lower().endswith(allowed_ext):
            continue

        path = os.path.join(template_dir, fname)
        img = cv2.imread(path)

        if img is None:
            print(f"[SKIP] read failed: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        if des is None or kp is None or len(kp) < 2:
            print(f"[SKIP] feature хангалтгүй: {fname}")
            continue

        templates.append(
            {
                "name": fname,
                "path": path,
                "img": img,
                "gray": gray,
                "kp": kp,
                "des": des,
            }
        )

        print(f"[LOAD] {fname} | keypoints={len(kp)}")

    if len(templates) == 0:
        raise RuntimeError(
            "Template folder дотор ашиглах боломжтой template олдсонгүй."
        )

    return sift, templates


def detect_with_one_template(frame, kp_f, des_f, template, matcher):
    kp_t = template["kp"]
    des_t = template["des"]
    template_gray = template["gray"]

    debug = {
        "template_name": template["name"],
        "kp_frame": 0 if kp_f is None else len(kp_f),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "match_vis": None,
    }

    if des_f is None or kp_f is None or len(kp_f) < 2:
        return None, debug

    matches = matcher.knnMatch(des_t, des_f, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESH * n.distance:
                good_matches.append(m)

    debug["good_matches"] = len(good_matches)

    if len(good_matches) < MIN_MATCH_COUNT:
        return None, debug

    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)

    if H is None or mask is None:
        return None, debug

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good_matches)

    debug["inliers"] = inliers
    debug["inlier_ratio"] = inlier_ratio

    if inlier_ratio < MIN_INLIER_RATIO:
        return None, debug

    h_t, w_t = template_gray.shape
    corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H)
    x, y, w, h = cv2.boundingRect(np.int32(projected))

    frame_h, frame_w = frame.shape[:2]

    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return None, debug

    if w < 30 or h < 30:
        return None, debug

    # Хэт том box нь ихэнхдээ ханын бичиг, шүүгээ, texture дээр буруу таарсан байдаг.
    if w > frame_w * 0.50 or h > frame_h * 0.55:
        return None, debug

    box_area = w * h
    frame_area = frame_w * frame_h
    if box_area > frame_area * 0.28:
        return None, debug

    aspect = w / max(h, 1)
    if aspect < 0.50 or aspect > 2.0:
        return None, debug

    if inliers < MIN_INLIERS:
        return None, debug

    if not sign_color_valid(frame, (x, y, w, h), template["name"]):
        return None, debug

    detection = {
        "bbox": (x, y, w, h),
        "polygon": np.int32(projected),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "template_name": template["name"],
        "score": len(good_matches) * inlier_ratio,
    }

    if SHOW_MATCH_WINDOW:
        debug["match_vis"] = cv2.drawMatches(
            cv2.cvtColor(template_gray, cv2.COLOR_GRAY2BGR),
            kp_t,
            frame,
            kp_f,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    return detection, debug


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(1, boxA[2] * boxA[3])
    areaB = max(1, boxB[2] * boxB[3])

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def nms_detections(detections, iou_thresh=0.35):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in detections:
        keep = True
        for k in kept:
            if compute_iou(det["bbox"], k["bbox"]) > iou_thresh:
                keep = False
                break
        if keep:
            kept.append(det)

    return kept


def detect_sign_in_frame_multi(frame, sift, templates):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = sift.detectAndCompute(gray, None)

    debug = {
        "kp_frame": 0 if kp_f is None else len(kp_f),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "template_name": "-",
        "match_vis": None,
    }

    if des_f is None or kp_f is None or len(kp_f) < 2:
        return [], debug

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=100)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    all_detections = []
    best_debug = debug
    best_score = -1

    for template in templates:
        detection, d = detect_with_one_template(frame, kp_f, des_f, template, matcher)

        if detection is None:
            continue

        all_detections.append(detection)

        score = detection["score"]
        if score > best_score:
            best_score = score
            best_debug = d

    final_detections = nms_detections(all_detections, iou_thresh=NMS_IOU_THRESH)
    final_detections = [
        det
        for det in final_detections
        if valid_bbox(det["bbox"], frame.shape)
        and sign_color_valid(frame, det["bbox"], det["template_name"])
    ]

    if len(final_detections) > 0:
        return final_detections, best_debug

    return [], debug


def draw_detection(frame, detection, font):
    x, y, w, h = detection["bbox"]
    polygon = detection["polygon"]

    cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(
        frame,
        "ALERT !!!",
        (x, max(30, y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    distance_m = estimate_distance(w)
    if distance_m is not None:
        text = f"{distance_m:.1f} м"
        text_y = y + h + 8
        frame = draw_text_pil(frame, text, (x, text_y), font, text_color=(0, 255, 0))

    return frame


def bbox_to_polygon(bbox):
    x, y, w, h = bbox
    return np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.int32)


def valid_bbox(bbox, frame_shape):
    """Tracker-iin box boditoi esehiig shalgana."""
    x, y, w, h = [int(v) for v in bbox]
    frame_h, frame_w = frame_shape[:2]

    if w < 25 or h < 25:
        return False

    if w > frame_w * 0.55 or h > frame_h * 0.55:
        return False

    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return False

    aspect = w / max(h, 1)
    if aspect < 0.55 or aspect > 1.8:
        return False

    return True


class MultiTrackerRuntime:
    def __init__(self, template_dir=TEMPLATE_DIR):
        self.sift, self.templates = load_templates(template_dir)
        self.trackers = []
        self.last_saved_frame_idx = -9999
        self.frame_idx = 0
        self.start_time = time.time()
        self.debug = None
        self.need_detection = True
        self.font_small = load_mongolian_font(28)

    def _reset_trackers_from_detections(self, frame, detections):
        """Detection-iin ur dun deer tracker-uudiig dahin uusgene."""
        self.trackers = []

        if not USE_TRACKER:
            return

        for det in detections:
            if not valid_bbox(det["bbox"], frame.shape):
                continue

            try:
                tr = create_tracker()
                tr.init(frame, det["bbox"])
                self.trackers.append(
                    {
                        "tracker": tr,
                        "template_name": det["template_name"],
                        "bbox": det["bbox"],
                        "fail_count": 0,
                    }
                )
            except Exception as e:
                print("Tracker init error:", e)

    def _update_trackers(self, frame):
        """Tracker-uudiig update hiigeed huchintei box-uudiig detection helbereer butsaana."""
        detections = []
        active_trackers = []

        for item in self.trackers:
            tracker = item["tracker"]
            template_name = item["template_name"]

            ok, bbox = tracker.update(frame)
            if not ok:
                item["fail_count"] += 1
                if item["fail_count"] <= MAX_TRACK_FAIL:
                    active_trackers.append(item)
                continue

            x, y, w, h = [int(v) for v in bbox]
            bbox = (x, y, w, h)

            if not valid_bbox(bbox, frame.shape) or not sign_color_valid(
                frame, bbox, template_name
            ):
                item["fail_count"] += 1
                if item["fail_count"] <= MAX_TRACK_FAIL:
                    active_trackers.append(item)
                continue

            item["bbox"] = bbox
            item["fail_count"] = 0

            detections.append(
                {
                    "bbox": bbox,
                    "polygon": bbox_to_polygon(bbox),
                    "good_matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "template_name": template_name,
                    "score": 1.0,
                }
            )
            active_trackers.append(item)

        self.trackers = active_trackers

        if len(self.trackers) == 0:
            self.need_detection = True

        return detections

    def process_frame(self, frame):
        display = frame.copy()
        detections = []

        # Tracker ajillaj baisan ch uye uye SIFT detection dahin hiine.
        # Ene ni shine temdeg garj irehed tanah, huuchin tracker drift hiihiig bagasgana.
        do_detection = (
            self.need_detection
            or len(self.trackers) == 0
            or self.frame_idx % DETECT_EVERY_N_FRAMES == 0
        )

        if do_detection:
            found_detections, self.debug = detect_sign_in_frame_multi(
                frame, self.sift, self.templates
            )

            if len(found_detections) > 0:
                detections = [
                    det
                    for det in found_detections
                    if valid_bbox(det["bbox"], frame.shape)
                    and sign_color_valid(frame, det["bbox"], det["template_name"])
                ]
                self._reset_trackers_from_detections(frame, detections)
                self.need_detection = len(self.trackers) == 0

                if len(detections) == 0 and len(self.trackers) > 0:
                    detections = self._update_trackers(frame)
            else:
                # Detection oldohgui ued tracker baival tuugeer tur urgeljluulne.
                detections = (
                    self._update_trackers(frame) if len(self.trackers) > 0 else []
                )
                self.need_detection = len(self.trackers) == 0
        else:
            detections = self._update_trackers(frame)
        if len(detections) > 0:
            for det in detections:
                display = draw_detection(display, det, self.font_small)

            display = draw_top_center_alert(display, detections)

            if SAVE_DETECTED_FRAMES:
                should_save = False

                if SAVE_EVERY_DETECTION:
                    should_save = True
                else:
                    if self.frame_idx - self.last_saved_frame_idx > 20:
                        should_save = True

                if should_save:
                    save_path = os.path.join(
                        SAVE_DIR, f"detected_{self.frame_idx:05d}.jpg"
                    )
                    cv2.imwrite(save_path, display)
                    self.last_saved_frame_idx = self.frame_idx

        elapsed = time.time() - self.start_time
        fps = (self.frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        cv2.putText(
            display,
            f"Frame: {self.frame_idx}",
            (10, display.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

#        cv2.putText(
#            display,
#            f"Trackers: {len(self.trackers)}",
#            (10, 30),
#            cv2.FONT_HERSHEY_SIMPLEX,
#            0.8,
#            (0, 255, 255),
#            2,
#        )

 #       if self.debug is not None:
 #           cv2.putText(
 #               display,
 #               f"Best template: {self.debug['template_name']}",
 #               (10, 60),
 #               cv2.FONT_HERSHEY_SIMPLEX,
 #               0.7,
 #               (0, 255, 255),
 #               2,
 #           )

        result = {
            "frame": display,
            "detections": detections,
            "debug": self.debug,
            "fps": fps,
            "trackers": len(self.trackers),
            "frame_idx": self.frame_idx,
        }

        self.frame_idx += 1
        return result


def create_runtime(template_dir=TEMPLATE_DIR):
    return MultiTrackerRuntime(template_dir=template_dir)


def process_frame_with_runtime(runtime, frame):
    return runtime.process_frame(frame)


def main():
    runtime = create_runtime(TEMPLATE_DIR)
    print(f"Loaded templates: {len(runtime.templates)}")


    # IRIUN 
    CAMERA_INDEX = 0  # Эхлээд 0, дараа нь 1, 2, 3 гэж солиод турш

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Камер нээгдсэнгүй.")
        print("1. Iriun Webcam PC дээр нээлттэй эсэхийг шалга.")
        print("2. iPhone дээр Iriun app нээлттэй эсэхийг шалга.")
        print("3. CAMERA_INDEX утгыг 0, 1, 2, 3 гэж солиод үз.")
        raise SystemExit

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Traffic Sign Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Sign Detection", DISPLAY_MAX_W, DISPLAY_MAX_H)

    if SHOW_MATCH_WINDOW:
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", 1200, 700)

    print("Камер ажиллаж байна.")
   # print("ESC дарвал гарна.")
   # print("R дарвал detection reset хийнэ.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Камераас frame уншиж чадсангүй.")
            break

        result = runtime.process_frame(frame)
        display = result["frame"]
        debug = result["debug"]

        display_show = fit_to_screen(display, DISPLAY_MAX_W, DISPLAY_MAX_H)
        cv2.imshow("Traffic Sign Detection", display_show)

        if SHOW_MATCH_WINDOW and debug is not None and debug["match_vis"] is not None:
            match_show = fit_to_screen(debug["match_vis"], 1200, 700)
            cv2.imshow("Matches", match_show)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == ord("r"):
            runtime.need_detection = True
            runtime.trackers = []
            print("Detection reset хийлээ.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
