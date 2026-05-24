import cv2
import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont

TEMPLATE_DIR = "templates"
FONT_PATH = "DejaVuSans.ttf"

RATIO_THRESH = 0.7
MIN_MATCH_COUNT = 8
RANSAC_THRESH = 3.0
MIN_INLIER_RATIO = 0.30

USE_TRACKER = True
TRACKER_TYPE = "CSRT"

REAL_SIGN_WIDTH_M = 0.60
FOCAL_LENGTH_PX = 620.0

SAVE_DETECTED_FRAMES = False
SAVE_EVERY_DETECTION = False
SHOW_MATCH_WINDOW = False

DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

NMS_IOU_THRESH = 0.35
MAX_TRACK_FAIL = 8


SIGN_CATEGORIES = {
    "20-1.png": "Замын тэмдэг",
    "20-6.png": "Замын тэмдэг",

    "child1.png": "Хүүхэд анхааруулах",
    "child2.png": "Хүүхэд анхааруулах",
    "child3.png": "Хүүхэд анхааруулах",

    "g.png": "Заах",

    "garts11.png": "Явган хүний гарц",
    "garts111.png": "Явган хүний гарц",
    "garts3.png": "Явган хүний гарц",
    "garts4.png": "Явган хүний гарц",
    "garts5.png": "Явган хүний гарц",
    "garts7.png": "Явган хүний гарц",
    "garts8.png": "Явган хүний гарц",
    "garts9.png": "Явган хүний гарц",

    "horigloh10.png": "Хориглох",
    "horigloh11.png": "Хориглох",
    "horigloh5.png": "Хориглох",
    "horigloh6.png": "Хориглох",
    "horigloh7.png": "Хориглох",
    "horigloh9.png": "Хориглох",

    "hurd1.png": "Хурдны хязгаарлалт",
    "hurd2.png": "Хурдны хязгаарлалт",

    "s.png": "Анхааруулах",

    "sur1.png": "Сургууль орчмын бүс",
    "sur2.png": "Сургууль орчмын бүс",
    "sur3.png": "Сургууль орчмын бүс",
    "sur4.png": "Сургууль орчмын бүс",
    "sur5.png": "Сургууль орчмын бүс",
    "sur6.png": "Сургууль орчмын бүс",
    "sur7.png": "Сургууль орчмын бүс",
    "sur8.png": "Сургууль орчмын бүс",

    "t2.png": "Замын тэмдэг",
    "t3.png": "Замын тэмдэг",
    "t4.png": "Замын тэмдэг",
    "t5.png": "Замын тэмдэг",
    "t6.png": "Замын тэмдэг",
    "t7.png": "Замын тэмдэг",
    "t8.png": "Замын тэмдэг",
    "t9.png": "Замын тэмдэг",
    "t10.png": "Замын тэмдэг",
    "t11.png": "Замын тэмдэг",
    "t12.png": "Замын тэмдэг",
    "t13.png": "Замын тэмдэг",
    "t14.png": "Замын тэмдэг",
    "t15.png": "Замын тэмдэг",
    "t16.png": "Замын тэмдэг",
    "t17.png": "Замын тэмдэг",
    "t18.png": "Замын тэмдэг",
    "t19.png": "Замын тэмдэг",
    "t20.png": "Замын тэмдэг",
    "t21.png": "Замын тэмдэг",
    "t22.png": "Замын тэмдэг",
    "t23.png": "Замын тэмдэг",
    "t24.png": "Замын тэмдэг",
    "t25.png": "Замын тэмдэг",
    "t26.png": "Замын тэмдэг",
    "t27.png": "Замын тэмдэг",
}



def get_sign_category(template_name):
    return SIGN_CATEGORIES.get(template_name, "Замын тэмдэг")


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
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()

    elif t == "KCF":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()

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

    if w < 15 or h < 15:
        return None, debug

    aspect = w / max(h, 1)
    if aspect < 0.4 or aspect > 2.5:
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

    def process_frame(self, frame):
        display = frame.copy()
        detections = []

        if self.need_detection:
            detections, self.debug = detect_sign_in_frame_multi(
                frame, self.sift, self.templates
            )

            self.trackers = []
            if len(detections) > 0 and USE_TRACKER:
                for det in detections:
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

                self.need_detection = False

        else:
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
                if w <= 10 or h <= 10:
                    item["fail_count"] += 1
                    if item["fail_count"] <= MAX_TRACK_FAIL:
                        active_trackers.append(item)
                    continue

                item["bbox"] = (x, y, w, h)
                item["fail_count"] = 0

                detection = {
                    "bbox": (x, y, w, h),
                    "polygon": bbox_to_polygon((x, y, w, h)),
                    "good_matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "template_name": template_name,
                    "score": 1.0,
                }
                detections.append(detection)
                active_trackers.append(item)

            self.trackers = active_trackers

            if len(self.trackers) == 0:
                self.need_detection = True

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


def process_video_file(input_video_path, output_video_path, template_dir=TEMPLATE_DIR):
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Input video олдсонгүй: {input_video_path}")

    runtime = create_runtime(template_dir=template_dir)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Видео нээгдсэнгүй: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Видео хэмжээ буруу байна.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Output video үүсгэж чадсангүй.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = runtime.process_frame(frame)
            writer.write(result["frame"])
    finally:
        cap.release()
        writer.release()

    return output_video_path


def main():
    input_video_path = "input.mp4"
    output_video_path = "output.mp4"

    process_video_file(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        template_dir=TEMPLATE_DIR,
    )
    print("Боловсруулалт дууслаа:", output_video_path)


if __name__ == "__main__":
    main()
