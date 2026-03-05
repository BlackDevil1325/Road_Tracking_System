import cv2
import numpy as np
import argparse
import os


def region_of_interest(img, vertices):
    """Return the portion of the image defined by the polygon `vertices`.

    `img` can be either a grayscale or color image.  `vertices` should be an
    array of shape (1, N, 2) or (N, 2) containing integer vertex coordinates.
    """
    mask = np.zeros_like(img)
    # match mask color depending on image channels
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
    else:
        match_mask_color = 255

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def detect_edges(img, low_threshold=50, high_threshold=150):
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply Gaussian smoothing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detector
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    return edges


def detect_lines(edges):
    # Hough transform for line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                           minLineLength=50, maxLineGap=150)
    return lines


def average_slope_intercept(lines):
    left_lines = []
    right_lines = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        intercept = y1 - slope * x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    
    left_avg = np.average(left_lines, axis=0) if left_lines else None
    right_avg = np.average(right_lines, axis=0) if right_lines else None
    
    return left_avg, right_avg


def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, y1), (x2, y2))


def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    output = np.copy(img)
    if lines is None:
        return output
    for line in lines:
        if line is not None:
            cv2.line(output, line[0], line[1], color, thickness)
    return output


def detect_potholes(img, area_threshold=500):
    """A very simple pothole/hole detector using dark spot thresholding.

    This is not production quality but serves to highlight dark regions that
    might represent potholes, puddles, oil stains, etc.  The threshold and area
    parameters can be tuned or replaced with a ML model down the road.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert binary threshold - dark regions become white
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return output


def process_frame(frame):
    h, w = frame.shape[:2]
    vertices = np.array([[(0, h), (w // 2, h // 2), (w, h)]], dtype=np.int32)
    edges = detect_edges(frame)
    roi = region_of_interest(edges, vertices)
    lines = detect_lines(roi)
    left_avg, right_avg = average_slope_intercept(lines)
    left_line = make_line_points(h, h // 2 + 50, left_avg)
    right_line = make_line_points(h, h // 2 + 50, right_avg)
    lane_lines = [left_line, right_line]
    line_img = draw_lines(frame, lane_lines)
    pothole_img = detect_potholes(frame)
    # Combine both detections
    combined = cv2.addWeighted(line_img, 0.8, pothole_img, 0.2, 0)
    return combined


def process_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return process_frame(img)


def process_video(input_path, output_path=None):
    cap = cv2.VideoCapture(input_path if input_path != 'webcam' else 0)
    if not cap.isOpened():
        raise ValueError("Could not open video source")
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        if output_path:
            out.write(processed)
        else:
            cv2.imshow('Lane Detection', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Lane & pothole detector")
    parser.add_argument("-i", "--input", required=True,
                        help="path to input image/video or 'webcam' for live")
    parser.add_argument("-o", "--output", default=None,
                        help="output file prefix (for images) or video file (for video)")
    parser.add_argument("--video", action="store_true",
                        help="process as video")
    args = parser.parse_args()

    if args.input == 'webcam' or args.video or args.input.endswith(('.mp4', '.avi', '.mov')):
        process_video(args.input, args.output)
    else:
        # Assume image
        processed = process_image(args.input)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_file = f"{args.output or 'output'}_{base}.png"
        cv2.imwrite(out_file, processed)
        print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
