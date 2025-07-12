import numpy as np
import pyrealsense2 as rs
import cv2
import os
import keyboard  # pip install keyboard

def get_last_apple_index(data_dir):
    max_idx = 0
    for fname in os.listdir(data_dir):
        if fname.endswith("_rgb.jpg") and "apple" in fname:
            parts = fname.split("_")[0]  # apple03
            idx = int(parts.replace("apple", ""))
            max_idx = max(max_idx, idx)
    return max_idx

def capture_apples(num_apples=10, views_per_apple=4, output_dir="data", start_idx=1):
    os.makedirs(output_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print(f"[INFO] Bắt đầu chụp {num_apples} quả táo, mỗi quả {views_per_apple} góc.")
    print(f"[INFO] Nhấn Enter để chụp, nhấn 'q' để thoát.")

    try:
        #for apple_idx in range(start_idx, start_idx + num_apples):
        for apple_idx in range(start_idx, start_idx + num_apples):
            print(f"\n🍎 [APPLE {apple_idx}/{num_apples}]")

            for view_idx in range(1, views_per_apple + 1):
                print(f"  → Xoay quả táo. Chờ hình ổn định và nhấn Enter để chụp góc {view_idx}/{views_per_apple}...")

                while True:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)

                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    spatial = rs.spatial_filter()
                    temporal = rs.temporal_filter()
                    depth_frame = spatial.process(depth_frame)
                    depth_frame = temporal.process(depth_frame)
                    if not color_frame or not depth_frame:
                        continue

                    color = np.asanyarray(color_frame.get_data())
                    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

                    # Hiển thị RGB + Depth
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth, alpha=0.03),
                        cv2.COLORMAP_TURBO
                    )
                    depth_mask = (depth == 0).astype(np.uint8) * 255
                    depth_colormap[depth_mask > 0] = [0, 0, 255]
                    stacked = np.hstack((color, depth_colormap))

                    cv2.imshow("RGB + Depth View (Enter = Capture, Q = Quit)", stacked)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

                    if keyboard.is_pressed('enter'):
                        rgb_path = os.path.join(output_dir, f"apple{apple_idx:02d}_view{view_idx:02d}_rgb.jpg")
                        depth_path = os.path.join(output_dir, f"apple{apple_idx:02d}_view{view_idx:02d}_depth.npy")

                        cv2.imwrite(rgb_path, color)
                        np.save(depth_path, depth)

                        print(f"    ✅ Đã lưu {rgb_path} và {depth_path}")
                        while keyboard.is_pressed('enter'):
                            continue
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng theo yêu cầu người dùng.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Camera đã dừng.")

if __name__ == "__main__":
    num_apples = int(input("Bạn muốn chụp thêm bao nhiêu quả táo? "))
    start_idx = get_last_apple_index("data") + 1
    capture_apples(num_apples=num_apples, views_per_apple=4, output_dir="data", start_idx=start_idx)
