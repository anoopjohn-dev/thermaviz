"""
thermaviz example — live webcam-style viewer with MQTT alerting

Run: python examples/live_view.py [--sim]
     --sim  uses simulated thermal data (no hardware needed)
"""
import argparse
import json
import sys
import time

from thermaviz.camera import MLX90640, ThermalRenderer, AnomalyDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim",        action="store_true", help="Use simulated camera")
    parser.add_argument("--bus",        type=int, default=1, help="I2C bus number")
    parser.add_argument("--rate",       type=int, default=8, help="Refresh rate Hz")
    parser.add_argument("--palette",    default="ironbow",   help="ironbow|rainbow")
    parser.add_argument("--alert-temp", type=float, default=45.0, help="Alert threshold °C")
    parser.add_argument("--mqtt",       default=None, help="MQTT broker host")
    args = parser.parse_args()

    # Override I2C bus with None to force sim mode
    cam      = MLX90640(i2c_bus=(None if args.sim else args.bus),
                        refresh_rate=args.rate)
    renderer = ThermalRenderer(palette=args.palette, scale=12)
    detector = AnomalyDetector(threshold_delta=args.alert_temp - 25.0)

    mqtt_client = None
    if args.mqtt:
        try:
            import paho.mqtt.client as mqtt
            mqtt_client = mqtt.Client("thermaviz")
            mqtt_client.connect(args.mqtt, 1883)
            mqtt_client.loop_start()
            print(f"[thermaviz] MQTT connected → {args.mqtt}")
        except ImportError:
            print("[thermaviz] paho-mqtt not installed — MQTT disabled")

    try:
        import cv2
        use_cv = True
    except ImportError:
        use_cv = False
        print("[thermaviz] OpenCV not found — text-only output")

    print(f"[thermaviz] Starting  rate={args.rate} Hz  palette={args.palette}")
    print(f"[thermaviz] Alert threshold: {args.alert_temp:.1f} °C above ambient")
    print("Press Q to quit\n")

    frame_count = 0
    t_start     = time.time()

    for frame in cam.stream():
        frame_count += 1

        spots = detector.detect(frame)

        # MQTT publish
        if mqtt_client and spots:
            payload = json.dumps({
                "timestamp": frame.timestamp,
                "spots": [
                    {"col": s.col, "row": s.row,
                     "max_temp": round(s.max_temp, 2),
                     "area": s.area_pixels}
                    for s in spots
                ]
            })
            mqtt_client.publish("thermaviz/alerts", payload)

        # Console output every 10 frames
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - t_start)
            print(f"\r[thermaviz] {fps:.1f} fps  "
                  f"min={frame.min_temp:.1f}°C  "
                  f"max={frame.max_temp:.1f}°C  "
                  f"spots={len(spots)}   ", end="", flush=True)
            if spots:
                print(f"\n  ⚠  Hot spot: ({spots[0].col}, {spots[0].row})"
                      f"  {spots[0].max_temp:.1f}°C  "
                      f"area={spots[0].area_pixels}px")

        # OpenCV display
        if use_cv:
            img = renderer.render(frame)
            import cv2
            for s in spots:
                cx_ = int(s.centroid[0]) * renderer.scale
                cy_ = int(s.centroid[1]) * renderer.scale
                cv2.circle(img, (cx_, cy_), 8, (0, 0, 255), 2)
                cv2.putText(img, f"{s.max_temp:.0f}C",
                            (cx_ + 10, cy_), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 0, 255), 1)

            cv2.imshow("thermaviz", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if use_cv:
        import cv2; cv2.destroyAllWindows()
    if mqtt_client:
        mqtt_client.loop_stop()
    print()

if __name__ == "__main__":
    main()
