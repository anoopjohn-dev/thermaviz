"""
thermaviz — Real-time thermal camera analysis
thermaviz/camera.py

MLX90640 I²C driver + false-color renderer + hot-spot detector
"""
from __future__ import annotations

import struct
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# MLX90640 I²C constants
MLX90640_ADDR        = 0x33
REG_STATUS           = 0x8000
REG_CTRL1            = 0x800D
REG_RAM_BASE         = 0x0400
REG_EEPROM_BASE      = 0x2400
FRAME_ROWS, FRAME_COLS = 24, 32
FRAME_PIXELS           = FRAME_ROWS * FRAME_COLS


@dataclass
class ThermalFrame:
    pixels: np.ndarray           # shape (24, 32), dtype float32, °C
    ta:     float = 0.0          # ambient temperature °C
    timestamp: float = field(default_factory=time.time)

    @property
    def min_temp(self) -> float: return float(self.pixels.min())
    @property
    def max_temp(self) -> float: return float(self.pixels.max())
    @property
    def mean_temp(self) -> float: return float(self.pixels.mean())


@dataclass
class HotSpot:
    row: int
    col: int
    max_temp: float
    mean_temp: float
    area_pixels: int
    centroid: Tuple[float, float]


# ── MLX90640 I²C driver ──────────────────────────────────────────────────────
class MLX90640:
    """
    Pure-Python MLX90640 driver over smbus2.

    Usage::
        cam = MLX90640(i2c_bus=1, refresh_rate=8)
        for frame in cam.stream():
            print(frame.max_temp)
    """

    REFRESH_RATES = {1: 0b001, 2: 0b010, 4: 0b011, 8: 0b100,
                     16: 0b101, 32: 0b110, 64: 0b111}

    def __init__(self, i2c_bus: int = 1, refresh_rate: int = 8):
        try:
            import smbus2
            self._bus = smbus2.SMBus(i2c_bus)
        except ImportError:
            self._bus = None  # simulation mode

        self._rate  = refresh_rate
        self._eeprom: Optional[List[int]] = None
        self._params: Optional[dict]      = None

        if self._bus:
            self._read_eeprom()
            self._extract_params()
            self._set_refresh_rate(refresh_rate)

    def _read_eeprom(self):
        # Read 832 words from EEPROM
        self._eeprom = self._read_words(REG_EEPROM_BASE, 832)

    def _set_refresh_rate(self, hz: int):
        code  = self.REFRESH_RATES.get(hz, 0b100)
        ctrl  = self._read_words(REG_CTRL1, 1)[0]
        ctrl  = (ctrl & ~(0b111 << 7)) | (code << 7)
        self._write_word(REG_CTRL1, ctrl)

    def _read_words(self, addr: int, count: int) -> List[int]:
        result = []
        for i in range(count):
            data = self._bus.read_i2c_block_data(
                MLX90640_ADDR, (addr + i) >> 8, 3)
            result.append((data[1] << 8) | data[2])
        return result

    def _write_word(self, addr: int, value: int):
        self._bus.write_i2c_block_data(
            MLX90640_ADDR, addr >> 8,
            [addr & 0xFF, value >> 8, value & 0xFF])

    def _extract_params(self):
        """Parse calibration constants from EEPROM (simplified)."""
        ee = self._eeprom or []
        self._params = {
            "vdd25":  ((ee[0x33] & 0xFF00) >> 8) - 256 if ee else 0,
            "kvdd":   (ee[0x33] & 0x00FF) if ee else 0,
            "ptat25": ee[0x31] if ee else 0,
            "kvptat": ((ee[0x32] & 0xFC00) >> 10) if ee else 0,
            "ktptat": ((ee[0x32] & 0x03FF)) if ee else 0,
        }

    def read_frame(self) -> ThermalFrame:
        """Read one calibrated temperature frame."""
        if self._bus is None:
            return self._sim_frame()

        # Wait for new data
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            status = self._read_words(REG_STATUS, 1)[0]
            if status & 0x0008:
                break
            time.sleep(0.005)

        raw = self._read_words(REG_RAM_BASE, FRAME_PIXELS + 64)
        self._write_word(REG_STATUS, 0x0030)  # clear new-data flag

        pixels = self._compensate(raw[:FRAME_PIXELS], raw[FRAME_PIXELS:])
        ta     = self._calc_ta(raw[FRAME_PIXELS:])
        return ThermalFrame(pixels=pixels.reshape(FRAME_ROWS, FRAME_COLS), ta=ta)

    def _compensate(self, raw_pixels, aux) -> np.ndarray:
        """Apply gain, offset, and IR compensation (simplified)."""
        gain_ram = aux[48]
        gain_ee  = self._eeprom[48] if self._eeprom else 4096
        gain     = gain_ee / gain_ram if gain_ram else 1.0

        result = np.zeros(FRAME_PIXELS, dtype=np.float32)
        for i, p in enumerate(raw_pixels):
            pixel_s = (p - 32768) * gain if p > 32767 else p * gain
            # Simplified: real implementation applies per-pixel sensitivity
            result[i] = pixel_s * 0.001 + 25.0
        return result

    def _calc_ta(self, aux) -> float:
        vbe = aux[32] if len(aux) > 32 else 25000
        return (vbe - 25000) / 100.0 + 25.0

    def _sim_frame(self) -> ThermalFrame:
        """Simulate a thermal scene for testing without hardware."""
        t = time.time()
        base = np.full((FRAME_ROWS, FRAME_COLS), 22.0, dtype=np.float32)
        # Hot spot in centre that pulses
        cy, cx = FRAME_ROWS // 2, FRAME_COLS // 2
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                r = math.sqrt(dy**2 + dx**2)
                heat = max(0, (20.0 - r * 2.5)) * (0.8 + 0.2 * math.sin(t))
                ry, rx = cy + dy, cx + dx
                if 0 <= ry < FRAME_ROWS and 0 <= rx < FRAME_COLS:
                    base[ry, rx] += heat
        noise = np.random.normal(0, 0.15, base.shape).astype(np.float32)
        return ThermalFrame(pixels=base + noise, ta=22.0)

    def stream(self):
        """Generator that yields ThermalFrame continuously."""
        while True:
            yield self.read_frame()
            time.sleep(1.0 / self._rate)


# ── Renderer ─────────────────────────────────────────────────────────────────
PALETTES = {
    "ironbow": [
        (0,0,0),(22,0,51),(51,0,102),(89,0,128),(128,0,128),
        (178,0,102),(204,0,51),(229,51,0),(255,128,0),(255,204,0),(255,255,255)
    ],
    "rainbow": [
        (0,0,128),(0,0,255),(0,128,255),(0,255,255),(0,255,0),
        (255,255,0),(255,128,0),(255,0,0),(128,0,0)
    ],
}

class ThermalRenderer:
    def __init__(self, palette: str = "ironbow", scale: int = 10):
        self.palette = PALETTES.get(palette, PALETTES["ironbow"])
        self.scale   = scale

    def render(self, frame: ThermalFrame) -> np.ndarray:
        """Return a (rows*scale, cols*scale, 3) uint8 BGR image."""
        try:
            import cv2
        except ImportError:
            return np.zeros((FRAME_ROWS * self.scale, FRAME_COLS * self.scale, 3), np.uint8)

        t_min = frame.min_temp
        t_rng = max(frame.max_temp - t_min, 0.1)
        norm  = ((frame.pixels - t_min) / t_rng * 255).astype(np.uint8)

        lut   = self._build_lut()
        color = lut[norm]  # (24, 32, 3)
        big   = cv2.resize(color, (FRAME_COLS * self.scale, FRAME_ROWS * self.scale),
                           interpolation=cv2.INTER_LINEAR)
        # Overlay min/max/mean
        cv2.putText(big, f"Max:{frame.max_temp:.1f}C", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(big, f"Min:{frame.min_temp:.1f}C", (4, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return big

    def _build_lut(self) -> np.ndarray:
        lut = np.zeros((256, 3), dtype=np.uint8)
        pal = self.palette
        n   = len(pal) - 1
        for i in range(256):
            t   = i / 255.0 * n
            lo  = int(t); hi = min(lo + 1, n)
            f   = t - lo
            r   = int(pal[lo][0] * (1-f) + pal[hi][0] * f)
            g   = int(pal[lo][1] * (1-f) + pal[hi][1] * f)
            b   = int(pal[lo][2] * (1-f) + pal[hi][2] * f)
            lut[i] = [b, g, r]   # OpenCV BGR
        return lut


# ── Anomaly detector ─────────────────────────────────────────────────────────
class AnomalyDetector:
    """Detects hot-spots via adaptive threshold + connected-component labeling."""

    def __init__(self, threshold_delta: float = 15.0, min_area: int = 3):
        self.threshold_delta = threshold_delta
        self.min_area        = min_area

    def detect(self, frame: ThermalFrame) -> List[HotSpot]:
        threshold = frame.mean_temp + self.threshold_delta
        mask      = frame.pixels > threshold

        try:
            import cv2
            mask_u8 = mask.astype(np.uint8) * 255
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)

            spots = []
            for lbl in range(1, len(stats)):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < self.min_area:
                    continue
                region = frame.pixels[labels == lbl]
                spot   = HotSpot(
                    row=int(centroids[lbl][1]),
                    col=int(centroids[lbl][0]),
                    max_temp=float(region.max()),
                    mean_temp=float(region.mean()),
                    area_pixels=int(area),
                    centroid=(float(centroids[lbl][0]), float(centroids[lbl][1]))
                )
                spots.append(spot)
            return sorted(spots, key=lambda s: s.max_temp, reverse=True)
        except ImportError:
            return []
