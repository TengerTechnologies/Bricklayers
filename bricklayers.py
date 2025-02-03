# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (c) [2025] [Roman Tenger]

from datetime import datetime
import mmap
import math
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.strtree import STRtree
from shapely.validation import make_valid
import logging
import os
import re
from enum import Enum
import argparse
from tempfile import NamedTemporaryFile
from pathlib import Path
from statistics import median
import logging.handlers


class PrinterType(Enum):
    BAMBU = "bambu"
    PRUSA = "prusa"


class PerimeterType(Enum):
    OUTER = "outer"
    INNER = "inner"
    SOLID = "solid"


class GCodeProcessor:
    def __init__(
        self,
        layer_height=None,
        extrusion_multiplier=1.0,
        first_layer_multiplier=None,
        last_layer_multiplier=None,
        min_distance=0.1,
        max_intersection_area=0.5,
        simplify_tolerance=0.03,
        z_speed=None,
        log_level=logging.INFO,
        max_area_deviation=0.06,  # More conservative default
        hausdorff_multiplier=0.3,  # Balanced default
        max_z_speed=6000.0,
        min_z_move_time=0.5,
        safe_z_distance=1.0,
        min_perimeter_points=3,
        full_layer_shifts=True,
        min_feature_size=0.2,  # More practical default
        critical_angle=25,
        precision_mode="auto",
    ):
        self.extrusion_multiplier = extrusion_multiplier
        self.first_layer_multiplier = (
            first_layer_multiplier
            if first_layer_multiplier is not None
            else 1.5 * extrusion_multiplier
        )
        self.last_layer_multiplier = (
            last_layer_multiplier
            if last_layer_multiplier is not None
            else 0.5 * extrusion_multiplier
        )
        self.min_distance = min_distance
        self.max_intersection_area = max_intersection_area
        self.simplify_tolerance = simplify_tolerance
        self.current_layer = 0
        self.current_z = 0.0
        self.total_layers = 0
        self.printer_type = None
        self.layer_height = layer_height
        self.z_shift = None
        self.shifted_blocks = 0
        self.z_speed = z_speed
        self.total_extrusion_adjustments = 0
        self.processing_start = None
        self.layer_times = []
        self.simplified_features = 0
        self.failed_simplifications = 0
        self.decoding_errors = 0
        self.max_area_deviation = max_area_deviation
        self.hausdorff_multiplier = hausdorff_multiplier
        self.max_z_speed = max_z_speed
        self.min_z_move_time = min_z_move_time
        self.safe_z_distance = safe_z_distance
        self.min_perimeter_points = min_perimeter_points
        self.full_layer_shifts = full_layer_shifts
        self.layer_shift_pattern = []
        self.min_feature_size = min_feature_size
        self.critical_angle = critical_angle
        self.precision_mode = precision_mode
        self.skipped_small_features = 0
        self.closure_tolerance = 0.001
        self.travel_threshold = 0.2  # mm
        self.closed_paths = 0
        self.open_paths = 0

        # Regex patterns
        self.re_z = re.compile(r"Z([\d.]+)")
        self.re_f = re.compile(r"F([\d.]+)")
        self.re_e = re.compile(r"E([\d.]+)")
        self.re_e_sub = re.compile(r"E[\d.]+")
        self.re_x = re.compile(r"X([\d.]+)")
        self.re_y = re.compile(r"Y([\d.]+)")

        self._configure_logging(log_level)

    def is_path_closed(self, path):
        if len(path) < 2:
            return False
        dx = abs(path[0][0] - path[-1][0])
        dy = abs(path[0][1] - path[-1][1])
        return dx < self.closure_tolerance and dy < self.closure_tolerance

    def _calculate_z_speed(self, delta_z):
        if delta_z <= self.safe_z_distance:
            required_speed = (delta_z / self.min_z_move_time) * 60
            return min(self.max_z_speed, required_speed)
        return self.max_z_speed

    def decode_line(self, line_bytes):
        try:
            return line_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            self.decoding_errors += 1
            self.logger.warning(f"Encoding error: {str(e)}")
            return line_bytes.decode("utf-8", errors="replace")

    def _is_travel_move(self, path):
        """Detect short moves that aren't real perimeters"""
        if len(path) < 2:
            return True
        return LineString(path).length < self.travel_threshold

    def _configure_logging(self, log_level):
        self.logger = logging.getLogger("Bricklayers")
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            rotating_handler = logging.handlers.RotatingFileHandler(
                "bricklayers.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            rotating_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(rotating_handler)
            self.logger.addHandler(console_handler)

    def detect_printer_type(self, file_path):
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    decoded_line = self.decode_line(line)
                    if "; FEATURE:" in decoded_line:
                        return PrinterType.BAMBU.value
                    if ";TYPE:" in decoded_line:
                        return PrinterType.PRUSA.value
                return PrinterType.PRUSA.value

    def detect_layer_height(self, file_path):
        z_values = []
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    decoded_line = self.decode_line(line)
                    if re.match(r"^G[01] Z\d", decoded_line.strip()):
                        z_match = self.re_z.search(decoded_line)
                        if z_match:
                            z_values.append(float(z_match.group(1)))

        layer_heights = [
            z_values[i + 1] - z_values[i]
            for i in range(len(z_values) - 1)
            if z_values[i + 1] > z_values[i]
        ]
        if not layer_heights:
            raise ValueError("No positive Z moves detected")
        median_height = median(layer_heights)
        if not 0.04 <= median_height <= 0.6:
            raise ValueError(f"Implausible layer height: {median_height:.2f}mm")
        return median_height

    def detect_z_speed(self, file_path):
        z_speeds = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("G0 ", "G1 ")) and "Z" in line:
                    f_match = self.re_f.search(line)
                    if f_match:
                        z_speeds.append(float(f_match.group(1)))
        return median(z_speeds) if z_speeds else 6000.0

    def parse_perimeter_paths(self, layer_lines):
        perimeter_paths = []
        current_type = None
        current_path = []
        current_lines = []

        for line_idx, line in enumerate(layer_lines):
            # Handle layer change markers first
            if any(marker in line for marker in [";LAYER:", "; CHANGE_LAYER"]):
                if current_path:
                    self.logger.debug(
                        f"Clearing buffer at layer change (line {line_idx}): {len(current_path)} points"
                    )
                    current_path = []
                    current_lines = []
                continue

            # Printer-specific type detection
            if self.printer_type == PrinterType.BAMBU.value:
                if "; FEATURE" in line:
                    current_type = (
                        PerimeterType.INNER if "Inner" in line else PerimeterType.OUTER
                    )
            elif self.printer_type == PrinterType.PRUSA.value:
                if ";TYPE:" in line:
                    current_type = (
                        PerimeterType.OUTER
                        if "External perimeter" in line
                        else PerimeterType.INNER
                    )

            # Path collection logic
            if line.startswith("G1") and "X" in line and "Y" in line:
                x_match = self.re_x.search(line)
                y_match = self.re_y.search(line)
                if x_match and y_match:
                    current_path.append(
                        (float(x_match.group(1)), float(y_match.group(1)))
                    )
                    current_lines.append(line_idx)
            else:
                if current_path:
                    try:
                        # Early rejection of invalid paths
                        if len(current_path) < 2:
                            self.logger.debug(
                                f"Skipping micro-path at line {line_idx} (only {len(current_path)} points)"
                            )
                            current_path = []
                            continue

                        # Filter travel moves
                        if self._is_travel_move(current_path):
                            self.logger.debug(
                                f"Ignoring travel move at line {line_idx}"
                            )
                            current_path = []
                            continue

                        # Closure handling
                        is_closed = self.is_path_closed(current_path)
                        if is_closed:
                            self.closed_paths += 1
                            if len(current_path) > 3:
                                current_path = current_path[
                                    :-1
                                ]  # Remove duplicate closure
                        else:
                            self.open_paths += 1
                            current_path.append(current_path[0])  # Force closure

                        # Post-closure validation
                        if len(current_path) < 3:
                            self.logger.debug(
                                f"Rejecting path at line {line_idx} - insufficient points after closure"
                            )
                            current_path = []
                            continue

                        # Geometry processing
                        poly = make_valid(Polygon(current_path))
                        if isinstance(poly, LineString):
                            poly = poly.buffer(0.15)
                        elif isinstance(poly, MultiPolygon):
                            poly = max(poly.geoms, key=lambda p: p.area)

                        # Precision processing
                        original_area = poly.area
                        min_feature_area = (self.min_feature_size**2) * 4
                        if original_area > 0 and self.precision_mode != "disabled":
                            if self.precision_mode == "auto":
                                # Auto-detect high precision needs
                                if (
                                    self.layer_height < 0.1
                                    or self.min_feature_size < 0.2
                                ):
                                    precision_class = "high_precision"
                                    self.logger.debug(
                                        "Auto-detected high precision requirements"
                                    )
                                else:
                                    precision_class = "balanced"
                            else:
                                precision_class = self.precision_mode

                            # Layer-height adaptive parameters
                            if self.layer_height < 0.15:
                                self.hausdorff_multiplier *= 0.8
                                self.max_area_deviation *= 1.2

                            precision_params = {
                                "high_precision": {"hausdorff": 0.1, "max_dev": 0.015},
                                "balanced": {"hausdorff": 0.2, "max_dev": 0.06},
                                "draft": {"hausdorff": 0.3, "max_dev": 0.04},
                            }[precision_class]

                            if original_area > min_feature_area:
                                dynamic_tolerance = max(
                                    math.sqrt(original_area)
                                    * self.hausdorff_multiplier,
                                    self.simplify_tolerance,
                                )
                                # Add layer-based adaptation
                                dynamic_tolerance *= (
                                    1 + self.current_layer / self.total_layers
                                )

                                simplified = poly.simplify(
                                    dynamic_tolerance, preserve_topology=True
                                )
                                area_deviation = (
                                    abs(simplified.area - original_area) / original_area
                                )

                                if area_deviation <= precision_params["max_dev"]:
                                    poly = simplified
                                    self.simplified_features += 1
                                else:
                                    self.failed_simplifications += 1
                            else:
                                self.skipped_small_features += 1

                        perimeter_paths.append((poly, current_lines, current_type))

                    except Exception as e:
                        self.failed_simplifications += 1
                        self.logger.debug(
                            f"Path processing failed at line {line_idx}: {str(e)}"
                        )
                    finally:
                        current_path = []
                        current_lines = []
                        current_type = None

        # Handle any remaining path in buffer
        if current_path:
            self.logger.warning(
                f"Unprocessed path remaining at layer end: {len(current_path)} points"
            )

        return perimeter_paths

    def get_memory_usage(self):
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            try:
                import resource

                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            except:
                return 0.0

    def classify_perimeters(self, perimeter_paths):
        classified = {"outer": [], "inner": []}
        unclassified = []

        for poly, lines, ptype in perimeter_paths:
            if ptype == PerimeterType.OUTER:
                classified["outer"].append((poly, lines))
            elif ptype == PerimeterType.INNER:
                classified["inner"].append((poly, lines))
            else:
                unclassified.append((poly, lines))

        if unclassified and (classified["outer"] or classified["inner"]):
            outer_polys = [p for p, _ in classified["outer"]]
            tree = STRtree(outer_polys) if outer_polys else None

            for poly, lines in unclassified:
                is_inner = False
                if tree:
                    candidate_indices = tree.query(poly)
                    candidates = [outer_polys[i] for i in candidate_indices]

                    containing_outers = [
                        op for op in candidates if op.contains(poly.buffer(-0.01))
                    ]

                    if containing_outers:
                        parent_outer = max(
                            containing_outers, key=lambda op: op.intersection(poly).area
                        )
                        area_ratio = poly.area / parent_outer.area
                        if area_ratio < 0.75:
                            intersection = parent_outer.intersection(poly).area
                            if intersection / poly.area >= 0.95:
                                is_inner = True
                    else:
                        nearest_idx = tree.nearest(poly)
                        nearest = outer_polys[nearest_idx]
                        distance = poly.distance(nearest)
                        if distance < self.min_distance * 0.5:
                            if poly.area / nearest.area < 0.25:
                                is_inner = True

                classified["inner" if is_inner else "outer"].append((poly, lines))

        return classified["outer"], classified["inner"]

    def process_layer(self, layer_lines, layer_num, total_layers):
        layer_start = datetime.now()
        perimeter_paths = self.parse_perimeter_paths(layer_lines)
        outer, inner = self.classify_perimeters(perimeter_paths)

        shift_layer = len(inner) > 0
        if shift_layer:
            self.layer_shift_pattern.append(1)
            self.shifted_blocks += 1
        else:
            self.layer_shift_pattern.append(0)

        processed = []
        current_z = None
        shift_amount = self.layer_height * 0.5

        for line in layer_lines:
            original_line = line

            if self.full_layer_shifts and line.startswith("G1 Z"):
                z_match = self.re_z.search(line)
                if z_match:
                    current_z = float(z_match.group(1))
                    if shift_layer:
                        new_z = current_z + shift_amount
                        line = f"G1 Z{new_z:.3f} F{self.z_speed}\n"
                        processed.append(line)
                        continue

            if "E" in line and (shift_layer or not self.full_layer_shifts):
                line = self._adjust_extrusion(line, layer_num == total_layers - 1)

            processed.append(line)

        self.layer_times.append((datetime.now() - layer_start).total_seconds())
        shift_reason = (
            "Shift applied: Found {0} inner perimeters".format(len(inner))
            if shift_layer
            else "No shift: No inner perimeters detected"
        )
        self.logger.info(
            f"Layer {layer_num+1}: {shift_reason} | "
            f"Processing mode: {self._get_current_precision_mode()}"
        )
        return processed

    def _get_current_precision_mode(self):
        if self.precision_mode == "auto":
            return (
                "auto-detected high precision"
                if (self.layer_height < 0.1 or self.min_feature_size < 0.2)
                else "auto-detected standard"
            )
        return self.precision_mode

    def _adjust_extrusion(self, line, is_last_layer):
        e_match = self.re_e.search(line)
        if e_match:
            e_value = float(e_match.group(1))
            if self.current_layer == 0:
                new_e = e_value * self.first_layer_multiplier
                comment = "first layer"
            elif is_last_layer:
                new_e = e_value * self.last_layer_multiplier
                comment = "last layer"
            else:
                new_e = e_value * self.extrusion_multiplier
                comment = "adjusted extrusion"
            self.total_extrusion_adjustments += 1
            return self.re_e_sub.sub(f"E{new_e:.5f}", line).rstrip() + f" ; {comment}\n"
        return line

    def process_gcode(self, input_file, is_bgcode=False):
        self.processing_start = datetime.now()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_handler = logging.FileHandler(
            os.path.join(script_dir, f"bricklayers_{current_time}.log"),
            encoding="utf-8",
        )
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(log_handler)

        try:
            input_path = Path(input_file)
            if is_bgcode:
                input_path = input_path.with_suffix(".gcode")
                os.system(f"bgcode decode {input_file} -o {input_path}")

            self.printer_type = self.detect_printer_type(input_path)
            self.layer_height = self.layer_height or self.detect_layer_height(
                input_path
            )
            self.z_shift = self.layer_height * 0.5
            self.z_speed = self.z_speed or self.detect_z_speed(input_path)

            with open(input_path, "r+") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    self.total_layers = sum(
                        1 for line in iter(mm.readline, b"") if b"G1 Z" in line
                    )
                    mm.seek(0)

                    with NamedTemporaryFile(mode="w", delete=False) as tmp_file:
                        layer_buffer = []
                        current_layer = 0

                        for line in iter(mm.readline, b""):
                            decoded_line = self.decode_line(line)
                            layer_buffer.append(decoded_line)

                            if (
                                "G1 Z" in decoded_line
                                or "; CHANGE_LAYER" in decoded_line
                            ):
                                processed = self.process_layer(
                                    layer_buffer, current_layer, self.total_layers
                                )
                                tmp_file.writelines(processed)
                                current_layer += 1
                                layer_buffer = []

                        if layer_buffer:
                            processed = self.process_layer(
                                layer_buffer, current_layer, self.total_layers
                            )
                            tmp_file.writelines(processed)

                        os.replace(tmp_file.name, input_path)

            final_report = (
                "════════════════════ Processing Complete ════════════════════\n"
                f"Total Layers: {self.total_layers} | Shifted Layers: {sum(self.layer_shift_pattern)}\n"
                f"Total Decoding Errors: {self.decoding_errors}\n"
                f"Total Z-Shifts Applied: {self.shifted_blocks}\n"
                f"Extrusion Adjustments: {self.total_extrusion_adjustments}\n"
                f"Simplified Features: {self.simplified_features}\n"
                f"Skipped Small Features: {self.skipped_small_features}\n"
                f"Failed Simplifications: {self.failed_simplifications}\n"
                f"Average Layer Time: {sum(self.layer_times)/len(self.layer_times)*1000:.2f}ms\n"
                f"Total Processing Time: {(datetime.now()-self.processing_start).total_seconds()*1000:.2f}ms\n"
                f"Peak Memory Usage: {self.get_memory_usage():.2f}MB\n"
                f"Safety Settings | Max Z Speed: {self.max_z_speed}mm/min | Min Move Time: {self.min_z_move_time}s\n"
                f"Precision Mode: {self.precision_mode} | Min Feature: {self.min_feature_size}mm | Critical Angle: {self.critical_angle}°\n"
                f"Valid Paths: Closed={self.closed_paths} | Open={self.open_paths}\n"
                "═════════════════════════════════════════════════════════════"
            )
            self.logger.info(final_report)

        finally:
            self.logger.removeHandler(log_handler)
            log_handler.close()


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive G-code Optimizer Leveraging "Bricklayers" for Structural Printing'
    )
    parser.add_argument("inputFile", help="Input G-code file path")
    parser.add_argument("-layerHeight", type=float, help="Manual layer height override")
    parser.add_argument("-extrusionMultiplier", type=float, default=1.0, help="...")
    parser.add_argument("-simplifyTolerance", type=float, default=0.03, help="...")
    parser.add_argument("-bgcode", action="store_true", help="...")
    parser.add_argument(
        "--logLevel", choices=["DEBUG", ...], default="INFO", help="..."
    )
    parser.add_argument("-firstLayerMultiplier", type=float, default=None, help="...")
    parser.add_argument("-lastLayerMultiplier", type=float, default=None, help="...")
    parser.add_argument("-zSpeed", type=float, default=None, help="...")
    parser.add_argument("-maxAreaDev", type=float, default=0.06, help="...")
    parser.add_argument("-hausdorffMult", type=float, default=0.3, help="...")
    parser.add_argument("-maxZSpeed", type=float, default=6000.0, help="...")
    parser.add_argument("-minZMoveTime", type=float, default=0.5, help="...")
    parser.add_argument("-safeZDistance", type=float, default=1.0, help="...")
    parser.add_argument(
        "--perPerimeterShifts",
        action="store_false",
        dest="fullLayerShifts",
        default=True,
        help="...",
    )
    parser.add_argument("-minDetail", type=float, default=0.2, help="...")
    parser.add_argument("-criticalAngle", type=float, default=25, help="...")
    parser.add_argument("-precision", choices=["auto", ...], default="auto", help="...")
    parser.add_argument("-minPerimeterPoints", type=int, default=3, help="...")

    args = parser.parse_args()

    processor = GCodeProcessor(
        log_level=getattr(logging, args.logLevel),
        layer_height=args.layerHeight,
        first_layer_multiplier=args.firstLayerMultiplier,
        last_layer_multiplier=args.lastLayerMultiplier,
        extrusion_multiplier=args.extrusionMultiplier,
        simplify_tolerance=args.simplifyTolerance,
        z_speed=args.zSpeed,
        max_area_deviation=args.maxAreaDev,
        hausdorff_multiplier=args.hausdorffMult,
        max_z_speed=args.maxZSpeed,
        min_z_move_time=args.minZMoveTime,
        safe_z_distance=args.safeZDistance,
        full_layer_shifts=args.fullLayerShifts,
        min_feature_size=args.minDetail,
        critical_angle=args.criticalAngle,
        precision_mode=args.precision,
        min_perimeter_points=args.minPerimeterPoints,
    )
    processor.process_gcode(args.inputFile, args.bgcode)


if __name__ == "__main__":
    main()
