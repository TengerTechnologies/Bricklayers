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


class FeatureType(Enum):
    NORMAL = "normal"
    BRIDGE = "bridge"
    OVERHANG = "overhang"


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
        nozzle_diameter=0.4,
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
        self.nozzle_diameter = nozzle_diameter
        self.open_paths = 0
        self.extrusion_mode = "absolute"  # Default assumption
        self.last_e_value = 0.0
        self.layer_parity = 0
        self.current_z = 0.0  # 👈 Track actual printed Z
        self.brick_height_layers = 2  # Layers per brick row
        self.brick_z_shift = (
            self.layer_height * 0.5
        )  # Vertical shift between brick rows
        self.brick_phase = 0  # Track brick rows

        # Regex patterns
        self.re_z = re.compile(r"Z([\d.]+)")
        self.re_f = re.compile(r"F([\d.]+)")
        self.re_e = re.compile(r"E([\d.]+)")
        self.re_e_sub = re.compile(r"E[\d.]+")
        # Capture both full and decimal-first numbers
        self.re_x = re.compile(r"X(-?\d*\.?\d+)")
        self.re_y = re.compile(r"Y(-?\d*\.?\d+)")

        self._configure_logging(log_level)

    def shift_coordinate(match, shift):
        try:
            val = float(match.group(1))
            return f"{match.group(0)[0]}{val + shift:.5f}"
        except:
            return match.group(0)

    def is_path_closed(self, path):
        if len(path) < 4:  # Need at least 4 points to form a closed polygon
            return False
        start = path[0]
        end = path[-1]
        return (
            abs(start[0] - end[0]) < self.closure_tolerance
            and abs(start[1] - end[1]) < self.closure_tolerance
        )

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
        """Detect short moves that aren't real perimeters using buffered length check"""
        if len(path) < 2:
            return True
        ls = LineString(path)
        return ls.length < self.travel_threshold or ls.buffer(0.05).area < (
            self.nozzle_diameter**2
        )

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

    def calculate_layer_bounds(self, polygons):
        """Calculate axis-aligned bounding box for entire layer"""
        if not polygons:
            return None
        minx = min(p.bounds[0] for p in polygons)
        miny = min(p.bounds[1] for p in polygons)
        maxx = max(p.bounds[2] for p in polygons)
        maxy = max(p.bounds[3] for p in polygons)
        return (minx, miny, maxx, maxy)

    def is_boundary_proximal(self, polygon, layer_bounds):
        """Check if polygon is near layer edge with nozzle-based tolerance"""
        if not layer_bounds:
            return False

        tol = self.nozzle_diameter * 2.5  # 2.5x nozzle width tolerance
        poly_bounds = polygon.bounds

        return any(
            [
                abs(poly_bounds[0] - layer_bounds[0]) < tol,  # minx
                abs(poly_bounds[1] - layer_bounds[1]) < tol,  # miny
                abs(poly_bounds[2] - layer_bounds[2]) < tol,  # maxx
                abs(poly_bounds[3] - layer_bounds[3]) < tol,  # maxy
            ]
        )

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

    @staticmethod
    def detect_nozzle_diameter(file_path):
        default_nozzle = 0.4
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("; nozzle_diameter = "):
                        return float(line.split("=")[1].strip())
                    if "Nozzle diameter" in line:
                        return float(re.search(r"\d+\.?\d*", line).group())
        except Exception as e:
            print(f"Nozzle detection failed: {str(e)}")

        return default_nozzle

    def _detect_infill_lines(self, layer_lines):
        """Precision infill detection for PrusaSlicer types"""
        infill_markers = {
            PrinterType.BAMBU.value: [
                "; FEATURE:Internal infill",
                "; FEATURE:Sparse infill",
            ],
            PrinterType.PRUSA.value: [
                ";TYPE:Solid infill",
                ";TYPE:Internal infill",
                ";TYPE:Top solid infill",
                ";TYPE:Bridge infill",
            ],
        }

        return {
            idx
            for idx, line in enumerate(layer_lines)
            if any(marker in line for marker in infill_markers[self.printer_type])
        }

    def _get_perimeter_indices(self, outer_perimeters, inner_perimeters):
        """Get line indices from geometrically classified perimeters"""
        return {
            line_idx
            for perim_set in [outer_perimeters, inner_perimeters]
            for poly_info in perim_set  # Each poly_info is (polygon, line_indices, perimeter_type)
            for line_idx in poly_info[1]  # Index 1 contains the original line numbers
        }

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

    def classify_perimeters(self, perimeter_paths, use_explicit_types=False):
        """Enhanced hierarchical perimeter classification with boundary awareness"""
        # Extract polygons and calculate layer bounds
        all_polygons = [poly for poly, _, _ in perimeter_paths]
        layer_bounds = self.calculate_layer_bounds(all_polygons)

        # Sort polygons by descending area for hierarchical processing
        sorted_polys = sorted(perimeter_paths, key=lambda x: x[0].area, reverse=True)

        classified = {"outer": [], "inner": [], "unclassified": []}
        hierarchy_tree = STRtree([poly for poly, _, _ in sorted_polys])

        layer_complex = False  # Track complexity for entire layer
        for poly, lines, ptype in sorted_polys:
            # First check explicit type from slicer
            if use_explicit_types:
                if ptype == PerimeterType.OUTER:
                    classified["outer"].append((poly, lines))
                    continue
                if ptype == PerimeterType.INNER:
                    classified["inner"].append((poly, lines))
                    continue

            # Automatic classification logic
            is_boundary = self.is_boundary_proximal(poly, layer_bounds)
            parents = hierarchy_tree.query(poly, predicate="contains")
            containment_depth = len(parents)

            # Classification rules
            if is_boundary:
                perimeter_type = PerimeterType.OUTER
            elif containment_depth % 2 == 0:  # Even containment = outer
                perimeter_type = PerimeterType.OUTER
            else:
                perimeter_type = PerimeterType.INNER

            # Secondary validation
            if perimeter_type == PerimeterType.OUTER:
                # Verify not contained by other outer perimeters
                existing_outers = [p for p, _ in classified["outer"]]
                outer_tree = STRtree(existing_outers)
                containing_outers = outer_tree.query(poly, predicate="contains")
                if any(containing_outers):
                    perimeter_type = PerimeterType.INNER

            # Complex feature detection
            complex_detected = False
            if perimeter_type == PerimeterType.OUTER:
                # Check for nested outer perimeters
                if containment_depth > 0:
                    complex_detected = True
                    self.logger.debug(
                        f"Nested outer perimeter detected at layer {self.current_layer}"
                    )

                # Check for disconnected outer features
                if len(classified["outer"]) >= 1 and not any(
                    p.contains(poly) for p, _ in classified["outer"]
                ):
                    complex_detected = True
                    self.logger.debug(
                        f"Multiple outer perimeters detected at layer {self.current_layer}"
                    )

            classified[
                "outer" if perimeter_type == PerimeterType.OUTER else "inner"
            ].append((poly, lines))

        # Log classification results
        if sorted_polys:  # Only log if we processed perimeters
            self.logger.debug(
                f"Layer {self.current_layer} classification: "
                f"Outer={len(classified['outer'])} "
                f"Inner={len(classified['inner'])} "
                f"Complex={layer_complex} "
                f"Unclassified={len(classified['unclassified'])}"
            )

        return classified["outer"], classified["inner"]

    def should_stagger(self, perimeter_type, feature_type):
        # Never stagger outer perimeters or bridges
        if perimeter_type == PerimeterType.OUTER:
            return False
        if feature_type == FeatureType.BRIDGE:
            return False
        return True

    def process_layer(self, layer_lines, layer_num, total_layers):
        self.current_layer = layer_num  # Critical for all layer-bound operations
        layer_start = datetime.now()
        processed = []

        # Calculate brick layering pattern (2-layer cycle)
        self.brick_phase = (layer_num // self.brick_height_layers) % 2
        z_shift_amount = self.brick_z_shift if self.brick_phase else 0

        # Parse and classify geometry
        try:
            perimeter_data = self.parse_perimeter_paths(layer_lines)
            outer_perimeters, inner_perimeters = self.classify_perimeters(
                perimeter_data,
                use_explicit_types=(self.printer_type == PrinterType.BAMBU.value),
            )
        except Exception as e:
            self.logger.error(f"Layer classification failed: {str(e)}")
            return layer_lines  # Fail-safe return of original lines

        # Feature detection
        infill_lines = self._detect_infill_lines(layer_lines)
        perimeter_indices = self._get_perimeter_indices(
            outer_perimeters, inner_perimeters
        )

        # Split layer content
        perimeter_section = [
            line for idx, line in enumerate(layer_lines) if idx in perimeter_indices
        ]
        infill_section = [
            line for idx, line in enumerate(layer_lines) if idx in infill_lines
        ]

        # Construct processed output
        processed.extend(perimeter_section)

        # Apply Z-shift with collision protection
        if z_shift_amount > 0:
            new_z = self.current_z + z_shift_amount
            processed.append(f"G1 Z{new_z:.3f} F{self.z_speed} ; BRICK_SHIFT\n")
            self.shifted_blocks += 1
            self.layer_shift_pattern.append(1)
            self.logger.debug(
                f"Applied Z-shift: {z_shift_amount}mm at layer {layer_num}"
            )
        else:
            new_z = self.current_z
            processed.append(f"; NO_SHIFT (phase {self.brick_phase})\n")
            self.layer_shift_pattern.append(0)

        processed.extend(infill_section)

        # Update state tracking
        self.current_z = new_z

        # Diagnostic logging
        layer_duration = (datetime.now() - layer_start).total_seconds() * 1000
        self.layer_times.append(layer_duration)

        self.logger.info(
            f"Layer {layer_num} [{'SHIFT' if z_shift_amount else '    '}] | "
            f"Outer: {len(outer_perimeters):02d} | "
            f"Inner: {len(inner_perimeters):02d} | "
            f"Staggered: {len(infill_lines):03d} | "
            f"Z: {new_z:06.2f}mm | "
            f"Time: {layer_duration:05.1f}ms"
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Phase: {self.brick_phase} | "
                f"Total shifts: {self.shifted_blocks} | "
                f"Memory: {self.get_memory_usage():.1f}MB | "
                f"Valid paths: {self.closed_paths} closed, {self.open_paths} open"
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

    def _get_multiplier(self, is_last_layer):
        """Determine extrusion multiplier based on layer position"""
        if self.current_layer == 0:
            return self.first_layer_multiplier
        elif is_last_layer:
            return self.last_layer_multiplier
        return self.extrusion_multiplier

    def _get_comment(self, is_last_layer):
        """Generate appropriate comment for extrusion adjustments"""
        if self.current_layer == 0:
            return "first layer"
        elif is_last_layer:
            return "last layer"
        return "adjusted extrusion"

    def _adjust_extrusion(self, line, is_last_layer):
        e_match = self.re_e.search(line)
        if e_match:
            e_value = float(e_match.group(1))
            multiplier = self._get_multiplier(is_last_layer)
            scaled_e = e_value * multiplier
            return (
                self.re_e_sub.sub(f"E{scaled_e:.5f}", line).rstrip()
                + f" ; {self._get_comment(is_last_layer)}\n"
            )
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
            self.z_shift = self.layer_height * 0.25
            self.z_speed = self.z_speed or self.detect_z_speed(input_path)

            with open(input_path, "r+") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Detect layer change comment based on printer type
                    layer_change_marker = {
                        PrinterType.BAMBU.value: "; CHANGE_LAYER",
                        PrinterType.PRUSA.value: ";LAYER_CHANGE",
                    }[self.printer_type]

                    # Count total layers based on layer change comments
                    self.total_layers = sum(
                        1
                        for line in iter(mm.readline, b"")
                        if layer_change_marker in self.decode_line(line)
                    )
                    mm.seek(0)

                    with NamedTemporaryFile(mode="w", delete=False) as tmp_file:
                        layer_buffer = []
                        current_layer = 0

                        for line in iter(mm.readline, b""):
                            decoded_line = self.decode_line(line)

                            # Track extrusion mode changes
                            if "M82" in decoded_line:
                                self.extrusion_mode = "absolute"
                            elif "M83" in decoded_line:
                                self.extrusion_mode = "relative"

                            layer_buffer.append(decoded_line)

                            # Split layers ONLY at the correct layer change markers
                            if layer_change_marker in decoded_line:
                                processed = self.process_layer(
                                    layer_buffer, current_layer, self.total_layers
                                )
                                tmp_file.writelines(processed)
                                current_layer += 1
                                layer_buffer = []

                        # Process remaining lines after last layer marker
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
    parser.add_argument("-nozzleDiameter", type=float, default=None, help="...")
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
        nozzle_diameter=args.nozzleDiameter
        or GCodeProcessor.detect_nozzle_diameter(args.inputFile),
    )
    processor.process_gcode(args.inputFile, args.bgcode)


if __name__ == "__main__":
    main()
