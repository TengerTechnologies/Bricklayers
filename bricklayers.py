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

# TODO: Figure out why there are scattered vertical layer gaps in perimeters and support structures and resolve those issues.

from datetime import datetime
import mmap
import math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
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
        first_layer_multiplier=1.5,
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
        self.extrusion_mode = "absolute"  # Default assumption
        self.total_shifts = 0  # New shift counter
        self.last_e_value = 0.0
        self.current_z = 0.0  # Track printed Z
        self.max_mem_usage = 0.0  # Peak memory tracking
        self.max_model_z = 0.0
        self.layer_data = []  # Stores (z_height, paths) tuples
        self.layer_height_source = "user" if layer_height else "auto"
        self.first_layer_height = (
            self.layer_height * self.first_layer_multiplier
            if self.layer_height
            else None
        )
        self.analysis_results = {
            "model_vertical_gaps": [],
            "support_vertical_gaps": [],
            "horizontal_overlaps": [],
            "potential_issues": [],
            "support_layer_indices": set(),
            "model_layer_indices": set(),
        }
        self.overhang_markers = {
            PrinterType.BAMBU: (
                ";FEATURE:Overhang",
                # ";BRIDGE",
                # ";_BRIDGE"
            ),
            PrinterType.PRUSA: (
                ";TYPE:Overhang perimeter",
                # ";_BRIDGE_",
                # ";BRIDGE",
                # ";TYPE:Bridge infill",
            ),
        }

        # Regex patterns
        self.re_z = re.compile(r"Z([\d.]+)")
        self.re_f = re.compile(r"F([\d.]+)")
        self.re_e = re.compile(r"E([\d.]+)")
        self.re_e_sub = re.compile(r"E[-+]?\d*\.?\d+([eE][-+]?\d+)?")
        self.re_x = re.compile(r"X([-+]?\d*\.?\d+([eE][-+]?\d+)?)")
        self.re_y = re.compile(r"Y([-+]?\d*\.?\d+([eE][-+]?\d+)?)")

        self._configure_logging(log_level)

    def _classify_layer_type(self, layer_lines):
        """Determine if layer contains support material"""
        for line in layer_lines:
            if (
                line.startswith(";TYPE:Support material")
                or line.startswith(";TYPE:Support material interface")
                or "support" in line.lower()
            ):
                return "support"
        return "model"

    def shift_coordinate(match, shift):
        try:
            val = float(match.group(1))
            return f"{match.group(0)[0]}{val + shift:.5f}"
        except:
            return match.group(0)

    def _identify_perimeter_blocks(self, layer_lines):
        """Group G-code lines into blocks based on perimeter features."""
        blocks = []
        current_block = []
        current_feature = None

        for line in layer_lines:
            # Check for feature changes in comment lines
            if line.startswith(";"):
                if self._is_perimeter(line):
                    # Save the previous block if it exists
                    if current_block:
                        blocks.append((current_block, current_feature))
                        current_block = []
                    # Start new perimeter block
                    current_feature = line.strip()
                elif "TYPE:" in line or "FEATURE:" in line:
                    # Non-perimeter feature marker - end current block
                    if current_block:
                        blocks.append((current_block, current_feature))
                        current_block = []
                    current_feature = None

            current_block.append(line)

        # Don't forget the last block
        if current_block:
            blocks.append((current_block, current_feature))

        return blocks

    def _is_custom_block(self, feature_type):
        """Check if this is a custom G-code block that should not be modified"""
        return feature_type and "TYPE:Custom" in feature_type

    def _process_alternating_blocks(self, layer_lines, current_z):
        """Process a layer's worth of G-code with bricklaying modifications."""
        processed = []
        current_feature = None
        current_z_position = current_z

        # First try to identify blocks by slicer markers
        blocks = self._identify_perimeter_blocks(layer_lines)

        # If no clear perimeter markers found, use geometric analysis
        if not any(block[1] for block in blocks):
            self.logger.debug(
                f"No slicer perimeter markers found in layer {self.current_layer}, using geometric analysis"
            )
            paths = self.parse_perimeter_paths(layer_lines)
            outer_paths, inner_paths = self.classify_perimeters(paths)

            if outer_paths or inner_paths:
                self.logger.debug(
                    f"Geometric analysis found {len(outer_paths)} outer and {len(inner_paths)} inner perimeters"
                )
                blocks = self.blocks_from_geometric_classification(
                    layer_lines, outer_paths, inner_paths
                )
            else:
                self.logger.debug("No perimeters identified through geometric analysis")
                blocks = [(layer_lines, None)]

        # Process each block
        for block_lines, feature_type in blocks:
            modifications = {
                "original_lines": block_lines.copy(),
                "modified_lines": [],
                "z_height": False,
                "extrusion": False,
                "extrusion_multiplier": 1.0,
            }

            # Skip processing for custom blocks
            if self._is_custom_block(feature_type):
                processed.extend(block_lines)
                continue

            # Track block statistics
            if feature_type:
                self.block_statistics["total_blocks"] += 1
                self.block_statistics["by_type"][feature_type] = (
                    self.block_statistics["by_type"].get(feature_type, 0) + 1
                )

            # Determine target Z height
            target_z = current_z
            if not self._is_perimeter(feature_type):
                if self.current_layer == 0:
                    # First layer: Print non-perimeters at 150% height
                    target_z = current_z + (self.layer_height * 0.5)
                    modifications["z_height"] = True
                    modifications["z_delta"] = self.layer_height * 0.5
                    modifications["new_height"] = target_z
                    self.block_statistics["first_layer_height_mods"] += 1
                    processed.append(
                        f"; [BRICKLAYER] First layer non-perimeter at 150% height ({target_z:.3f}mm)\n"
                    )
                else:
                    # Subsequent layers: maintain the offset
                    target_z = current_z + (self.layer_height * 0.5)
                    if abs(target_z - current_z_position) > 0.001:
                        modifications["z_height"] = True
                        modifications["z_delta"] = target_z - current_z_position
                        processed.append(
                            f"; [BRICKLAYER] Maintaining +{self.layer_height * 0.5:.3f}mm offset\n"
                        )

            # Process lines in the block
            modified_lines = []
            for line in block_lines:
                # Handle Z movements
                if line.startswith(("G0", "G1")) and "Z" in line:
                    if modifications["z_height"]:
                        z_match = self.re_z.search(line)
                        if z_match:
                            z_speed = self._calculate_z_speed(
                                abs(target_z - current_z_position)
                            )
                            line = f"G1 Z{target_z:.3f} F{z_speed:.0f} ; Z adjustment\n"
                            current_z_position = target_z
                            self.block_statistics["total_z_modifications"] += 1

                # Handle extrusion
                if line.startswith("G1") and "E" in line:
                    if self.current_layer == 0 and not self._is_perimeter(feature_type):
                        # Adjust extrusion for first layer height increase
                        e_match = self.re_e.search(line)
                        if e_match:
                            original_e = float(e_match.group(1))
                            modified_e = (
                                original_e * 1.5
                            )  # 150% extrusion for 150% height
                            line = self.re_e_sub.sub(f"E{modified_e:.5f}", line)
                            modifications["extrusion"] = True
                            modifications["extrusion_multiplier"] = 1.5
                            self.block_statistics["total_extrusion_modifications"] += 1

                modified_lines.append(line)

            # Log block details at DEBUG level
            self._log_block_details(block_lines, feature_type, modifications)

            processed.extend(modified_lines)

        return processed

    def _is_perimeter(self, feature_comment):
        """Check if a feature comment indicates perimeter"""
        if not feature_comment:
            return False

        # Check for printer-specific overhang markers
        if self.printer_type in self.overhang_markers:
            if any(
                marker in feature_comment
                for marker in self.overhang_markers[self.printer_type]
            ):
                return True

        return any(
            p in feature_comment.lower()
            for p in [
                "perimeter",
                "outer wall",
                "inner wall",
                "support",
                "tree",
                "brim",
                "overhang",  # Added to include overhang features
                "skirt",
                "skirt/brim",
            ]
        )

    def _is_top_layer(self, current_z):
        """Determine if current layer is in top section using Z-height"""
        top_threshold = self.max_model_z - (self.layer_height * 3)
        return current_z >= top_threshold

    def _analyze_layer_bonding(self):
        """Improved gap detection with buffering"""
        prev_z = None
        prev_paths = []
        prev_type = None
        prev_layer_idx = 0

        for layer_idx, (z, paths) in enumerate(self.layer_data):
            current_type = self._classify_layer_type(paths)

            # Store layer type for reporting
            if current_type == "support":
                self.analysis_results["support_layer_indices"].add(layer_idx)
            else:
                self.analysis_results["model_layer_indices"].add(layer_idx)

            if prev_z is not None:
                vertical_gap = z - prev_z
                gap_threshold = self.layer_height * 1.2  # 20% tolerance

                # Only check gaps between same layer types
                if vertical_gap > gap_threshold:
                    gap_data = {
                        "from_z": prev_z,
                        "to_z": z,
                        "gap_size": vertical_gap,
                        "layers": (prev_layer_idx, layer_idx),
                    }

                    if prev_type == "support" and current_type == "support":
                        self.analysis_results["support_vertical_gaps"].append(gap_data)
                    elif prev_type == "model" and current_type == "model":
                        self.analysis_results["model_vertical_gaps"].append(gap_data)
                    else:
                        # Cross-type transition (model<->support)
                        self.analysis_results["potential_issues"].append(
                            f"Type transition gap between {prev_type} and {current_type} "
                            f"at Z{prev_z:.2f}-{z:.2f} (Δ{vertical_gap:.2f}mm)"
                        )

        for z, paths in self.layer_data:
            # Create buffered polygons
            current_poly = Polygon(paths).buffer(0.05)
            prev_poly = prev_paths.buffer(0.05) if prev_paths else None

            # Check vertical spacing
            if prev_z is not None:
                vertical_gap = z - prev_z

            # Check overlap
            if prev_poly and not current_poly.intersects(prev_poly):
                self.analysis_results["vertical_gaps"].append((prev_z, z, vertical_gap))

            # Check horizontal overlap
            if prev_paths:
                current_polys = []
                prev_polys = []

                # Log current layer processing
                self.logger.debug(f"Analyzing layer {z}mm with {len(paths)} points")

                try:
                    if paths:
                        self.logger.debug(
                            f"Creating current polygons from {len(paths)} points"
                        )
                        current_poly = make_valid(Polygon(paths))
                        if current_poly.is_valid:
                            current_polys = [current_poly]
                            self.logger.debug(
                                f"Created valid current polygon with area {current_poly.area:.2f}mm²"
                            )
                        else:
                            self.logger.warning(f"Invalid current polygon at {z}mm")
                            self.analysis_results["potential_issues"].append(
                                f"Invalid geometry at {z}mm"
                            )

                    if prev_paths:
                        self.logger.debug(
                            f"Creating previous polygons from {len(prev_paths)} points"
                        )
                        prev_poly = make_valid(Polygon(prev_paths))
                        if prev_poly.is_valid:
                            prev_polys = [prev_poly]
                            self.logger.debug(
                                f"Created valid previous polygon with area {prev_poly.area:.2f}mm²"
                            )
                        else:
                            self.logger.warning(
                                f"Invalid previous polygon at {prev_z}mm"
                            )
                            self.analysis_results["potential_issues"].append(
                                f"Invalid geometry at {prev_z}mm"
                            )

                    if current_polys and prev_polys:
                        self.logger.info(
                            f"Comparing {len(current_polys)} current vs {len(prev_polys)} previous polys"
                        )
                        tree = STRtree(current_polys)

                        for prev_poly in prev_polys:
                            self.logger.debug(
                                f"Checking overlaps for previous poly area {prev_poly.area:.2f}mm²"
                            )
                            candidate_indices = tree.query(prev_poly)
                            candidates = [current_polys[i] for i in candidate_indices]
                            self.logger.debug(
                                f"Found {len(candidates)} potential overlaps"
                            )

                            overlap_found = any(
                                self.safe_intersection_area(cp, prev_poly) > 0
                                for cp in candidates
                            )

                        if not overlap_found:
                            self.logger.warning(
                                f"No overlap found between layers {prev_z}mm and {z}mm"
                            )
                            self.analysis_results["horizontal_overlaps"].append(
                                (prev_z, z, prev_poly.centroid)
                            )
                except Exception as e:
                    self.logger.error(
                        f"Overlap analysis failed: {str(e)}", exc_info=True
                    )
                    self.analysis_results["potential_issues"].append(
                        f"Analysis error: {str(e)}"
                    )

            prev_z = z
            prev_paths = paths
            prev_type = current_type
            prev_layer_idx = layer_idx

    def _get_support_gap_report(self):
        report = []
        support_gaps = self.analysis_results["support_vertical_gaps"]

        if support_gaps:
            report.append("\nSupport Structure Gaps:")
            for idx, gap in enumerate(support_gaps, 1):
                report.append(
                    f"  Gap {idx}: Layers {gap['layers'][0]}-{gap['layers'][1]} "
                    f"(Z{gap['from_z']:.2f} → Z{gap['to_z']:.2f}, "
                    f"Δ{gap['gap_size']:.2f}mm)"
                )
        else:
            report.append("\nNo support structure gaps detected")

        return report

    def safe_intersection_area(self, poly1, poly2):
        try:
            self.logger.info(f"Intersection succeeded")
            return poly1.intersection(poly2).area

        except Exception as e:
            self.logger.warning(f"Intersection failed: {str(e)}")
            return 0

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

            # Rotating file handler with debug level
            rotating_handler = logging.handlers.RotatingFileHandler(
                "bricklayers.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            rotating_handler.setLevel(log_level)  # Critical fix!
            rotating_handler.setFormatter(formatter)

            # Console handler with debug level
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)  # Changed from logging.ERROR
            console_handler.setFormatter(formatter)

            self.logger.addHandler(rotating_handler)
            self.logger.addHandler(console_handler)

    def detect_printer_type(self, file_path):
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    decoded_line = self.decode_line(line)
                    if "; FEATURE:" in decoded_line:
                        return PrinterType.BAMBU
                    if ";TYPE:" in decoded_line:
                        return PrinterType.PRUSA
                return PrinterType.PRUSA

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
            if self.printer_type == PrinterType.BAMBU:
                if "; FEATURE" in line:
                    current_type = (
                        PerimeterType.INNER if "Inner" in line else PerimeterType.OUTER
                    )
            elif self.printer_type == PrinterType.PRUSA:
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
        """Classify perimeter paths into outer and inner walls using geometric analysis."""
        if not perimeter_paths:
            return [], []

        outer_paths = []
        inner_paths = []

        # Convert paths to Shapely polygons
        polygons = []
        for poly, lines, ptype in perimeter_paths:
            if isinstance(poly, (Polygon, MultiPolygon)):
                polygons.append((poly, lines, ptype))

        if not polygons:
            return [], []

        # Create spatial index for efficiency
        tree = STRtree([p for p, _, _ in polygons])

        # Classify each polygon
        for idx, (poly, lines, _) in enumerate(polygons):
            is_inner = False

            # Find potential containing polygons
            candidates = tree.query(poly)
            candidates = [polygons[i][0] for i in candidates if i != idx]

            if candidates:
                # Check if this polygon is contained by any others
                containing_polys = [
                    p for p in candidates if p.contains(poly.buffer(-0.01))
                ]

                if containing_polys:
                    # If contained by others, this is an inner wall
                    is_inner = True
                    # Find the closest containing polygon by area difference
                    parent = max(containing_polys, key=lambda p: p.area)
                    area_ratio = poly.area / parent.area
                    # Verify it's not just a small offset of the outer wall
                    if area_ratio > 0.95:
                        is_inner = False
                else:
                    # Check for near-coincident walls that might be part of the same feature
                    nearest = min(candidates, key=lambda p: p.distance(poly))
                    if poly.distance(nearest) < self.min_distance:
                        # Compare areas to determine if this is likely part of a multi-wall feature
                        area_ratio = poly.area / nearest.area
                        if 0.8 < area_ratio < 1.2:  # Similar sized walls
                            is_inner = False
                        else:
                            is_inner = True

            if is_inner:
                inner_paths.append((poly, lines))
            else:
                outer_paths.append((poly, lines))

        self.logger.debug(
            f"Geometric classification complete: {len(outer_paths)} outer, {len(inner_paths)} inner paths"
        )
        return outer_paths, inner_paths

    def blocks_from_geometric_classification(
        self, layer_lines, outer_paths, inner_paths
    ):
        """Convert geometric analysis results back into G-code blocks."""
        blocks = []
        current_block = []
        current_type = None
        in_extrusion_move = False

        # Create lookup dictionaries for quick point-in-polygon testing
        outer_polys = [poly for poly, _ in outer_paths]
        inner_polys = [poly for poly, _ in inner_paths]
        outer_tree = STRtree(outer_polys) if outer_polys else None
        inner_tree = STRtree(inner_polys) if inner_polys else None

        for line in layer_lines:
            # Always include non-movement commands in current block
            if not line.startswith(("G0", "G1")):
                if current_block:
                    current_block.append(line)
                continue

            # Extract X/Y coordinates from movement commands
            x_match = self.re_x.search(line)
            y_match = self.re_y.search(line)

            # If this is an extrusion move (has X/Y and E)
            if x_match and y_match and "E" in line:
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                point = Point(x, y)

                # Determine point type using spatial index
                new_type = None
                if outer_tree:
                    potential_outers = [
                        i
                        for i in outer_tree.query(point)
                        if outer_polys[i].contains(point)
                    ]
                    if potential_outers:
                        new_type = ";TYPE:External perimeter"

                if not new_type and inner_tree:
                    potential_inners = [
                        i
                        for i in inner_tree.query(point)
                        if inner_polys[i].contains(point)
                    ]
                    if potential_inners:
                        new_type = ";TYPE:Internal perimeter"

                # If point type changed, start new block
                if new_type != current_type:
                    if current_block:
                        blocks.append((current_block, current_type))
                        current_block = []
                    current_type = new_type
                    if current_type:
                        current_block.append(
                            f"{current_type} ; [BRICKLAYER] Geometrically classified\n"
                        )

                in_extrusion_move = True

            else:
                # For non-extrusion moves, only close block if we were in an extrusion move
                if in_extrusion_move:
                    if current_block:
                        blocks.append((current_block, current_type))
                        current_block = []
                    current_type = None
                    in_extrusion_move = False

            # Add the current line to whatever block we're building
            if current_type or not in_extrusion_move:
                current_block.append(line)

        # Don't forget the last block
        if current_block:
            blocks.append((current_block, current_type))

        # Handle any lines that didn't fit into perimeter blocks
        unclassified_lines = []
        for lines, type_ in blocks:
            if not type_:
                unclassified_lines.extend(lines)

        if unclassified_lines:
            self.logger.debug(f"Found {len(unclassified_lines)} unclassified lines")
            # Add unclassified lines as a separate block
            blocks.append(
                (
                    unclassified_lines,
                    ";TYPE:Internal infill ; [BRICKLAYER] Unclassified geometry",
                )
            )

        self.logger.debug(f"Created {len(blocks)} blocks from geometric classification")
        return blocks

    def _adjust_top_layers(self):
        # Detect last 5 layers using actual Z position
        if (self.max_model_z - self.current_z) <= 5 * self.layer_height:
            self.logger.debug(
                f"Applying top layer compensation at Z{self.current_z:.2f}"
            )
            self.extrusion_multiplier *= 1.15  # 15% extra flow
            # Smooth transition for preceding layers
            for i in range(1, 4):
                if (self.max_z - self.current_z) == i * self.layer_height:
                    self.extrusion_multiplier *= 1 + 0.05 * (4 - i)

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

    def _log_block_details(self, block_lines, feature_type, modifications):
        """Log detailed block information at DEBUG level"""
        if self.logger.getEffectiveLevel() == logging.DEBUG:
            block_summary = [
                f"\n──── Block Details - Layer {self.current_layer} ────",
                f"Feature Type: {feature_type}",
                f"Classification: {'Perimeter' if self._is_perimeter(feature_type) else 'Non-Perimeter'}",
            ]

            # Log modifications
            if modifications.get("extrusion"):
                block_summary.append(f"Extrusion Modified: Yes")
                block_summary.append(
                    f"  Multiplier: {modifications['extrusion_multiplier']:.3f}"
                )
            else:
                block_summary.append("Extrusion Modified: No")

            if modifications.get("z_height"):
                block_summary.append(f"Z-Height Modified: Yes")
                block_summary.append(f"  Change: {modifications['z_delta']:+.3f}mm")
                if self.current_layer == 0 and not self._is_perimeter(feature_type):
                    block_summary.append(
                        f"  First Layer Height: {modifications['new_height']:.3f}mm (150% of base height)"
                    )
            else:
                block_summary.append("Z-Height Modified: No")

            # Log original and modified code
            block_summary.append("\nOriginal G-code:")
            block_summary.extend(f"  {line.strip()}" for line in block_lines)

            if modifications.get("modified_lines"):
                block_summary.append("\nModified G-code:")
                block_summary.extend(
                    f"  {line.strip()}" for line in modifications["modified_lines"]
                )

            self.logger.debug("\n".join(block_summary))

    def _is_header_line(self, line):
        """Check if a line is part of the header section"""
        header_markers = [
            "; generated by PrusaSlicer",
            "; thumbnail begin",
            "; thumbnail end",
            "; external perimeters extrusion width",
            "; perimeters extrusion width",
            "; infill extrusion width",
            "; solid infill extrusion width",
            "; top infill extrusion width",
            "; support material extrusion width",
            "; first layer extrusion width",
            "M73 P0",  # Initial progress indicator
            "; printing object",
            "; stop printing object",
        ]
        return any(marker in line for marker in header_markers)

    def _is_footer_content(self, line):
        """Check if a line is part of the footer section"""
        footer_markers = [
            "; filament used [mm]",
            "; filament used [cm3]",
            "; filament used [g]",
            "; filament cost",
            "; total filament",
            "; estimated printing time",
            "; prusaslicer_config",
            "; objects_info",
        ]
        return any(marker in line for marker in footer_markers)

    def _generate_final_report(self):
        """Generate and log the final processing report."""
        # Calculate timing metrics
        self.total_time = sum(self.layer_times)
        self.avg_layer_time = (
            self.total_time / len(self.layer_times) if self.layer_times else 0
        )
        self.total_processing_time = (
            datetime.now() - self.processing_start
        ).total_seconds() * 1000
        self.time_variance = (
            sum((t - self.avg_layer_time) ** 2 for t in self.layer_times)
            / len(self.layer_times)
            if self.layer_times
            else 0
        )

        # Update memory tracking
        current_mem = self.get_memory_usage()
        self.max_mem_usage = max(self.max_mem_usage, current_mem)

        final_report = [
            "════════════════════ Processing Complete ════════════════════",
            f"Total Layers: {self.total_layers} | Layer Height: {self.layer_height:.3f}mm ({self.layer_height_source})",
            f"Max Z: {self.max_model_z:.2f}mm | Printer Type: {self.printer_type.value}",
            "",
            "Block Statistics:",
            f"Total Blocks Processed: {self.block_statistics['total_blocks']}",
            f"First Layer Height Modifications: {self.block_statistics['first_layer_height_mods']}",
            f"Total Z-Shifts Applied: {self.block_statistics['total_z_modifications']}",
            f"Total Extrusion Adjustments: {self.block_statistics['total_extrusion_modifications']}",
            "",
            "Blocks by Type:",
        ]

        for feature_type, count in sorted(self.block_statistics["by_type"].items()):
            final_report.append(f"  {feature_type}: {count}")

        final_report.extend(
            [
                "",
                "Analysis Results:",
                f"Model Layers: {len(self.analysis_results['model_layer_indices'])}",
                f"Support Layers: {len(self.analysis_results['support_layer_indices'])}",
                f"Model Vertical Gaps: {len(self.analysis_results['model_vertical_gaps'])}",
                f"Support Vertical Gaps: {len(self.analysis_results['support_vertical_gaps'])}",
                f"Horizontal Overlap Issues: {len(self.analysis_results['horizontal_overlaps'])}",
                f"Critical Transitions: {len([i for i in self.analysis_results['potential_issues'] if 'transition' in i])}",
                "",
                "Performance Metrics:",
                f"Processing Time: {self.total_processing_time:.2f}ms",
                f"Peak Memory Usage: {self.max_mem_usage:.2f}MB",
                f"Average Layer Time: {self.avg_layer_time:.2f}ms ±{self.time_variance**0.5:.2f}ms",
                f"Simplified Features: {self.simplified_features} | Failed Simplifications: {self.failed_simplifications}",
                f"Closed Paths: {self.closed_paths} | Open Paths: {self.open_paths}",
                f"Skipped Small Features: {self.skipped_small_features}",
                "═════════════════════════════════════════════════════════════",
            ]
        )

        # Add potential issues
        if self.analysis_results["potential_issues"]:
            final_report.append("\nPotential Issues:")
            for issue in self.analysis_results["potential_issues"]:
                final_report.append(f"  ! {issue}")

        # Add support gap report
        final_report.append("\n" + "\n".join(self._get_support_gap_report()))

        self.logger.info("\n".join(final_report))

    def process_gcode(self, input_file, is_bgcode=False):
        """Process G-code file with bricklaying modifications."""
        self.processing_start = datetime.now()
        self.block_statistics = {
            "total_blocks": 0,
            "by_type": {},
            "total_z_modifications": 0,
            "total_extrusion_modifications": 0,
            "first_layer_height_mods": 0,
        }

        # Set up logging
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

            # First detect printer type and basic characteristics
            self.printer_type = self.detect_printer_type(input_path)
            self.layer_height = self.layer_height or self.detect_layer_height(
                input_path
            )
            self.z_speed = self.z_speed or self.detect_z_speed(input_path)
            self.z_shift = self.layer_height * 0.5

            # Log initial detection
            self.logger.info(f"Detected printer type: {self.printer_type}")
            self.logger.info(
                f"Layer height: {self.layer_height:.3f}mm ({self.layer_height_source})"
            )

            # Now detect file characteristics
            self.logger.info("Analyzing file characteristics...")
            with open(input_path, "r+") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Detect max Z height
                    for line in iter(mm.readline, b""):
                        decoded_line = self.decode_line(line)
                        if decoded_line.startswith("G1 Z"):
                            z_match = self.re_z.search(decoded_line)
                            if z_match:
                                current_z = float(z_match.group(1))
                                if current_z > self.max_model_z:
                                    self.max_model_z = current_z

                    mm.seek(0)

                    # Count total layers using correct marker for detected printer type
                    layer_change_marker = {
                        PrinterType.BAMBU: "; CHANGE_LAYER",
                        PrinterType.PRUSA: ";LAYER_CHANGE",
                    }[self.printer_type]

                    self.total_layers = sum(
                        1
                        for line in iter(mm.readline, b"")
                        if layer_change_marker in self.decode_line(line)
                    )

            # Log file characteristics
            self.logger.info(f"Maximum Z height: {self.max_model_z:.2f}mm")
            self.logger.info(f"Total layers: {self.total_layers}")

            # Second pass: Process and modify G-code
            with open(input_path, "r+") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    with NamedTemporaryFile(mode="w", delete=False) as tmp_file:
                        # Detect the layer change marker
                        layer_change_marker = {
                            PrinterType.BAMBU: "; CHANGE_LAYER",
                            PrinterType.PRUSA: ";LAYER_CHANGE",
                        }[self.printer_type]

                        # Read all lines until the first layer change into header
                        header_lines = []
                        line = mm.readline()
                        while line:
                            decoded_line = self.decode_line(line)
                            if layer_change_marker in decoded_line:
                                break
                            header_lines.append(decoded_line)
                            line = mm.readline()

                        # Write header unmodified
                        tmp_file.writelines(header_lines)

                        # Process remaining lines
                        layer_buffer = [decoded_line]  # Include the layer change line
                        self.current_layer = 0
                        self.current_z = 0.0
                        in_custom_block = False

                        while line:
                            decoded_line = self.decode_line(line)

                            # Track Z position
                            if decoded_line.startswith("G1 Z"):
                                z_match = self.re_z.search(decoded_line)
                                if z_match:
                                    self.current_z = float(z_match.group(1))

                            # Handle custom blocks
                            if decoded_line.startswith(";TYPE:Custom"):
                                if layer_buffer:
                                    processed_lines = self._process_alternating_blocks(
                                        layer_buffer, self.current_z
                                    )
                                    tmp_file.writelines(processed_lines)
                                    layer_buffer = []
                                in_custom_block = True
                                tmp_file.write(decoded_line)
                            elif in_custom_block:
                                if decoded_line.startswith(";TYPE:"):
                                    in_custom_block = False
                                    layer_buffer.append(decoded_line)
                                else:
                                    tmp_file.write(decoded_line)
                            # Handle footer
                            elif self._is_footer_content(decoded_line):
                                if layer_buffer:
                                    processed_lines = self._process_alternating_blocks(
                                        layer_buffer, self.current_z
                                    )
                                    tmp_file.writelines(processed_lines)
                                    layer_buffer = []
                                tmp_file.write(decoded_line)
                                # Copy remaining footer content
                                remaining = mm.read()
                                tmp_file.write(self.decode_line(remaining))
                                break
                            # Handle layer changes
                            elif layer_change_marker in decoded_line:
                                layer_buffer.append(decoded_line)
                                processed_lines = self._process_alternating_blocks(
                                    layer_buffer, self.current_z
                                )
                                tmp_file.writelines(processed_lines)
                                layer_buffer = []
                                self.current_layer += 1
                                self.logger.debug(
                                    f"Processed layer {self.current_layer-1} at Z={self.current_z:.3f}"
                                )
                            else:
                                layer_buffer.append(decoded_line)

                            line = mm.readline()

                        # Process any remaining content
                        if layer_buffer:
                            processed_lines = self._process_alternating_blocks(
                                layer_buffer, self.current_z
                            )
                            tmp_file.writelines(processed_lines)

                    # Replace original file with modified version
                    os.replace(tmp_file.name, input_path)

            # Perform post-processing analysis
            self.logger.info("Performing post-processing analysis...")
            self._analyze_layer_bonding()

            # Generate final report
            self._generate_final_report()

        except Exception as e:
            self.logger.error(f"Critical processing error: {str(e)}", exc_info=True)
            raise

        finally:
            self.logger.removeHandler(log_handler)
            log_handler.close()

        return self.block_statistics, self.analysis_results


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
    parser.add_argument("--version", action="version", version="Bricklayers 1.2")

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
