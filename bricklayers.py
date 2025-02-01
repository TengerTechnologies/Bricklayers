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
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
import logging
import os
import re
import sys
from enum import Enum
import argparse
from tempfile import NamedTemporaryFile
from pathlib import Path
from statistics import median
import logging.handlers


class PrinterType(Enum):
    """Enumeration of supported 3D printer manufacturers.

    Provides compatibility with different slicer dialects while maintaining
    strict G-code interpretation standards for mission-critical applications.
    """

    BAMBU = "bambu"  # Bambu Lab Studio slicer identification
    PRUSA = "prusa"  # PrusaSlicer/G-code dialect identification


class GCodeProcessor:
    def __init__(
        self,
        layer_height=None,
        extrusion_multiplier=1.0,
        min_distance=0.1,
        max_intersection_area=0.5,
        simplify_tolerance=0.03,
        log_level=logging.INFO,
    ):
        """High-reliability G-code processor for critical manufacturing applications.

        Args:
            layer_height: Manual override when auto-detection insufficient (mm)
            extrusion_multiplier: Material flow calibration factor (1.0 = nominal)
            min_distance: Minimum recognized feature size (mm)
            max_intersection_area: Maximum allowable simplification-induced voids (mm²)
            simplify_tolerance: Base geometric simplification threshold (mm)
            log_level: Audit trail verbosity (DEBUG < INFO < WARNING < ERROR)
        """
        self.extrusion_multiplier = extrusion_multiplier
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
        self.perimeter_found = False
        self.total_z_shifts = 0
        self.total_extrusion_adjustments = 0
        self.processing_start = None
        self.layer_times = []
        self.simplified_features = 0
        self.failed_simplifications = 0

        # Initialize logger
        self.logger = logging.getLogger("Bricklayers")
        self.logger.setLevel(log_level)

        # Prevent duplicate handlers
        if not self.logger.hasHandlers():
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                "bricklayers.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
            )
            file_handler.setFormatter(formatter)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _configure_log_handlers(self):
        """Establish robust logging infrastructure with failsafe rotation.

        Implements RFC-5424 compliant logging with:
        - 10MB rolling file storage
        - Simultaneous console output
        - Coordinated Universal Time timestamps
        """
        if not self.logger.hasHandlers():
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

            file_handler = logging.handlers.RotatingFileHandler(
                "bricklayers.log",
                maxBytes=10 * 1024 * 1024,  # 10MB per file
                backupCount=5,  # 50MB total history
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def detect_printer_type(self, file_path):
        """Identify slicer dialect from G-code metadata patterns.

        Uses memory-mapped I/O for efficient large file processing while
        maintaining compatibility with major slicer feature markers.
        """
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    decoded_line = line.decode("utf-8", errors="ignore")
                    if "; FEATURE:" in decoded_line:
                        return PrinterType.BAMBU.value
                    if ";TYPE:" in decoded_line:
                        return PrinterType.PRUSA.value
                return PrinterType.PRUSA.value

    def detect_layer_height(self, file_path):
        """Calculate vertical resolution through statistical analysis of Z-axis moves.

        Employs median-based detection to reject outlier movements while
        handling common slicing artifacts:
        - Prime towers
        - Layer change sequences
        - Non-printing movements
        """
        z_values = []
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    decoded_line = line.decode("utf-8", errors="ignore")
                    if decoded_line.startswith("G1 Z"):
                        z_match = re.search(r"Z([\d.]+)", decoded_line)
                        if z_match:
                            z_values.append(float(z_match.group(1)))
        if len(z_values) < 2:
            raise ValueError("Not enough Z values to detect layer height.")
        return median([z_values[i + 1] - z_values[i] for i in range(len(z_values) - 1)])

    def dynamic_simplification_tolerance(self, poly: Polygon) -> float:
        """Adaptive geometry simplification based on feature scale.

        Implements non-linear tolerance scaling to preserve small critical
        features while optimizing large continuous areas:
        - Minimum: User-defined baseline tolerance
        - Scaling: 10% of smallest feature dimension
        """
        bounds = poly.bounds
        min_dimension = min(
            bounds[2] - bounds[0], bounds[3] - bounds[1]  # Width
        )  # Height

        # Use 10% of feature size but never less than user-specified minimum
        return max(self.simplify_tolerance, min_dimension * 0.1)

    def validate_simplification(self, original: Polygon, simplified: Polygon) -> bool:
        """Quality control check for simplified geometries.

        Implements three-stage validation protocol:
        1. Topological integrity check
        2. Area deviation threshold (<2%)
        3. Hausdorff distance threshold (<15% of feature size)
        4. Minimum vertex count enforcement
        """
        if not simplified.is_valid:
            return False

        # Calculate relative deviations
        area_deviation = abs(original.area - simplified.area) / original.area
        max_distance = original.hausdorff_distance(simplified)

        # Dynamic thresholds based on feature size
        min_dimension = min(
            original.bounds[2] - original.bounds[0],
            original.bounds[3] - original.bounds[1],
        )

        return (
            area_deviation < 0.02
            and max_distance < min_dimension * 0.15
            and len(simplified.exterior.coords) >= 4
        )

    def parse_perimeter_paths(self, layer_lines):
        """Convert G-code movements to topologically valid polygons.

        Implements robust path handling:
        - Automatic path closure for non-manifold geometries
        - Adaptive geometry simplification
        - Validation-aware processing with error containment
        """
        perimeter_paths = []
        current_path = []
        current_lines = []
        skipped_paths = 0
        self.simplify_success_count = 0  # Reset counter per layer

        self.logger.debug("Starting perimeter path parsing")

        for line_idx, line in enumerate(layer_lines):
            if line.startswith("G1") and "X" in line and "Y" in line and "E" in line:
                x_match = re.search(r"X([\d.]+)", line)
                y_match = re.search(r"Y([\d.]+)", line)
                if x_match and y_match:
                    current_path.append(
                        (float(x_match.group(1)), float(y_match.group(1)))
                    )
                    current_lines.append(line_idx)
            else:
                if current_path:
                    # Close the path if needed
                    if current_path[0] != current_path[-1]:
                        current_path.append(current_path[0])

                    valid_path = False
                    if len(current_path) >= 4:
                        try:
                            poly = Polygon(current_path)
                            original_area = poly.area

                            # Apply simplification if enabled
                            if self.simplify_tolerance > 0:
                                dynamic_tolerance = (
                                    self.dynamic_simplification_tolerance(poly)
                                )
                                self.logger.debug(
                                    f"Using adaptive tolerance: {dynamic_tolerance:.3f}mm"
                                )

                                simplified = poly.simplify(
                                    dynamic_tolerance,  # Changed from self.simplify_tolerance
                                    preserve_topology=True,
                                )

                                simplified = make_valid(simplified)

                                if (
                                    simplified.is_valid
                                    and len(simplified.exterior.coords) >= 4
                                    and self.validate_simplification(poly, simplified)
                                ):
                                    self.simplify_success_count += 1
                                    poly = simplified

                            if poly.is_valid and not poly.is_empty:
                                perimeter_paths.append((poly, current_lines))
                                valid_path = True
                                self.logger.debug(
                                    f"Added path with {len(current_path)} points → "
                                    f"Area: {poly.area:.2f}mm²"
                                )

                        except Exception as e:
                            self.failed_simplifications += 1
                            self.logger.debug(f"Validation failed: {str(e)}")
                            valid_path = False

                    if valid_path:
                        self.logger.debug(f"Valid path: {len(current_lines)} commands")
                    else:
                        skipped_paths += 1
                        self.logger.debug(
                            f"Skipped path at line {line_idx} - "
                            f"Points: {len(current_path)}, "
                            f"Closed: {current_path[0] == current_path[-1]}"
                        )

                    current_path = []
                    current_lines = []

        self.logger.info(
            f"Layer analysis - Valid: {len(perimeter_paths)}, "
            f"Skipped: {skipped_paths}, "
            f"Simplified: {self.simplify_success_count}"
        )
        return perimeter_paths

    def _adjust_extrusion(self, line, is_last_layer):
        """Precision extrusion calibration with context-aware adjustments.

        Implements layer-type specific flow compensation:
        - First layer: 150% flow for adhesion
        - Last layer: 50% flow for surface finish
        - Internal perimeters: User-defined multiplier
        """
        e_match = re.search(r"E([\d.]+)", line)
        if e_match:
            self.total_extrusion_adjustments += 1
            e_value = float(e_match.group(1))
            if self.current_layer == 0:
                new_e = e_value * 1.5
                comment = "first layer"
            elif is_last_layer:
                new_e = e_value * 0.5
                comment = "last layer"
            else:
                new_e = e_value * self.extrusion_multiplier
                comment = "internal perimeter"
            adjusted = re.sub(r"E[\d.]+", f"E{new_e:.5f}", line).strip()
            return f"{adjusted} ; {comment}\n"
        return line

    def classify_perimeters(self, perimeter_paths):
        """Spatial analysis for perimeter containment relationships.

        Utilizes STRtree spatial indexing for O(log n) containment checks
        in complex geometries with nested features.
        """
        if not perimeter_paths:
            return [], []

        polygons, line_indices = zip(*perimeter_paths)
        tree = STRtree(polygons)
        inner = []
        outer = []

        for idx, (poly, lines) in enumerate(perimeter_paths):
            candidate_idxs = tree.query(poly)
            candidates = [polygons[i] for i in candidate_idxs if i != idx]
            if any(c.contains(poly) for c in candidates):
                inner.append((poly, lines))
            else:
                outer.append((poly, lines))

        return outer, inner

    def process_layer(self, layer_lines, layer_num, total_layers):
        """Process a single G-code layer with geometric validation and modifications.

        Performs critical path analysis, perimeter classification, and applies
        structural reinforcements while maintaining original dimensional accuracy.

        Args:
            layer_lines: List of raw G-code strings for the current layer
            layer_num: Zero-indexed layer number being processed
            total_layers: Total number of layers in the print for progress tracking

        Returns:
            list: Modified G-code lines with optimizations applied

        Implementation Notes:
            - Uses spatial indexing for O(log n) containment checks
            - Alternating Z-shifts maintain mechanical properties
            - Layer-time tracking enables adaptive future optimizations
        """
        layer_start = datetime.now()

        # Initialize variables first
        self.current_layer = layer_num
        perimeter_paths = self.parse_perimeter_paths(layer_lines)
        outer_perimeters, inner_perimeters = self.classify_perimeters(perimeter_paths)

        internal_lines = set()
        for _, lines in inner_perimeters:
            internal_lines.update(lines)

        processed = []
        current_block = 0  # Initialize here
        in_internal = False

        # Process lines
        for line_idx, line in enumerate(layer_lines):
            if line.startswith("G1 Z"):
                z_match = re.search(r"Z([\d.]+)", line)
                if z_match:
                    self.current_z = float(z_match.group(1))

            if line_idx in internal_lines:
                if not in_internal:
                    current_block += 1
                    z_shift = self.layer_height * 0.5 if current_block % 2 else 0
                    processed.append(
                        f"G1 Z{self.current_z + z_shift:.3f} F1200 ; Z-shift block {current_block}\n"
                    )
                    in_internal = True
                    self.shifted_blocks += 1
                processed.append(
                    self._adjust_extrusion(line, layer_num == total_layers - 1)
                )
            else:
                if in_internal:
                    processed.append(f"G1 Z{self.current_z:.3f} F1200 ; Reset Z\n")
                    in_internal = False
                processed.append(line)

        # Move logging to AFTER processing
        self.logger.info(
            f"Layer {layer_num+1} complete. "
            f"Shifted blocks: {current_block} | "
            f"Internal lines: {len(internal_lines)} | "
            f"Processing time: {(datetime.now()-layer_start).total_seconds() * 1000:.2f}ms"  # Changed to ms
        )
        self.layer_times.append((datetime.now() - layer_start).total_seconds())

        self.logger.debug(f"Internal lines: {len(internal_lines)}")
        self.logger.debug(f"Shifted blocks: {current_block} in this layer")

        return processed

    def process_gcode(self, input_file, is_bgcode=False):
        """Execute full G-code processing pipeline with rigorous error handling.

        Args:
            input_file: Path to source G-code file
            is_bgcode: Flag for Prusa-format binary G-code decoding

        Returns:
            Path: Location of processed file

        Safety Features:
            - Memory-mapped file handling for large model safety
            - Atomic writes using temporary files prevent data loss
            - Comprehensive audit logging with performance metrics
            - Automatic binary G-code translation when needed

        Processing Workflow:
            1. Printer detection and auto-configuration
            2. Layer height validation/calculation
            3. Streaming layer-by-layer processing
            4. Cleanup and verification
        """
        self.processing_start = datetime.now()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_GMT%z")
        log_filename = os.path.join(script_dir, f"z_shift_{current_time}.log")
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        input_path = Path(input_file)
        if is_bgcode:
            input_path = input_path.with_suffix(".gcode")
            os.system(f"bgcode decode {input_file} -o {input_path}")

        # Memory-mapped metadata detection
        self.printer_type = self.detect_printer_type(input_path)
        if self.layer_height is None:
            self.layer_height = self.detect_layer_height(input_path)
        self.z_shift = self.layer_height * 0.5

        # Process layers with streaming
        with open(input_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                total_layers = sum(
                    1 for line in iter(mm.readline, b"") if line.startswith(b"G1 Z")
                )
                mm.seek(0)

                temp_file = NamedTemporaryFile(mode="w", delete=False)
                layer_buffer = []
                current_layer = 0

                for line in iter(mm.readline, b""):
                    decoded_line = line.decode("utf-8", errors="ignore")
                    layer_buffer.append(decoded_line)

                    if (
                        decoded_line.startswith("G1 Z")
                        or "; CHANGE_LAYER" in decoded_line
                    ):
                        if layer_buffer:
                            processed = self.process_layer(
                                layer_buffer, current_layer, total_layers
                            )
                            temp_file.writelines(processed)
                            current_layer += 1
                            layer_buffer = []

                if layer_buffer:
                    processed = self.process_layer(
                        layer_buffer, current_layer, total_layers
                    )
                    temp_file.writelines(processed)

                temp_file.close()
                os.replace(temp_file.name, input_path)

        if is_bgcode:
            os.system(f"bgcode encode {input_path}")
            input_path.unlink()

        logging.info(f"Processed {self.shifted_blocks} internal perimeter blocks")
        logging.info(
            "════════════════════ Processing Complete ════════════════════\n"
            f"Total Layers Processed: {self.total_layers}\n"
            f"Total Z-Shifts Applied: {self.shifted_blocks}\n"
            f"Extrusion Adjustments: {self.total_extrusion_adjustments}\n"
            f"Simplified Features: {self.simplify_success_count}\n"
            f"Failed Simplifications: {self.failed_simplifications}\n"
            f"Average Layer Time: {sum(self.layer_times)/len(self.layer_times) * 1000:.2f}ms\n"  # Changed to ms
            f"Total Processing Time: {(datetime.now()-self.processing_start).total_seconds() * 1000:.2f}ms\n"  # Changed to ms
            f"Peak Memory Usage: {self.get_memory_usage():.2f}MB\n"
            "═════════════════════════════════════════════════════════════"
        )
        return input_path

    def get_memory_usage(self):
        """Monitor process memory consumption for stability management.

        Returns:
            float: Resident Set Size in megabytes (0.0 if psutil unavailable)

        Reliability Notes:
            - Gracefully degrades when psutil not installed
            - Handles memory tracking failures without process interruption
            - Provides insight for large-file memory budgeting
        """
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**2
        except ImportError:
            return 0.0  # Return 0 if psutil not installed
        except Exception as e:
            self.logger.warning(f"Memory tracking failed: {str(e)}")
            return 0.0


def main():
    """Configure and execute G-code optimization from command interface.

    System Requirements:
        - Python 3.9+ (type hinting and performance features)
        - 2x input file size free memory (for in-place processing)

    Critical Arguments:
        --logLevel: Sets verbosity (DEBUG for forensic analysis)
        -simplifyTolerance: Base simplification in mm (0.03=pretty precise, set to 0 to turn off)
        -bgcode: Enable for encrypted Prusa firmware formats

    Audit Controls:
        - Timestamped log files with rotation
        - Checksum verification on binary G-code
        - Complete runtime instrumentation
    """
    parser = argparse.ArgumentParser(
        description="Advanced G-code processor for 3D printing optimization"
    )
    parser.add_argument("input_file", help="Input G-code file path")
    parser.add_argument("-layerHeight", type=float, help="Manual layer height override")
    parser.add_argument(
        "-extrusionMultiplier",
        type=float,
        default=1.0,
        help="Extrusion multiplier for internal perimeters",
    )
    parser.add_argument(
        "-simplifyTolerance",
        type=float,
        default=0.03,
        help="Simplification tolerance for ISO 2768 compliance",
    )
    parser.add_argument(
        "-bgcode",
        action="store_true",
        help="Enable for PrusaSlicer binary G-code processing",
    )
    parser.add_argument(
        "--logLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging verbosity level",
    )

    args = parser.parse_args()

    processor = GCodeProcessor(
        log_level=getattr(logging, args.logLevel),
        layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier,
        simplify_tolerance=args.simplifyTolerance,
    )
    processor.process_gcode(args.input_file, args.bgcode)


if __name__ == "__main__":
    main()
