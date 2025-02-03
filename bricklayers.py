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
from enum import Enum
import argparse
from tempfile import NamedTemporaryFile
from pathlib import Path
from statistics import median
import logging.handlers


class PrinterType(Enum):

    BAMBU = "bambu"  # Bambu Lab Studio slicer identification
    PRUSA = "prusa"  # PrusaSlicer/G-code dialect identification


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
        log_level=logging.INFO,
    ):
        self.extrusion_multiplier = extrusion_multiplier
        self.first_layer_multiplier = (
            first_layer_multiplier
            if first_layer_multiplier is not None
            else 1.5 * extrusion_multiplier
        )  # Scale base multiplier
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
        self.perimeter_found = False
        self.total_z_shifts = 0
        self.total_extrusion_adjustments = 0
        self.processing_start = None
        self.layer_times = []
        self.simplified_features = 0
        self.failed_simplifications = 0
        self.decoding_errors = 0

        # Unified logging configuration
        self._configure_logging(log_level)

    def decode_line(self, line_bytes):
        try:
            return line_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            self.decoding_errors += 1
            self.logger.warning(
                f"Encoding error in line (replaced invalid bytes): {str(e)}"
            )
            return line_bytes.decode("utf-8", errors="replace")

    def _configure_logging(self, log_level):
        self.logger = logging.getLogger("Bricklayers")
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Rotating main log (keep latest runs)
            rotating_handler = logging.handlers.RotatingFileHandler(
                "bricklayers.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            rotating_handler.setFormatter(formatter)

            # Error console output
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(rotating_handler)
            self.logger.addHandler(console_handler)

    def detect_printer_type(self, file_path):
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    decoded_line = decoded_line = self.decode_line(line)
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
                    decoded_line = decoded_line = self.decode_line(line)
                    if decoded_line.startswith("G1 Z"):
                        z_match = re.search(r"Z([\d.]+)", decoded_line)
                        if z_match:
                            z_values.append(float(z_match.group(1)))

        # Calculate all positive layer height candidates
        layer_heights = []
        for i in range(len(z_values) - 1):
            delta = z_values[i + 1] - z_values[i]
            if delta > 0:  # Only consider positive Z movements
                layer_heights.append(delta)

        # Validation checks
        if not layer_heights:
            raise ValueError("No valid positive Z-axis movements detected")

        if len(set(round(h, 3) for h in layer_heights)) > 3:
            self.logger.warning("Multiple different layer heights detected")

        median_height = median(layer_heights)

        # Sanity bounds check (0.04mm - 0.6mm)
        if not 0.04 <= median_height <= 0.6:
            raise ValueError(
                f"Implausible layer height: {median_height:.2f}mm. "
                "Check Z-axis movements or specify manually with -layerHeight"
            )

        # Verify reasonable distribution around median
        within_tolerance = [
            h
            for h in layer_heights
            if 0.95 * median_height <= h <= 1.05 * median_height
        ]
        if len(within_tolerance) / len(layer_heights) < 0.8:
            self.logger.warning(
                "Inconsistent layer heights detected - verify Z-axis commands"
            )

        return median_height

    def dynamic_simplification_tolerance(self, poly: Polygon) -> float:
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Handle degenerate geometries with zero/negative dimensions
        if width <= 0.0 or height <= 0.0:
            return self.simplify_tolerance

        min_dimension = min(width, height)
        return max(self.simplify_tolerance, min_dimension * 0.1)

    def validate_simplification(self, original: Polygon, simplified: Polygon) -> bool:
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
        e_match = re.search(r"E([\d.]+)", line)
        if e_match:
            self.total_extrusion_adjustments += 1
            e_value = float(e_match.group(1))
            if self.current_layer == 0:
                new_e = e_value * self.first_layer_multiplier
                comment = "first layer"
            elif is_last_layer:
                new_e = e_value * self.last_layer_multiplier
                comment = "last layer"
            else:
                new_e = e_value * self.extrusion_multiplier
                comment = "internal perimeter"
            adjusted = re.sub(r"E[\d.]+", f"E{new_e:.5f}", line).strip()
            return f"{adjusted} ; {comment}\n"
        return line

    def classify_perimeters(self, perimeter_paths):
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
        layer_start = datetime.now()

        # Analyze perimeter geometry
        perimeter_paths = self.parse_perimeter_paths(layer_lines)
        outer_perimeters, inner_perimeters = self.classify_perimeters(perimeter_paths)

        # Track line indices of internal perimeters
        internal_lines = set()
        for _, line_indices in inner_perimeters:
            internal_lines.update(line_indices)

        # Processing state
        processed = []
        current_block = 0
        in_internal = False
        current_f = None  # Tracks current feedrate
        original_f = None  # Stores pre-shift feedrate
        z_speed = 300  # Z-axis move speed (300mm/min = 5mm/s)

        # Initial feedrate detection
        for line in layer_lines:
            if f_match := re.search(r"F([\d.]+)", line):
                current_f = float(f_match.group(1))
                break

        # Process each line in layer
        for line_idx, line in enumerate(layer_lines):
            # Update current feedrate from line
            if f_match := re.search(r"F([\d.]+)", line):
                current_f = float(f_match.group(1))

            # Handle Z position updates
            if line.startswith("G1 Z"):
                if z_match := re.search(r"Z([\d.]+)", line):
                    self.current_z = float(z_match.group(1))

            # Internal perimeter processing
            if line_idx in internal_lines:
                if not in_internal:
                    # Start of internal perimeter block
                    current_block += 1
                    original_f = current_f  # Capture pre-shift feedrate
                    z_shift = self.layer_height * 0.5 if current_block % 2 else 0

                    # Insert Z-shift with isolated Z-speed
                    processed.append(
                        f"G1 Z{self.current_z + z_shift:.3f} F{z_speed} ; Z-shift block {current_block}\n"
                    )

                    # Restore original XY feedrate
                    if original_f is not None:
                        processed.append(f"G1 F{original_f:.1f} ; Restore feedrate\n")
                    else:
                        self.logger.warning(
                            f"No feedrate captured before Z-shift at layer {layer_num}"
                        )

                    in_internal = True
                    self.shifted_blocks += 1

                # Apply extrusion adjustments
                processed.append(
                    self._adjust_extrusion(line, layer_num == total_layers - 1)
                )
            else:
                if in_internal:
                    # End of internal perimeter block
                    processed.append(
                        f"G1 Z{self.current_z:.3f} F{z_speed} ; Reset Z position\n"
                    )

                    # Restore current feedrate at block exit
                    if current_f is not None:
                        processed.append(
                            f"G1 F{current_f:.1f} ; Resume outer feedrate\n"
                        )

                    in_internal = False
                processed.append(line)

        # Performance logging
        self.layer_times.append((datetime.now() - layer_start).total_seconds())
        self.logger.info(
            f"Layer {layer_num+1}: Shifted {current_block} blocks | "
            f"Feedrate preservation: {original_f or 'default'}"
        )

        return processed

    def process_gcode(self, input_file, is_bgcode=False):
        self.processing_start = datetime.now()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_path = None  # Track temporary file path at instance level

        # Generate timestamped log filename
        current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_GMT%z")
        timestamped_log = os.path.join(script_dir, f"bricklayers_{current_time}.log")

        # Create timestamped log handler
        file_handler = logging.FileHandler(timestamped_log, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        self.logger.addHandler(file_handler)

        try:
            self.logger.info("═" * 55)
            self.logger.info(f"Starting processing of {input_file}")
            self.logger.info(f"Timestamped log: {timestamped_log}")

            input_path = Path(input_file)
            if is_bgcode:
                input_path = input_path.with_suffix(".gcode")
                self.logger.info(f"Decoding binary G-code to: {input_path}")
                os.system(f"bgcode decode {input_file} -o {input_path}")

            # Printer detection and layer height calculation
            self.logger.info("Detecting printer type...")
            self.printer_type = self.detect_printer_type(input_path)

            if self.layer_height is None:
                self.logger.info("Auto-detecting layer height...")
                self.layer_height = self.detect_layer_height(input_path)

            self.z_shift = self.layer_height * 0.5
            self.logger.info(
                f"Layer height: {self.layer_height:.2f}mm | Z-shift: {self.z_shift:.2f}mm"
            )

            # Streaming layer processing
            with open(input_path, "r") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    self.logger.info("Counting total layers...")
                    self.total_layers = sum(
                        1 for line in iter(mm.readline, b"") if line.startswith(b"G1 Z")
                    )
                    mm.seek(0)
                    self.logger.info(f"Found {self.total_layers} layers to process")

                    with NamedTemporaryFile(mode="w", delete=False) as temp_file:
                        self.temp_path = temp_file.name  # Capture path immediately
                        layer_buffer = []
                        current_layer = 0

                        self.logger.info("Begin layer processing...")
                        try:
                            for line in iter(mm.readline, b""):
                                decoded_line = decoded_line = self.decode_line(line)
                                layer_buffer.append(decoded_line)

                                if (
                                    decoded_line.startswith("G1 Z")
                                    or "; CHANGE_LAYER" in decoded_line
                                ):
                                    if layer_buffer:
                                        processed = self.process_layer(
                                            layer_buffer,
                                            current_layer,
                                            self.total_layers,
                                        )
                                        temp_file.writelines(processed)
                                        current_layer += 1
                                        layer_buffer = []

                            if layer_buffer:
                                processed = self.process_layer(
                                    layer_buffer, current_layer, self.total_layers
                                )
                                temp_file.writelines(processed)
                        except Exception as e:
                            self.logger.error(
                                f"Critical error during processing: {str(e)}"
                            )
                            raise  # Re-raise to trigger finally cleanup

                    # Only reach here if processing completed successfully
                    os.replace(self.temp_path, input_path)
                    self.temp_path = None  # Successfully replaced, clear cleanup flag
                    self.logger.info(f"Processed file saved to: {input_path}")

            if is_bgcode:
                self.logger.info("Re-encoding to binary G-code format...")
                os.system(f"bgcode encode {input_path}")
                input_path.unlink()

            # Final report
            self.logger.info(
                "════════════════════ Processing Complete ════════════════════\n"
                f"Total Layers Processed: {self.total_layers}\n"
                f"Total Decoding Errors: {self.decoding_errors}\n"
                f"Total Z-Shifts Applied: {self.shifted_blocks}\n"
                f"Extrusion Adjustments: {self.total_extrusion_adjustments}\n"
                f"Simplified Features: {self.simplify_success_count}\n"
                f"Failed Simplifications: {self.failed_simplifications}\n"
                f"Average Layer Time: {sum(self.layer_times)/len(self.layer_times)*1000:.2f}ms\n"
                f"Total Processing Time: {(datetime.now()-self.processing_start).total_seconds()*1000:.2f}ms\n"
                f"Peak Memory Usage: {self.get_memory_usage():.2f}MB\n"
                "═════════════════════════════════════════════════════════════"
            )

        finally:
            # Clean up timestamped handler
            self.logger.removeHandler(file_handler)
            file_handler.close()

            # Clean up temporary file if it still exists
            if self.temp_path and os.path.exists(self.temp_path):
                try:
                    os.unlink(self.temp_path)
                    self.logger.info(f"Cleaned up temporary file: {self.temp_path}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to clean up temporary file {self.temp_path}: {str(e)}"
                    )
            self.temp_path = None  # Reset tracking variable

        return input_path

    def get_memory_usage(self):
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
    parser.add_argument(
        "-firstLayerMultiplier",
        type=float,
        default=None,
        help="Extrusion multiplier for first layer (default: 1.5 × base multiplier)",
    )
    parser.add_argument(
        "-lastLayerMultiplier",
        type=float,
        default=None,
        help="Extrusion multiplier for last layer (default: 0.5 × base multiplier)",
    )

    args = parser.parse_args()

    processor = GCodeProcessor(
        log_level=getattr(logging, args.logLevel),
        layer_height=args.layerHeight,
        first_layer_multiplier=args.firstLayerMultiplier,
        last_layer_multiplier=args.lastLayerMultiplier,
        extrusion_multiplier=args.extrusionMultiplier,
        simplify_tolerance=args.simplifyTolerance,
    )
    processor.process_gcode(args.input_file, args.bgcode)


if __name__ == "__main__":
    main()
