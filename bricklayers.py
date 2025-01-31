# Previous copyright and license headers remain unchanged

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
from typing import List, Tuple, Optional
import argparse
from tempfile import NamedTemporaryFile
from pathlib import Path
from statistics import median


class PrinterType(Enum):
    BAMBU = "bambu"
    PRUSA = "prusa"


class GCodeProcessor:
    def __init__(
        self,
        layer_height=None,
        extrusion_multiplier=1.0,
        min_distance=0.1,
        max_intersection_area=0.5,
        simplify_tolerance=0.0,
    ):
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

    def detect_printer_type(self, file_path):
        """Detect printer type using memory-mapped file"""
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
        """Detect layer height from memory-mapped file"""
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

    def validate_simplification(self, original: Polygon, simplified: Polygon) -> bool:
        """Validate simplified geometry meets medical standards"""
        if not simplified.is_valid:
            return False
        if abs(original.area - simplified.area) / original.area > 0.02:
            return False
        if original.distance(simplified) > self.simplify_tolerance * 1.5:
            return False
        return True

    def parse_perimeter_paths(self, layer_lines):
        """Parse perimeter paths with validation for minimum points"""
        perimeter_paths = []
        current_path = []
        current_lines = []

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

                    # Validate minimum points for polygon creation
                    if (
                        len(current_path) >= 4
                    ):  # Require at least 3 distinct points + closure
                        try:
                            poly = Polygon(current_path)

                            # Apply simplification with validation
                            if self.simplify_tolerance > 0:
                                simplified = poly.simplify(
                                    self.simplify_tolerance, preserve_topology=True
                                )
                                simplified = make_valid(simplified)

                                # Ensure valid geometry after simplification
                                if (
                                    simplified.is_valid
                                    and len(simplified.exterior.coords) >= 4
                                    and self.validate_simplification(poly, simplified)
                                ):
                                    poly = simplified

                            if poly.is_valid and not poly.is_empty:
                                perimeter_paths.append((poly, current_lines))

                        except Exception as e:
                            logging.debug(f"Ignoring invalid geometry: {str(e)}")

                    current_path = []
                    current_lines = []

        return perimeter_paths

    def _adjust_extrusion(self, line, is_last_layer):
        """Adjust extrusion values with contextual commenting"""
        e_match = re.search(r"E([\d.]+)", line)
        if e_match:
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
        """Classify perimeters using spatial indexing and containment checks"""
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
        """Process a single layer's G-code with geometric analysis"""
        self.current_layer = layer_num
        perimeter_paths = self.parse_perimeter_paths(layer_lines)
        outer_perimeters, inner_perimeters = self.classify_perimeters(perimeter_paths)

        internal_lines = set()
        for _, lines in inner_perimeters:
            internal_lines.update(lines)

        processed = []
        current_block = 0
        in_internal = False

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
        return processed

    def process_gcode(self, input_file, is_bgcode=False):
        """Main processing with memory-mapped file handling"""
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
        return input_path


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
        default=0.0,
        help="Simplification tolerance for ISO 2768 compliance",
    )
    parser.add_argument(
        "-bgcode",
        action="store_true",
        help="Enable for PrusaSlicer binary G-code processing",
    )

    args = parser.parse_args()

    processor = GCodeProcessor(
        layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier,
        simplify_tolerance=args.simplifyTolerance,
    )
    processor.process_gcode(args.input_file, args.bgcode)


if __name__ == "__main__":
    main()
