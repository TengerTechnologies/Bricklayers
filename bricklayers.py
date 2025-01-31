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

from enum import Enum
from typing import List, Tuple, Optional
import re
import sys
import logging
import os
import argparse
from tempfile import NamedTemporaryFile
import io
from pathlib import Path
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from statistics import median


class PrinterType(Enum):
    """Enumeration for supported printer types."""

    BAMBU = "bambu"
    PRUSA = "prusa"


class GCodeProcessor:
    def __init__(
        self,
        layer_height=None,
        extrusion_multiplier=1.0,
        min_distance=0.1,
        max_intersection_area=0.5,
    ):
        self.extrusion_multiplier = extrusion_multiplier
        self.min_distance = min_distance
        self.max_intersection_area = max_intersection_area
        self.current_layer = 0
        self.current_z = 0.0
        self.total_layers = 0
        self.buffer_size = 1024 * 1024  # 1MB buffer size
        self.printer_type = None
        self.layer_height = layer_height
        self.z_shift = None
        self.shifted_blocks = 0
        self.perimeter_found = False

    def detect_printer_type(self, lines):
        """Detect printer type based on G-code features"""
        for line in lines:
            if "; FEATURE:" in line:
                return "bambu"
            elif ";TYPE:" in line:
                return "prusa"
        return "prusa"

    def detect_layer_height(self, lines):
        """Automatically detect layer height from G-code"""
        z_values = []
        for line in lines:
            if line.startswith("G1 Z"):
                z_match = re.search(r"Z([\d.]+)", line)
                if z_match:
                    z_values.append(float(z_match.group(1)))
        if len(z_values) < 2:
            raise ValueError("Not enough Z values to detect layer height.")
        return median([z_values[i + 1] - z_values[i] for i in range(len(z_values) - 1)])

    def parse_perimeter_paths(self, layer_lines):
        """Parse perimeter paths with line indices and coordinates"""
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
                    if current_path[0] != current_path[-1]:
                        current_path.append(current_path[0])
                    try:
                        poly = Polygon(current_path)
                        if poly.is_valid:
                            perimeter_paths.append((poly, current_lines))
                    except ValueError:
                        pass  # Skip invalid polygons
                    current_path = []
                    current_lines = []
        return perimeter_paths

    def classify_perimeters(self, perimeter_paths):
        """Classify perimeters using spatial indexing and containment checks"""
        if not perimeter_paths:
            return [], []
        
        polygons, line_indices = zip(*perimeter_paths)
        tree = STRtree(polygons)
        inner = []
        outer = []
        
        for idx, (poly, lines) in enumerate(perimeter_paths):
            # Get indexes of candidate polygons
            candidate_idxs = tree.query(poly)
            # Convert indexes to actual polygons, exclude self
            candidates = [polygons[i] for i in candidate_idxs if i != idx]
            # Check if any candidate contains the current polygon
            if any(c.contains(poly) for c in candidates):
                inner.append((poly, lines))
            else:
                outer.append((poly, lines))
        
        return outer, inner

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
            return re.sub(r"E[\d.]+", f"E{new_e:.5f}", line).strip() + f" ; {comment}\n"
        return line

    def process_layer(self, layer_lines, layer_num, total_layers):
        """Process a single layer's G-code with geometric analysis"""
        # Detect printer type if not already set
        if not self.printer_type:
            self.printer_type = self.detect_printer_type(layer_lines)

        # Parse and classify perimeters
        perimeter_paths = self.parse_perimeter_paths(layer_lines)
        outer_perimeters, inner_perimeters = self.classify_perimeters(perimeter_paths)

        # Track lines in internal perimeters
        internal_lines = set()
        for _, lines in inner_perimeters:
            internal_lines.update(lines)

        # Process layer lines with geometric awareness
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
                    self.perimeter_found = True
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
        """Main processing workflow with layer buffering"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logging.basicConfig(
            filename=os.path.join(script_dir, "z_shift_log.txt"),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        input_path = Path(input_file)
        if is_bgcode:
            input_path = input_path.with_suffix(".gcode")
            os.system(f"bgcode decode {input_file} -o {input_path}")

        with open(input_path, "r") as f:
            initial_lines = f.readlines()
            self.printer_type = self.detect_printer_type(initial_lines)
            self.total_layers = sum(
                1 for line in initial_lines if line.startswith("G1 Z")
            )
            if self.layer_height is None:
                self.layer_height = self.detect_layer_height(initial_lines)
            self.z_shift = self.layer_height * 0.5

        temp_file = NamedTemporaryFile(mode="w", delete=False)
        try:
            layer_buffer = []
            current_layer = 0
            with open(input_path, "r") as infile:
                for line in infile:
                    layer_buffer.append(line)
                    if line.startswith("G1 Z") or "; CHANGE_LAYER" in line:
                        if layer_buffer:
                            processed = self.process_layer(
                                layer_buffer, current_layer, self.total_layers
                            )
                            temp_file.writelines(processed)
                            current_layer += 1
                            layer_buffer = []
                if layer_buffer:
                    processed = self.process_layer(
                        layer_buffer, current_layer, self.total_layers
                    )
                    temp_file.writelines(processed)
            temp_file.close()
            os.replace(temp_file.name, input_path)
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise

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
        "-bgcode",
        action="store_true",
        help="Enable for PrusaSlicer binary G-code processing",
    )
    args = parser.parse_args()

    processor = GCodeProcessor(
        layer_height=args.layerHeight, extrusion_multiplier=args.extrusionMultiplier
    )
    processor.process_gcode(args.input_file, args.bgcode)


if __name__ == "__main__":
    main()
