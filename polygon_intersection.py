"""
This script uses an adaptive, hybrid approach leveraging bounding boxes and
polygon-intersection while also propagating extra (overextruded) height from first layer
upward and "undoing" that offset at the top. This keeps both build-plate contact and the
top surface flat. 

The first layer's non-perimeter infill is printed at a multiplier (150%) so effective
thickness is:

    first_layer_thickness = first_layer_infill_multiplier × layer_height

The extra height (offset) is:
    infill_offset = (first_layer_infill_multiplier – 1.0) × layer_height

For layers beyond first, non-perimeter regions above infill get raised by infill_offset
(e.g. 0.1mm for layer_height=0.2mm, multiplier=1.5). When exiting infill (or at file
end/top layer), z resets to nominal "perimeter" value.

Currently checks for open non-perimeter sections at EOF and outputs final "reset". May
need revision for complex files.

How It Works:
-------------
1. Hybrid Geometry Detection:
   Collects XY coords for non-perimeter moves. At boundaries, finalizes by computing both 
   bounding box (fast rejection) and Shapely polygon (precise inclusion). Stores by
   layer.

2. Adaptive Z-Shift Propagation:
   First layer extra height set by multiplier (default 1.5). For 0.2mm layer, extrudes
   0.3mm in non-perimeter areas. Extra offset (0.1mm) used in later layers - when new
   infill appears above previous infill, first XY move triggers z-shift to 
   base_z + infill_offset.

3. Top Layer Reset:
   When exiting non-perimeter region or at EOF, G-code resets z height to nominal
   perimeter base for flat top surface.

This hybrid/adaptive approach accurately detects complex infill regions and propagates 
overextrusion offset upward while maintaining flat top surface.
"""

#!/usr/bin/env python3
import re
import sys
import os
import logging
import argparse
from datetime import datetime
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPoint


def setup_logging(level_str):
    """
    Create (if necessary) a 'logs' subdirectory relative to this script, and configure
    logging to write to a timestamped log file with the provided log level.
    """
    # Determine numeric logging level from the string.
    numeric_level = getattr(logging, level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_str}")

    # Find the directory where this script is located.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"bricklayer_{timestamp}.log")

    # Set up logging: output to both file and console.
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(
        "Logging initialized at level %s. Log file: %s", level_str.upper(), log_filename
    )


def auto_detect_layer_height(file_path):
    """
    Automatically detect the layer height by scanning the G-code file for G1 commands
    containing 'Z' moves. It collects unique Z values, then computes the smallest positive
    difference between consecutive values as the presumed layer height.
    """
    z_values = set()
    try:
        with open(file_path, "r") as infile:
            for line in infile:
                if line.startswith("G1 ") and "Z" in line:
                    z_match = re.search(r"Z([\d\.]+)", line)
                    if z_match:
                        try:
                            z_val = float(z_match.group(1))
                            z_values.add(round(z_val, 5))
                        except ValueError:
                            continue
    except Exception as e:
        logging.error("Error reading file for auto-detection: %s", e)
        return None

    if len(z_values) < 2:
        logging.error("Not enough distinct Z values found for auto-detection.")
        return None

    sorted_z = sorted(z_values)
    logging.debug("Unique Z values detected: %s", sorted_z)
    differences = [
        round(sorted_z[i + 1] - sorted_z[i], 5)
        for i in range(len(sorted_z) - 1)
        if round(sorted_z[i + 1] - sorted_z[i], 5) > 0
    ]
    if not differences:
        logging.error(
            "Failed to compute layer height differences; differences list is empty."
        )
        return None

    layer_height = min(differences)
    logging.info("Auto-detected layer height: %.3f mm", layer_height)
    return layer_height


class BricklayeringProcessor:
    def __init__(self, layer_height=0.2):
        self.layer_height = layer_height
        self.current_z = 0.0
        self.current_layer = 0
        self.current_section = "perimeter"  # "perimeter" or "non_perimeter"
        self.extrusion_mode = "absolute"

        # Slicer–defined section types:
        self.perimeter_types = {
            "Custom",
            "Skirt/Brim",
            "Support material",
            "External perimeter",
            "Perimeter",
            "Overhang perimeter",
        }
        self.non_perimeter_types = {
            "Solid infill",
            "Internal infill",
            "Top solid infill",
            "Bridge infill",
        }

        # Settings for first-layer non-perimeter adjustments.
        self.first_layer_infill_multiplier = 1.5
        self.infill_offset = (
            self.first_layer_infill_multiplier - 1.0
        ) * self.layer_height
        logging.info(
            "First-layer multiplier: %.2f, Infill offset: %.3f mm",
            self.first_layer_infill_multiplier,
            self.infill_offset,
        )

        # Infill geometry storage by layer.
        # Each entry is a dict with keys: "bbox": (minx, miny, maxx, maxy) and "polygon": a shapely Polygon.
        self.layer_infill_regions = {}
        self.current_infill_coords = []
        self.pending_z_shift = False

    def compute_bbox(self, coords):
        """Compute bounding box (minx, miny, maxx, maxy) for a list of (x, y) points."""
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        logging.debug("Computed bbox: %s for %d points", bbox, len(coords))
        return bbox

    def finalize_current_infill_region(self):
        """
        Build the bounding box and polygon for the current non-perimeter region and store it.
        If there are insufficient points (< 3) to form a valid polygon, fall back to using the bounding box.
        """
        if not self.current_infill_coords:
            return

        # Always compute the bounding box.
        bbox = self.compute_bbox(self.current_infill_coords)

        # If there are fewer than 3 points, we cannot create a valid polygon.
        if len(self.current_infill_coords) < 3:
            logging.warning(
                "Insufficient points (%d) for region on layer %d; using bbox as polygon.",
                len(self.current_infill_coords),
                self.current_layer,
            )
            from shapely.geometry import box

            poly = box(*bbox)
        else:
            try:
                poly = Polygon(self.current_infill_coords)
                # If not valid or zero area, try a convex hull; if still not a Polygon, fall back to bbox.
                if not poly.is_valid or poly.area == 0:
                    logging.debug("Polygon invalid or zero area; using convex hull.")
                    poly = MultiPoint(self.current_infill_coords).convex_hull
                    if poly.geom_type != "Polygon":
                        from shapely.geometry import box

                        poly = box(*bbox)
            except Exception as e:
                logging.error("Error creating polygon: %s", e)
                from shapely.geometry import box

                poly = box(*bbox)

        region = {"bbox": bbox, "polygon": poly}
        self.layer_infill_regions.setdefault(self.current_layer, []).append(region)
        logging.info(
            "Finalized infill region on layer %d; now %d region(s) on this layer.",
            self.current_layer,
            len(self.layer_infill_regions[self.current_layer]),
        )
        self.current_infill_coords = []
        self.pending_z_shift = False

    def _previous_infill_at(self, x, y):
        """
        Check if a given (x, y) point lies within any infill region on the previous layer.
        Uses a fast bounding box test, then checks the full polygon.
        """
        prev_layer = self.current_layer - 1
        if prev_layer < 1:
            return False
        regions = self.layer_infill_regions.get(prev_layer, [])
        test_point = ShapelyPoint(x, y)
        for region in regions:
            minx, miny, maxx, maxy = region["bbox"]
            if not (minx <= x <= maxx and miny <= y <= maxy):
                continue
            poly = region["polygon"]
            if poly is not None and poly.contains(test_point):
                logging.debug(
                    "Point (%.3f, %.3f) inside region bbox %s.", x, y, region["bbox"]
                )
                return True
        logging.debug(
            "Point (%.3f, %.3f) not contained in any region on previous layer %d.",
            x,
            y,
            prev_layer,
        )
        return False

    def process_line(self, line):
        output = []

        # Detect layer changes via G1 Z moves.
        if line.startswith("G1 ") and "Z" in line:
            z_match = re.search(r"Z([\d\.]+)", line)
            if z_match:
                new_z = float(z_match.group(1))
                new_layer = int(round(new_z / self.layer_height))
                if new_layer != self.current_layer:
                    if self.current_infill_coords:
                        self.finalize_current_infill_region()
                    self.current_z = new_z
                    self.current_layer = new_layer
                    logging.info(
                        "Layer change: now at layer %d (Z=%.3f)",
                        self.current_layer,
                        self.current_z,
                    )
                else:
                    self.current_z = new_z

        # Update extrusion mode.
        if line.startswith("M82"):
            self.extrusion_mode = "absolute"
            logging.debug("Switched extrusion mode: absolute.")
        elif line.startswith("M83"):
            self.extrusion_mode = "relative"
            logging.debug("Switched extrusion mode: relative.")

        # Process slicer comment to check section type.
        if line.startswith(";TYPE:"):
            section_type = line.split(":")[-1].strip()
            new_section = (
                "non_perimeter"
                if section_type in self.non_perimeter_types
                else "perimeter"
            )
            if new_section != self.current_section:
                logging.info(
                    "Section change detected: %s -> %s",
                    self.current_section,
                    new_section,
                )
                self.handle_section_change(new_section, output)
                self.current_section = new_section

        # Process non-perimeter (infill) moves.
        if line.startswith("G1 ") and self.current_section == "non_perimeter":
            if self.pending_z_shift:
                x_match = re.search(r"X([\d\.]+)", line)
                y_match = re.search(r"Y([\d\.]+)", line)
                if x_match and y_match:
                    x_val = float(x_match.group(1))
                    y_val = float(y_match.group(1))
                    base_z = self.current_layer * self.layer_height
                    if self.current_layer >= 2 and self._previous_infill_at(
                        x_val, y_val
                    ):
                        new_z = base_z + self.infill_offset
                        note = "Aligned with infill: raising Z"
                    else:
                        new_z = base_z
                        note = "No underlying infill: no Z shift"
                    output.append(f"G1 Z{new_z:.3f} F9000 ; {note}\n")
                    logging.info(
                        "Layer %d non-perimeter: first XY (%.3f, %.3f) => %s (Z set to %.3f)",
                        self.current_layer,
                        x_val,
                        y_val,
                        note,
                        new_z,
                    )
                    self.pending_z_shift = False

            # Gather XY point coordinates.
            x_match = re.search(r"X([\d\.]+)", line)
            y_match = re.search(r"Y([\d\.]+)", line)
            if x_match and y_match:
                x_val = float(x_match.group(1))
                y_val = float(y_match.group(1))
                self.current_infill_coords.append((x_val, y_val))
                logging.debug(
                    "Collected XY point for infill region: (%.3f, %.3f)", x_val, y_val
                )

        # Adjust extrusion for first-layer non-perimeter moves.
        if (
            "E" in line
            and self.current_section == "non_perimeter"
            and self.current_layer == 1
        ):
            old_line = line
            line = self.adjust_extrusion(line)
            logging.debug(
                "Adjusted first-layer extrusion: '%s' -> '%s'",
                old_line.strip(),
                line.strip(),
            )

        output.append(line)
        return output

    def handle_section_change(self, new_section, output):
        """
        When switching sections (for example, leaving non-perimeter infill), finalize the current
        infill region and, if appropriate, output a command to reset Z to the nominal perimeter height.
        """
        base_z = self.current_layer * self.layer_height
        if self.current_layer < 1:
            return

        if self.current_section == "non_perimeter" and new_section == "perimeter":
            if self.current_infill_coords:
                self.finalize_current_infill_region()
            if self.current_layer >= 2:
                output.append(
                    f"G1 Z{base_z:.3f} F9000 ; Reset to nominal perimeter height\n"
                )
                logging.info(
                    "Reset Z to nominal perimeter height on layer %d",
                    self.current_layer,
                )
        elif new_section == "non_perimeter":
            if self.current_layer == 1:
                logging.info(
                    "Entering first-layer non-perimeter; using extrusion adjustment only."
                )
            elif self.current_layer >= 2:
                self.pending_z_shift = True
                logging.info(
                    "Entering non-perimeter on layer %d; awaiting first XY for Z test.",
                    self.current_layer,
                )

    def adjust_extrusion(self, line):
        """
        For first-layer non-perimeter moves, adjust the extrusion value to print a thicker layer.
        This extra material is propagated upward and later reset so that both the bottom and top surfaces remain flat.
        """
        e_match = re.search(r"E([\d\.]+)", line)
        if not e_match:
            return line
        try:
            e_value = float(e_match.group(1))
        except ValueError:
            return line
        new_e = e_value * self.first_layer_infill_multiplier
        return re.sub(r"E[\d\.]+", f"E{new_e:.5f}", line)

    def process_file(self, input_path, output_path):
        processed_lines = []
        with open(input_path, "r") as infile:
            lines = infile.readlines()
            logging.info("Opened file: %s (%d lines)", input_path, len(lines))
            for line in lines:
                processed = self.process_line(line)
                processed_lines.extend(processed)

        # Top-layer adjustment: if ending within a non-perimeter region, reset Z.
        if self.current_section == "non_perimeter":
            base_z = self.current_layer * self.layer_height
            processed_lines.append(
                f"G1 Z{base_z:.3f} F9000 ; Reset top layer to nominal perimeter height\n"
            )
            logging.info(
                "Reset top layer Z to nominal perimeter height on layer %d",
                self.current_layer,
            )
            if self.current_infill_coords:
                self.finalize_current_infill_region()

        with open(output_path, "w") as outfile:
            outfile.writelines(processed_lines)
        logging.info("Processed G-code file written to: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Bricklayering GCode Post-Processor with Hybrid Geometry, Auto Layer-Height Detection, and Enhanced Logging"
    )
    parser.add_argument(
        "gcode_file", help="Path to the G-code file generated by PrusaSlicer"
    )
    parser.add_argument(
        "--layer-height",
        type=float,
        default=None,
        help="Layer height in mm. If not provided, the script will auto-detect it.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Determine layer height automatically if not provided.
    if args.layer_height is None:
        logging.info("Layer height not provided; attempting auto-detection.")
        detected_layer_height = auto_detect_layer_height(args.gcode_file)
        if detected_layer_height is None:
            logging.error("Failed to auto-detect layer height. Exiting.")
            sys.exit(1)
        args.layer_height = detected_layer_height

    logging.info("Using layer height: %.3f mm", args.layer_height)

    try:
        processor = BricklayeringProcessor(layer_height=args.layer_height)
        # Process the file, overwriting the input file (or specify a separate output as needed).
        processor.process_file(args.gcode_file, args.gcode_file)
        logging.info("Successfully processed file: %s", args.gcode_file)
    except Exception as e:
        logging.error("Processing failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
