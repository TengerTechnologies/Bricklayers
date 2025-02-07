#!/usr/bin/env python3
import re
import sys
import os
import mmap
import socket
import logging
import argparse
import concurrent.futures
from datetime import datetime, timezone
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPoint, box
import math

# Pre-compile regex patterns as class attributes.
RE_LAYER_Z = re.compile(r"Z([\d\.]+)")
RE_LAYER_X = re.compile(r"X([\d\.]+)")
RE_LAYER_Y = re.compile(r"Y([\d\.]+)")
RE_LAYER_E = re.compile(r"E([\d\.]+)")
RE_NOZZLE = re.compile(r"([\d\.]+)\s*mm", re.IGNORECASE)
RE_LAYER_HEIGHT = re.compile(r"Z([\d\.]+)")


# -----------------------------------------------------------------------------#
# RFC-5424 compliant formatter for multi-handler logging
# -----------------------------------------------------------------------------
class RFC5424Formatter(logging.Formatter):
    def format(self, record):
        # RFC5424 format: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID MSG
        # This is a simplified version.
        hostname = socket.gethostname()
        appname = "Bricklayer"
        pid = record.process
        # For PRI, we use a fixed dummy value (14) based on severity.
        pri = "<14>"
        timestamp = (
            datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        version = "1"
        msgid = "-"
        message = record.getMessage()
        return (
            f"{pri}{version} {timestamp} {hostname} {appname} {pid} {msgid} {message}"
        )


# -----------------------------------------------------------------------------#
# Setup logging to file and console using RFC5424 formatting.
# -----------------------------------------------------------------------------
def setup_logging(level_str):
    numeric_level = getattr(logging, level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_str}")

    # Create a logs subdirectory next to this script.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"bricklayer_{timestamp}.log")

    # Create two handlers: one to file and one to console.
    file_handler = logging.FileHandler(log_filename, mode="a")
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = RFC5424Formatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=numeric_level, handlers=[file_handler, console_handler])
    logging.info(
        "Logging initialized at level %s. Log file: %s", level_str.upper(), log_filename
    )


# -----------------------------------------------------------------------------#
# Auto-detect layer height by scanning for unique G1 Z moves.
# -----------------------------------------------------------------------------
def auto_detect_layer_height(file_path):
    z_values = set()
    try:
        # Use mmap-based file reading.
        with open(file_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = mm.read().decode("utf-8", errors="replace")
            mm.close()
        for line in data.splitlines():
            if line.startswith("G1 ") and "Z" in line:
                m = RE_LAYER_Z.search(line)
                if m:
                    try:
                        z_val = float(m.group(1))
                        z_values.add(round(z_val, 5))
                    except ValueError:
                        continue
    except Exception as e:
        logging.error("Error reading file for auto-detecting layer height: %s", e)
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


# -----------------------------------------------------------------------------#
# Auto-detect nozzle diameter by scanning for comments.
# -----------------------------------------------------------------------------
def auto_detect_nozzle_diameter(file_path):
    default = 0.4
    try:
        with open(file_path, "r") as f:
            for line in f:
                if "nozzle" in line.lower() and "diameter" in line.lower():
                    match = RE_NOZZLE.search(line)
                    if match:
                        d = float(match.group(1))
                        logging.info("Auto-detected nozzle diameter: %.3f mm", d)
                        return d
    except Exception as e:
        logging.error("Error auto-detecting nozzle diameter: %s", e)
    logging.info("Using default nozzle diameter: %.3f mm", default)
    return default


# -----------------------------------------------------------------------------#
# BGCode file decoding support stub.
# If a file has a .bgc extension, assume it is BGCode and decode it into standard G-code.
# -----------------------------------------------------------------------------
def decode_bgcode(file_path):
    logging.info("Decoding BGCode file: %s", file_path)
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
            # This is a stub decoder. Replace this with a real BGCode decode routine.
            decoded = raw.decode("utf-8", errors="replace")
            lines = decoded.splitlines(keepends=True)
            return lines
    except Exception as e:
        logging.error("Error decoding BGCode file: %s", e)
        sys.exit(1)


# -----------------------------------------------------------------------------#
# The main bricklayering processor class.
# -----------------------------------------------------------------------------
class BricklayeringProcessor:
    def __init__(
        self,
        layer_height,
        overall_extrusion_multiplier=1.0,
        first_layer_extrusion_multiplier=1.5,
        last_layer_extrusion_multiplier=1.0,
        nozzle_diameter=0.4,
        parallel_processing=False,
    ):
        self.layer_height = layer_height
        self.overall_extrusion_multiplier = overall_extrusion_multiplier
        self.first_layer_extrusion_multiplier = first_layer_extrusion_multiplier
        self.last_layer_extrusion_multiplier = last_layer_extrusion_multiplier
        self.nozzle_diameter = nozzle_diameter
        self.parallel_processing = parallel_processing

        self.current_z = 0.0
        self.current_layer = 0
        self.current_section = "perimeter"  # "perimeter" or "non_perimeter"
        self.extrusion_mode = "absolute"
        self.current_last_layer = False  # Set if a ";LAST_LAYER" comment is seen.
        self.awaiting_layer_change = False

        # Slicer-defined types.
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

        # Infill properties.
        self.infill_offset = (
            self.first_layer_extrusion_multiplier - 1.0
        ) * self.layer_height
        logging.info(
            "Extrusion multipliers (overall=%.2f, first=%.2f, last=%.2f); Infill offset=%.3f mm",
            self.overall_extrusion_multiplier,
            self.first_layer_extrusion_multiplier,
            self.last_layer_extrusion_multiplier,
            self.infill_offset,
        )

        self.layer_infill_regions = (
            {}
        )  # {layer: [ { "bbox":..., "polygon":... }, ... ]}
        self.current_infill_coords = []
        self.pending_z_shift = False

        # If parallel processing is enabled, set up a ThreadPoolExecutor.
        if self.parallel_processing:
            self.pool = concurrent.futures.ThreadPoolExecutor()
            logging.info("Parallel processing enabled: Using ThreadPoolExecutor.")
        else:
            self.pool = None

    # -------------------------------------------------------------------------#
    # Compute bounding box for a list of (x,y) tuples.
    # -------------------------------------------------------------------------#
    def compute_bbox(self, coords):
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        logging.debug("Computed bbox: %s for %d points", bbox, len(coords))
        return bbox

    # -------------------------------------------------------------------------#
    # A helper method to build a polygon from coordinates.
    # This is designed to be executed in parallel.
    # -------------------------------------------------------------------------#
    def build_polygon(self, coords):
        bbox = self.compute_bbox(coords)
        if len(coords) < 3:
            logging.warning(
                "Insufficient points (%d) for polygon; using bbox.", len(coords)
            )
            return bbox, box(*bbox)
        try:
            poly = Polygon(coords)
            if not poly.is_valid or poly.area == 0:
                logging.debug("Polygon invalid or zero-area; using convex hull.")
                poly = MultiPoint(coords).convex_hull
                if poly.geom_type != "Polygon":
                    poly = box(*bbox)
        except Exception as e:
            logging.error("Error creating polygon: %s", e)
            poly = box(*bbox)
        return bbox, poly

    # -------------------------------------------------------------------------#
    # Merge two regions by combining their coordinate lists. Build a new bbox and polygon.
    # -------------------------------------------------------------------------#
    def merge_two_regions(self, region1, region2):
        merged_coords = region1.get("coords", []) + region2.get("coords", [])
        bbox, poly = self.build_polygon(merged_coords)
        merged_region = {"bbox": bbox, "polygon": poly, "coords": merged_coords}
        return merged_region

    # -------------------------------------------------------------------------#
    # Check if two bounding boxes are sufficiently close or overlapping.
    # Adjust tolerance as needed (in mm).
    # -------------------------------------------------------------------------#
    def bboxes_are_close(self, bbox1, bbox2, tolerance=0.01):
        minx1, miny1, maxx1, maxy1 = bbox1
        minx2, miny2, maxx2, maxy2 = bbox2
        # They are far apart if one is completely to the left/right or above/below the other minus a tolerance.
        if maxx1 < (minx2 - tolerance) or maxx2 < (minx1 - tolerance):
            return False
        if maxy1 < (miny2 - tolerance) or maxy2 < (miny1 - tolerance):
            return False
        return True

    # -------------------------------------------------------------------------#
    # After processing a layer, merge adjacent regions that have overlapping (or close) bounding boxes.
    # -------------------------------------------------------------------------#
    def merge_adjacent_regions(self, layer):
        regions = self.layer_infill_regions.get(layer, [])
        if not regions:
            return
        merged = []
        for region in regions:
            if not merged:
                merged.append(region)
            else:
                last_region = merged[-1]
                if self.bboxes_are_close(last_region["bbox"], region["bbox"]):
                    new_region = self.merge_two_regions(last_region, region)
                    merged[-1] = new_region
                    logging.debug(
                        "Merged two regions on layer %d; new bbox: %s",
                        layer,
                        new_region["bbox"],
                    )
                else:
                    merged.append(region)
        self.layer_infill_regions[layer] = merged

    # -------------------------------------------------------------------------#
    # Finalize the current non-perimeter infill region.
    # If parallel processing is enabled, submit the polygon building to the thread pool.
    # Modified here: we save the raw collected coordinates in the region dictionary.
    # Also, if the number of collected points is less than a threshold, merge them with the preceding region.
    # -------------------------------------------------------------------------#
    def finalize_current_infill_region(self):
        if not self.current_infill_coords:
            return

        coords = self.current_infill_coords.copy()

        # If too few points are collected, attempt to merge with the last finalized region
        if len(coords) < 3:
            if (
                self.current_layer in self.layer_infill_regions
                and self.layer_infill_regions[self.current_layer]
            ):
                last_region = self.layer_infill_regions[self.current_layer][-1]
                merged_region = self.merge_two_regions(last_region, {"coords": coords})
                # Replace the last region with the merged one.
                self.layer_infill_regions[self.current_layer][-1] = merged_region
                logging.info(
                    "Merged a very small region (only %d points) with the previous region on layer %d.",
                    len(coords),
                    self.current_layer,
                )
            else:
                # Not much to merge with; build the region anyway.
                bbox, poly = self.build_polygon(coords)
                region = {"bbox": bbox, "polygon": poly, "coords": coords}
                self.layer_infill_regions.setdefault(self.current_layer, []).append(
                    region
                )
        else:
            bbox, poly = self.build_polygon(coords)
            region = {"bbox": bbox, "polygon": poly, "coords": coords}
            self.layer_infill_regions.setdefault(self.current_layer, []).append(region)

        logging.info(
            "Finalized infill region on layer %d; now %d region(s) on this layer.",
            self.current_layer,
            len(self.layer_infill_regions[self.current_layer]),
        )
        self.current_infill_coords = []
        self.pending_z_shift = False

        # Optionally, merge adjacent regions for this layer after each finalization.
        self.merge_adjacent_regions(self.current_layer)

    # -------------------------------------------------------------------------#
    # Check if the point (x,y) is inside any infill region from the previous layer.
    # -------------------------------------------------------------------------#
    def _previous_infill_at(self, x, y):
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
            "Point (%.3f, %.3f) not in any region on previous layer %d.",
            x,
            y,
            prev_layer,
        )
        return False

    # -------------------------------------------------------------------------#
    # Process a single G-code line.
    # Inject annotation comments and adjust extrusion values as needed.
    # -------------------------------------------------------------------------#
    def process_line(self, line):
        output = []

        # Process layer change comments first.
        if line.startswith(";LAYER_CHANGE"):
            self.awaiting_layer_change = True
            output.append(line)
            return output

        if self.awaiting_layer_change and line.startswith(";Z:"):
            try:
                z_val = float(line.split(":", 1)[1].strip())
                new_layer = int(round(z_val / self.layer_height))
                if new_layer != self.current_layer and self.current_infill_coords:
                    self.finalize_current_infill_region()
                self.current_z = z_val
                self.current_layer = new_layer
                logging.info(
                    "Layer change via comment: now at layer %d (Z=%.3f)",
                    new_layer,
                    z_val,
                )
            except Exception as e:
                logging.error("Error parsing layer Z from comment: %s", e)
            output.append(line)
            self.awaiting_layer_change = False
            return output

        # Handle G1 Z moves.
        if line.startswith("G1 ") and "Z" in line:
            if ";TEMP_Z" in line:
                logging.debug("Processing temporary/reset Z move: %s", line.strip())
                output.append(line)
            else:
                logging.debug("Ignoring unflagged G1 Z move: %s", line.strip())
                output.append(line)
            # Note: Do not update internal layer state here.
            # (Genuine layer changes should be handled via the comments.)
            return output

        # if line.startswith("G1 ") and "Z" in line:
        #     logging.debug("Processing Z move: %s", line.strip())
        #     # Ignore temporary shifts and resets from affecting layer state.
        #     if ";TEMP_Z" in line or "Reset to nominal" in line:
        #         logging.debug("Ignoring temporary/reset Z move: %s", line.strip())
        #         output.append(line)
        #     else:
        #         m = RE_LAYER_Z.search(line)
        #         if m:
        #             new_z = float(m.group(1))
        #             new_layer = int(round(new_z / self.layer_height))
        #             if new_layer != self.current_layer:
        #                 if self.current_infill_coords:
        #                     self.finalize_current_infill_region()
        #                 self.current_z = new_z
        #                 self.current_layer = new_layer
        #                 logging.info(
        #                     "Layer change: now at layer %d (Z=%.3f)",
        #                     self.current_layer,
        #                     self.current_z,
        #                 )
        #             else:
        #                 self.current_z = new_z
        #         output.append(line)

        # Extrusion mode detection.
        if line.startswith("M82"):
            self.extrusion_mode = "absolute"
            logging.debug("Extrusion mode set to absolute.")
        elif line.startswith("M83"):
            self.extrusion_mode = "relative"
            logging.debug("Extrusion mode set to relative.")

        # Detect special comments.
        if line.startswith(";TYPE:"):
            section_type = line.split(":")[-1].strip()
            new_section = (
                "non_perimeter"
                if section_type in self.non_perimeter_types
                else "perimeter"
            )
            if new_section != self.current_section:
                logging.info(
                    "Section change: %s -> %s", self.current_section, new_section
                )
                self.handle_section_change(new_section, output)
                self.current_section = new_section

        if line.strip().startswith(";LAST_LAYER"):
            self.current_last_layer = True
            logging.info("Detected last layer marker.")
            output.append(line)
            return output

        # Process non-perimeter (infill) moves.
        if line.startswith("G1 ") and self.current_section == "non_perimeter":
            if self.pending_z_shift:
                m_x = RE_LAYER_X.search(line)
                m_y = RE_LAYER_Y.search(line)
                if m_x and m_y:
                    x_val = float(m_x.group(1))
                    y_val = float(m_y.group(1))
                    base_z = self.current_layer * self.layer_height
                    if self.current_layer >= 2 and self._previous_infill_at(
                        x_val, y_val
                    ):
                        new_z = base_z + self.infill_offset
                        note = "Aligned with infill: raising Z"
                    else:
                        new_z = base_z
                        note = "No underlying infill: Z unchanged"
                    out_line = f"G1 Z{new_z:.3f} F9000 ; {note} ;TEMP_Z\n"
                    output.append(out_line)
                    logging.info(
                        "Layer %d non-perimeter: first XY (%.3f, %.3f) => %s (Z=%.3f)",
                        self.current_layer,
                        x_val,
                        y_val,
                        note,
                        new_z,
                    )
                    self.pending_z_shift = False

            m_x = RE_LAYER_X.search(line)
            m_y = RE_LAYER_Y.search(line)
            if m_x and m_y:
                x_val = float(m_x.group(1))
                y_val = float(m_y.group(1))
                self.current_infill_coords.append((x_val, y_val))
                logging.debug("Collected infill point: (%.3f, %.3f)", x_val, y_val)

        if "E" in line and "G1" in line:
            old_line = line.strip()
            new_line = self.adjust_extrusion(line)
            if old_line != new_line.strip():
                logging.debug(
                    "Extrusion adjusted: '%s' -> '%s'", old_line, new_line.strip()
                )
                line = new_line
            else:
                line = new_line

        output.append(line)
        return output

    # -------------------------------------------------------------------------#
    # Depending on section change, finalize regions and possibly reset Z.
    # -------------------------------------------------------------------------#
    def handle_section_change(self, new_section, output):
        base_z = self.current_layer * self.layer_height
        if self.current_layer < 1:
            return
        if self.current_section == "non_perimeter" and new_section == "perimeter":
            if self.current_infill_coords:
                self.finalize_current_infill_region()
            if self.current_layer >= 2:
                out_cmd = f"G1 Z{base_z:.3f} F9000 ; Reset to nominal perimeter height ;TEMP_Z\n"
                output.append(out_cmd)
                logging.info(
                    "Reset Z to nominal perimeter height on layer %d",
                    self.current_layer,
                )
            # Reset last-layer flag when leaving non-perimeter.
            self.current_last_layer = False
        elif new_section == "non_perimeter":
            if self.current_layer == 1:
                logging.info(
                    "Entering first-layer non-perimeter; applying first-layer extrusion adjustments."
                )
            elif self.current_layer >= 2:
                self.pending_z_shift = True
                logging.info(
                    "Entering non-perimeter on layer %d; awaiting first XY for Z test.",
                    self.current_layer,
                )

    # -------------------------------------------------------------------------#
    # Adjust the extrusion value based on configurable multipliers.
    # Uses first_layer_extrusion_multiplier if on layer 1,
    # last_layer_extrusion_multiplier if the last layer is marked,
    # otherwise overall_extrusion_multiplier.
    # -------------------------------------------------------------------------#
    def adjust_extrusion(self, line):
        """Adjust the extrusion value in G-code if necessary,
        leaving the line unchanged if the result is numerically identical."""
        m = RE_LAYER_E.search(line)
        if not m:
            return line
        try:
            e_val = float(m.group(1))
        except ValueError:
            return line

        # Determine the extrusion multiplier based on layer and section.
        if self.current_layer == 1 and self.current_section == "perimeter":
            multiplier = 1.0  # Do not adjust first-layer perimeter moves.
        elif self.current_layer == 1:
            multiplier = self.first_layer_extrusion_multiplier
        elif self.current_last_layer:
            multiplier = self.last_layer_extrusion_multiplier
        else:
            multiplier = self.overall_extrusion_multiplier

        # If the multiplier is effectively 1.0, then no adjustment is needed.
        if math.isclose(multiplier, 1.0, rel_tol=1e-9):
            return line

        new_e = e_val * multiplier

        # If the new extrusion value is numerically the same as the original, do nothing.
        if math.isclose(new_e, e_val, rel_tol=1e-9):
            return line

        # Otherwise, perform the replacement.
        new_line = re.sub(r"E[\d\.]+", f"E{new_e:.5f}", line)
        new_line = new_line.rstrip("\n") + " ; extrusion adjusted\n"
        return new_line

    # -------------------------------------------------------------------------#
    # Process the entire G-code file.
    # Uses mmap for file reading, and includes basic file-size checks.
    # -------------------------------------------------------------------------#
    def process_file(self, input_path, output_path):
        processed_lines = []
        try:
            file_size = os.path.getsize(input_path)
            max_size = 500 * 1024 * 1024  # 500 MB limit for safety.
            if file_size > max_size:
                logging.error(
                    "File size (%.2f MB) exceeds safe processing limits.",
                    file_size / (1024 * 1024),
                )
                sys.exit(1)
        except Exception as e:
            logging.error("Failed to get file size: %s", e)
            sys.exit(1)

        # Use mmap-based reading.
        try:
            with open(input_path, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                data = mm.read().decode("utf-8", errors="replace")
                lines = data.splitlines(keepends=True)
                mm.close()
        except Exception as e:
            logging.error("Error during mmap file processing: %s", e)
            sys.exit(1)

        logging.info("Opened file: %s (%d lines)", input_path, len(lines))
        for line in lines:
            processed = self.process_line(line)
            processed_lines.extend(processed)

        # Finalize any open infill region on file end.
        if self.current_section == "non_perimeter":
            base_z = self.current_layer * self.layer_height
            processed_lines.append(
                f"G1 Z{base_z:.3f} F9000 ; Reset top layer to nominal perimeter height ;TEMP_Z\n"
            )
            logging.info(
                "Reset top layer Z to nominal perimeter height on layer %d",
                self.current_layer,
            )
            if self.current_infill_coords:
                self.finalize_current_infill_region()

        try:
            with open(output_path, "w") as outfile:
                outfile.writelines(processed_lines)
            logging.info("Processed G-code file written to: %s", output_path)
        except Exception as e:
            logging.error("Error writing output file: %s", e)
            sys.exit(1)

        # Shutdown parallel pool if created.
        if self.pool:
            self.pool.shutdown()


# -----------------------------------------------------------------------------#
# Main entry point.
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Bricklayering GCode Post-Processor with numerous features:\n"
        "  - Configurable extrusion multipliers (overall/first/last layer)\n"
        "  - BGCode file decoding support\n"
        "  - Extrusion mode detection (absolute/relative)\n"
        "  - Automatic nozzle diameter detection\n"
        "  - Annotation comments for changes\n"
        "  - Buffer overflow & safety features\n"
        "  - MMap-based file processing\n"
        "  - Optional parallel geometry processing\n"
        "  - Multi-handler RFC-5424 compliant logging"
    )
    parser.add_argument("gcode_file", help="Path to the G-code (or BGCode) file")
    parser.add_argument(
        "--layer-height",
        type=float,
        default=None,
        help="Layer height in mm. If not provided, auto-detection is attempted.",
    )
    parser.add_argument(
        "--overall-extrusion-multiplier",
        type=float,
        default=1.0,
        help="Extrusion multiplier applied to non-first/last layers (default=1.0)",
    )
    parser.add_argument(
        "--first-layer-extrusion-multiplier",
        type=float,
        default=1.5,
        help="Extrusion multiplier for the first layer (default=1.5)",
    )
    parser.add_argument(
        "--last-layer-extrusion-multiplier",
        type=float,
        default=1.0,
        help="Extrusion multiplier for the last layer (default=1.0)",
    )
    parser.add_argument(
        "--nozzle-diameter",
        type=float,
        default=None,
        help="Nozzle diameter in mm. If not provided, auto-detection is attempted.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing for geometry calculations.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Check file extension for BGCode.
    file_path = args.gcode_file
    if file_path.lower().endswith((".bgc", ".bgcode")):
        lines = decode_bgcode(file_path)
        # Write decoded lines to a temporary file for processing.
        tmp_path = file_path + ".decoded.gcode"
        with open(tmp_path, "w") as tmp_file:
            tmp_file.writelines(lines)
        file_path = tmp_path
        logging.info("BGCode decoded to temporary G-code file: %s", tmp_path)

    # Auto-detect layer height if not provided.
    if args.layer_height is None:
        logging.info("Layer height not provided; attempting auto-detection.")
        detected = auto_detect_layer_height(file_path)
        if detected is None:
            logging.error("Failed to auto-detect layer height. Exiting.")
            sys.exit(1)
        args.layer_height = detected
    logging.info("Using layer height: %.3f mm", args.layer_height)

    # Auto-detect nozzle diameter if not provided.
    if args.nozzle_diameter is None:
        args.nozzle_diameter = auto_detect_nozzle_diameter(file_path)
    logging.info("Using nozzle diameter: %.3f mm", args.nozzle_diameter)

    try:
        processor = BricklayeringProcessor(
            layer_height=args.layer_height,
            overall_extrusion_multiplier=args.overall_extrusion_multiplier,
            first_layer_extrusion_multiplier=args.first_layer_extrusion_multiplier,
            last_layer_extrusion_multiplier=args.last_layer_extrusion_multiplier,
            nozzle_diameter=args.nozzle_diameter,
            parallel_processing=args.parallel,
        )
        # Process the G-code file. (The output overwrites the input file. Adjust as desired.)
        processor.process_file(file_path, file_path)
        logging.info("Successfully processed file: %s", file_path)
    except Exception as e:
        logging.error("Processing failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
