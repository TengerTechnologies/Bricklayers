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

import re
import sys
import logging
import argparse
from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPoint


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

        # For first layer non-perimeter, we want to overextrude.
        # For example, a multiplier of 1.5 means that although our nominal layer height is 0.2 mm,
        # non-perimeter extrusions are printed 1.5× thicker (0.3 mm) so that the extra 0.1 mm
        # is propagated upward in layers where infill occurs.
        self.first_layer_infill_multiplier = 1.5
        self.infill_offset = (
            self.first_layer_infill_multiplier - 1.0
        ) * self.layer_height

        # For each layer, we store a list of infill region data.
        # Each region is a dict with keys "bbox" (minx,miny,maxx,maxy) and "polygon" (a shapely Polygon)
        self.layer_infill_regions = {}
        # For the current non-perimeter region, we collect (X,Y) points.
        self.current_infill_coords = []
        # In layers 2+, when entering a new non-perimeter section we wait for the first move with XY
        # data so we can test whether the region below had been overextruded.
        self.pending_z_shift = False

    def compute_bbox(self, coords):
        """Given a list of (x, y) points, compute the bounding box: (minx, miny, maxx, maxy)."""
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        return (min(xs), min(ys), max(xs), max(ys))

    def finalize_current_infill_region(self):
        """Finalize the geometry for the current non-perimeter (infill) region.

        Build a bounding box and a shapely Polygon from the XY coordinates and store them by layer.
        Afterwards, clear the collected coordinates and reset pending_z_shift.
        """
        if not self.current_infill_coords:
            return
        bbox = self.compute_bbox(self.current_infill_coords)
        try:
            poly = Polygon(self.current_infill_coords)
            if not poly.is_valid:
                # Fall back to the convex hull for self-intersecting polygons.
                poly = MultiPoint(self.current_infill_coords).convex_hull
        except Exception as e:
            logging.error("Error creating polygon: %s", e)
            poly = None
        region = {"bbox": bbox, "polygon": poly}
        self.layer_infill_regions.setdefault(self.current_layer, []).append(region)
        self.current_infill_coords = []
        self.pending_z_shift = False

    def _previous_infill_at(self, x, y):
        """
        Check if the (x, y) point lies within any non-perimeter (infill) region of the previous layer.
        This uses a fast bounding box check first, then a polygon intersection test.
        """
        prev_layer = self.current_layer - 1
        if prev_layer < 1:
            return False
        regions = self.layer_infill_regions.get(prev_layer, [])
        test_point = ShapelyPoint(x, y)
        for region in regions:
            minx, miny, maxx, maxy = region["bbox"]
            # Quick rejection if the point is not inside the bounding box.
            if not (minx <= x <= maxx and miny <= y <= maxy):
                continue
            poly = region["polygon"]
            if poly is not None and poly.contains(test_point):
                return True
        return False

    def process_line(self, line):
        output = []

        # --- Detect layer changes via G1 Z movements ---
        # When a new layer is started, finalize any pending non-perimeter geometry.
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
                    logging.debug(
                        "Layer update: %d, Z=%.3f", self.current_layer, self.current_z
                    )
                else:
                    self.current_z = new_z

        # --- Process extrusion mode commands ---
        if line.startswith("M82"):
            self.extrusion_mode = "absolute"
        elif line.startswith("M83"):
            self.extrusion_mode = "relative"

        # --- Handle section/type change comments from the slicer ---
        if line.startswith(";TYPE:"):
            section_type = line.split(":")[-1].strip()
            new_section = (
                "non_perimeter"
                if section_type in self.non_perimeter_types
                else "perimeter"
            )
            if new_section != self.current_section:
                self.handle_section_change(new_section, output)
                self.current_section = new_section

        # --- Process non-perimeter (infill) moves ---
        if line.startswith("G1 ") and self.current_section == "non_perimeter":
            # For layers 2 and above, if just beginning a non-perimeter region,
            # use the first XY move to decide whether to apply a z shift.
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
                        # Raise the infill moves above the underlying overextruded region.
                        new_z = base_z + self.infill_offset
                        note = "Aligned with infill below: Z-shift applied"
                    else:
                        new_z = base_z
                        note = "No aligned infill below: no Z-shift"
                    output.append(f"G1 Z{new_z:.3f} F9000 ; {note}\n")
                    self.pending_z_shift = False

            # Collect XY coordinates for building the region's geometry.
            x_match = re.search(r"X([\d\.]+)", line)
            y_match = re.search(r"Y([\d\.]+)", line)
            if x_match and y_match:
                x_val = float(x_match.group(1))
                y_val = float(y_match.group(1))
                self.current_infill_coords.append((x_val, y_val))

        # --- Process extrusion moves for the first layer non-perimeter ---
        if (
            "E" in line
            and self.current_section == "non_perimeter"
            and self.current_layer == 1
        ):
            line = self.adjust_extrusion(line)

        output.append(line)
        return output

    def handle_section_change(self, new_section, output):
        """
        When switching sections (for example, leaving non-perimeter infill to a perimeter region),
        finalize the current infill geometry. Also, if exiting an overextruded region in layers
        2+ (or the top layer), output a command to reset to the nominal (perimeter) z height so that
        the visible surface remains flat.
        """
        base_z = self.current_layer * self.layer_height
        if self.current_layer < 1:
            return

        # If leaving a non-perimeter (infill) region.
        if self.current_section == "non_perimeter" and new_section == "perimeter":
            if self.current_infill_coords:
                self.finalize_current_infill_region()
            if self.current_layer >= 2:
                # Reset z back to the nominal perimeter height.
                output.append(f"G1 Z{base_z:.3f} F9000 ; Reset to perimeter height\n")
        # When entering a non-perimeter region:
        elif new_section == "non_perimeter":
            if self.current_layer == 1:
                logging.debug("First layer non-perimeter; extrusion adjusted only")
            elif self.current_layer >= 2:
                # For a new infill region in later layers, trigger a z–test on the first XY move.
                self.pending_z_shift = True

    def adjust_extrusion(self, line):
        """
        For the first layer's non-perimeter moves, scale the extrusion value so that the extruded
        plastic is thicker (e.g. 150%). This is done only on the first layer so that the buildplate
        (bottom) stays flat.
        """
        e_match = re.search(r"E([\d\.]+)", line)
        if not e_match:
            return line
        e_value = float(e_match.group(1))
        new_e = e_value * self.first_layer_infill_multiplier
        return re.sub(r"E[\d\.]+", f"E{new_e:.5f}", line)

    def process_file(self, input_path, output_path):
        processed_lines = []
        with open(input_path, "r") as infile:
            for line in infile:
                processed = self.process_line(line)
                processed_lines.extend(processed)

        # --- Finalize the top layer.
        # If we end in a non-perimeter region, then output a final reset command so that the top
        # surface is flat.
        if self.current_section == "non_perimeter":
            base_z = self.current_layer * self.layer_height
            processed_lines.append(
                f"G1 Z{base_z:.3f} F9000 ; Reset top layer to perimeter height\n"
            )
            # Finalize any remaining infill region geometry
            if self.current_infill_coords:
                self.finalize_current_infill_region()

        with open(output_path, "w") as outfile:
            outfile.writelines(processed_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Bricklayering GCode Post-Processor with Hybrid Geometry and Adaptive Z-Shift"
    )
    parser.add_argument(
        "gcode_file", help="Path to the G-code file generated by PrusaSlicer"
    )
    parser.add_argument(
        "--layer-height", type=float, required=True, help="Layer height in mm"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="bricklayer.log",
        filemode="a",
    )

    try:
        processor = BricklayeringProcessor(layer_height=args.layer_height)
        # Process the G-code file.
        processor.process_file(args.gcode_file, args.gcode_file)
        logging.info("Successfully processed %s", args.gcode_file)
    except Exception as e:
        logging.error("Failed to process file: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
