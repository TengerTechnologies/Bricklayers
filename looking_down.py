"""
Below is one approach to “look one layer down” so that non‐perimeter (infill) moves get a 
z‐shift only if the same XY area was printed with a z-offset in the layer below. The idea 
is to:
    1. Track the XY coordinates (or a summary of them) for every non‐perimeter section as 
       you process a layer. In our example, we do this by collecting all “G1” moves in a 
       non‑perimeter region. When that region ends (or at a layer change) we combine its points into a simple bounding box.
       
    2. Save these bounding boxes for each layer (for example, in a dictionary keyed by
       layer number).
       
    3. Then, when processing a layer “n” for a non‑perimeter region, wait for the very
       first move that has XY data and check: Is that point (or its small bounding box)
       inside one o f the non‑perimeter bounding boxes from the previous layer
       (layer n–1)? If yes, output a z move that raises that extruded infill (e.g. base_z
       + layer_height/2). Otherwise, leave z at the standard “base” height.
"""

import re
import sys
import logging
import argparse


class BricklayeringProcessor:
    def __init__(self, layer_height=0.2):
        self.layer_height = layer_height
        self.current_z = 0.0
        self.current_layer = 0
        self.current_section = "perimeter"
        self.extrusion_mode = "absolute"

        # Section types (as defined by PrusaSlicer comments)
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

        # These variables help track the geometry of infill areas:
        # For each layer, we’ll store one or more bounding boxes (min_x,min_y,max_x,max_y) for non-perimeter moves.
        # When processing a new non-perimeter section, we check the first XY coordinate against the bounding
        # boxes of the previous layer.
        self.layer_infill_regions = (
            {}
        )  # {layer_number: [ (minx, miny, maxx, maxy), ... ]}
        self.current_infill_coords = (
            []
        )  # list of (x, y) tuples for current non-perimeter region
        self.pending_z_shift = (
            False  # True when a new non-perimeter region has just begun
        )

    def compute_bbox(self, coords):
        """Given a list of (x, y) points, return a bounding box (minx, miny, maxx, maxy)."""
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        return (min(xs), min(ys), max(xs), max(ys))

    def _previous_infill_at(self, x, y):
        """Check if the given (x,y) falls within any bounding box recorded for the non-perimeter
        (infill) regions of the previous layer."""
        prev_layer = self.current_layer - 1
        if prev_layer < 1:
            return False
        regions = self.layer_infill_regions.get(prev_layer, [])
        for bbox in regions:
            minx, miny, maxx, maxy = bbox
            if minx <= x <= maxx and miny <= y <= maxy:
                return True
        return False

    def process_line(self, line):
        output = []
        original_line = line.strip()

        # --- Detect a layer change via a Z move ---
        # When a new layer begins, if there are any pending non-perimeter (infill) moves,
        # finalize their geometry (here, store a bounding box for that area).
        if line.startswith("G1 ") and "Z" in line:
            z_match = re.search(r"Z([\d\.]+)", line)
            if z_match:
                new_z = float(z_match.group(1))
                new_layer = int(round(new_z / self.layer_height))
                if new_layer != self.current_layer:
                    if self.current_infill_coords:
                        bbox = self.compute_bbox(self.current_infill_coords)
                        self.layer_infill_regions.setdefault(
                            self.current_layer, []
                        ).append(bbox)
                        self.current_infill_coords = []
                        self.pending_z_shift = False
                    self.current_z = new_z
                    self.current_layer = new_layer
                    logging.debug(
                        f"Layer update: {self.current_layer}, Z={self.current_z}"
                    )
                else:
                    self.current_z = new_z

        # --- Handle extrusion mode commands ---
        if line.startswith("M82"):
            self.extrusion_mode = "absolute"
        elif line.startswith("M83"):
            self.extrusion_mode = "relative"

        # --- Handle section type changes based on comments ---
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
            # If we are starting a new non-perimeter region (pending flag is set) and the
            # current line includes X/Y position, decide whether we need the z-height shift.
            if self.pending_z_shift:
                x_match = re.search(r"X([\d\.]+)", line)
                y_match = re.search(r"Y([\d\.]+)", line)
                if x_match and y_match:
                    x_val = float(x_match.group(1))
                    y_val = float(y_match.group(1))
                    base_z = self.current_layer * self.layer_height
                    # Only for layers 2 and above do we try aligning with layer below:
                    if self.current_layer >= 2 and self._previous_infill_at(
                        x_val, y_val
                    ):
                        new_z = base_z + (self.layer_height / 2)
                        note = "Non-perimeter height shift (aligned with infill below)"
                    else:
                        new_z = base_z
                        note = "Non-perimeter no shift (no aligned infill below)"
                    output.append(f"G1 Z{new_z:.3f} F9000 ; {note}\n")
                    self.pending_z_shift = False

            # Also record the XY coordinate from this G1 line (if available) for geometry tracking.
            x_match = re.search(r"X([\d\.]+)", line)
            y_match = re.search(r"Y([\d\.]+)", line)
            if x_match and y_match:
                x_val = float(x_match.group(1))
                y_val = float(y_match.group(1))
                self.current_infill_coords.append((x_val, y_val))

        # --- Process extrusion moves on the first layer for non-perimeter ---
        # (brick layer effect on first layer is handled via adjusting the extrusion E value)
        if (
            "E" in line
            and self.current_section == "non_perimeter"
            and self.current_layer == 1
        ):
            line = self.adjust_extrusion(line)

        output.append(line)
        return output

    def handle_section_change(self, new_section, output):
        base_z = self.current_layer * self.layer_height
        if self.current_layer < 1:
            return

        # If we are leaving a non-perimeter (infill) region, finalize the region data.
        if self.current_section == "non_perimeter" and new_section == "perimeter":
            if self.current_infill_coords:
                bbox = self.compute_bbox(self.current_infill_coords)
                self.layer_infill_regions.setdefault(self.current_layer, []).append(
                    bbox
                )
                self.current_infill_coords = []
            self.pending_z_shift = False
            if self.current_layer >= 2:
                output.append(f"G1 Z{base_z:.3f} F9000 ; Reset to perimeter height\n")
        # When entering a non-perimeter section, set a flag so that the very first move’s XY can be
        # checked against the infill geometry from below.
        elif new_section == "non_perimeter":
            if self.current_layer == 1:
                logging.debug("First layer non-perimeter - extrusion adjustment only")
            elif self.current_layer >= 2:
                self.pending_z_shift = True

    def adjust_extrusion(self, line):
        """For the first layer non-perimeter, do an extrusion adjustment rather than a z-height shift."""
        e_match = re.search(r"E([\d\.]+)", line)
        if not e_match:
            return line

        e_value = float(e_match.group(1))
        new_e = e_value * 1.5  # 150% extrusion for infill on the base layer
        return re.sub(r"E[\d\.]+", f"E{new_e:.5f}", line)

    def process_file(self, input_path, output_path):
        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            for line in infile:
                processed = self.process_line(line)
                outfile.writelines(processed)


def main():
    parser = argparse.ArgumentParser(description="Bricklayering GCode Post-Processor")
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
        # Read the input file
        with open(args.gcode_file, "r") as f:
            lines = f.readlines()

        # Process each line
        processed_lines = []
        for line in lines:
            processed = processor.process_line(line)
            processed_lines.extend(processed)

        # Write back to the file (or use an output file)
        with open(args.gcode_file, "w") as f:
            f.writelines(processed_lines)

        logging.info(f"Successfully processed {args.gcode_file}")
    except Exception as e:
        logging.error(f"Failed to process file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
