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
#
import re
import sys
import logging
import os
import argparse

def detect_printer_type(gcode_lines):
    """Detect printer type based on G-code features"""
    logging.info("Starting printer type detection")
    
    for i, line in enumerate(gcode_lines):
        if "; FEATURE:" in line:
            logging.info(f"Detected Bambu/Orca printer from feature marker in line {i}: {line.strip()}")
            return "bambu"
        elif ";TYPE:" in line:
            logging.info(f"Detected Prusa printer from type marker in line {i}: {line.strip()}")
            return "prusa"
    
    logging.warning("No printer type markers found - defaulting to Prusa")
    return "prusa"

def get_z_height_from_comment(line):
    """Extract Z height from comment if present"""
    if "; Z_HEIGHT:" in line:
        match = re.search(r'; Z_HEIGHT: ([\d.]+)', line)
        if match:
            return float(match.group(1))
    return None

def process_gcode(input_file, layer_height, extrusion_multiplier):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "z_shift_log.txt")
    logging.basicConfig(
        filename=log_file_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

    current_layer = 0
    current_z = 0.0
    perimeter_type = None
    perimeter_block_count = 0
    inside_perimeter_block = False
    z_shift = layer_height * 0.5
    perimeter_found = False
    
    logging.info("Starting G-code processing")
    logging.info(f"Settings: Layer height={layer_height}mm, Z-shift={z_shift}mm, Extrusion multiplier={extrusion_multiplier}")

    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    printer_type = detect_printer_type(lines)
    logging.info(f"Detected printer type: {printer_type}")

    modified_lines = []
    shifted_blocks = 0
    in_object = False  # Track if we're inside an object printing section
    
    for line_num, line in enumerate(lines):
        # Track object printing sections
        if "M624" in line:  # Start printing object
            in_object = True
            perimeter_block_count = 0  # Reset block count for new object
        elif "M625" in line:  # Stop printing object
            in_object = False
            if inside_perimeter_block:
                modified_lines.append(f"G1 Z{current_z:.3f} F1200 ; Reset Z at object end\n")
                inside_perimeter_block = False

        # Check for layer changes and Z height updates
        if "; CHANGE_LAYER" in line:
            z_height = get_z_height_from_comment(lines[line_num + 1]) if line_num + 1 < len(lines) else None
            if z_height is not None:
                current_z = z_height
                current_layer += 1
                perimeter_block_count = 0
                logging.info(f"Layer change detected: Z={current_z:.3f}")

        # Handle perimeter detection and Z shifts only when actively printing object
        if in_object:
            # Detect wall transitions
            if "; FEATURE:" in line:
                # Reset Z height when transitioning between features
                if inside_perimeter_block:
                    modified_lines.append(f"G1 Z{current_z:.3f} F1200 ; Reset Z for feature transition\n")
                    inside_perimeter_block = False
                
                if "; FEATURE: Inner wall" in line:
                    perimeter_type = "internal"
                    perimeter_found = True
                elif "; FEATURE: Outer wall" in line:
                    perimeter_type = "external"
                    if inside_perimeter_block:
                        modified_lines.append(f"G1 Z{current_z:.3f} F1200 ; Reset Z for outer wall\n")
                        inside_perimeter_block = False
                else:
                    perimeter_type = None

            # Handle Z shifts for internal perimeters
            if perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line:
                if "E" in line:  # Extrusion move
                    if not inside_perimeter_block:
                        perimeter_block_count += 1
                        inside_perimeter_block = True
                        
                        # Apply Z shift for inner wall
                        adjusted_z = current_z + z_shift
                        modified_lines.append(f"G1 Z{adjusted_z:.3f} F1200 ; Z shift for inner wall\n")
                        shifted_blocks += 1
                        
                        # Adjust extrusion
                        e_match = re.search(r'E([-\d.]+)', line)
                        if e_match:
                            e_value = float(e_match.group(1))
                            new_e_value = e_value * extrusion_multiplier
                            line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line.strip())
                            line += f" ; Adjusted E for inner wall\n"
                
                elif "F" in line and not "E" in line and inside_perimeter_block:
                    modified_lines.append(f"G1 Z{current_z:.3f} F1200 ; Reset Z after inner wall\n")
                    inside_perimeter_block = False

        modified_lines.append(line)

    if not perimeter_found:
        logging.warning("No internal perimeters found in the file.")
    else:
        logging.info(f"Processing complete: Modified {shifted_blocks} blocks across {current_layer} layers")

    with open(input_file, 'w') as outfile:
        outfile.writelines(modified_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process G-code for Z-shifting and extrusion adjustments.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    parser.add_argument("-layerHeight", type=float, default=0.2, help="Layer height in mm (default: 0.2mm)")
    parser.add_argument("-extrusionMultiplier", type=float, default=1, help="Extrusion multiplier for first layer (default: 1.5x)")
    args = parser.parse_args()

    process_gcode(
        input_file=args.input_file,
        layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier,
    )
