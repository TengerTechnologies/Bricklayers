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
import re
import sys
import logging
import os
import argparse

def detect_printer_type(gcode_lines):
    """Detect printer type based on G-code features"""
    for line in gcode_lines[:100]:  # Check first 100 lines should be enough
        if "; FEATURE:" in line:
            return "bambu"
        elif ";TYPE:" in line:
            return "prusa"
    return "unknown"

def get_wall_type(line, printer_type):
    """Determine wall type based on printer type"""
    if printer_type == "bambu":
        if "; FEATURE: Outer wall" in line:
            return "external"
        elif "; FEATURE: Inner wall" in line:
            return "internal"
        elif "; FEATURE:" in line:
            return None
    else:  # PrusaSlicer
        if ";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line:
            return "external"
        elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
            return "internal"
        elif ";TYPE:" in line:
            return None
    return None

def process_gcode(input_file, layer_height, extrusion_multiplier):
    # Configure logging
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

    # Read the input G-code
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Detect printer type
    printer_type = detect_printer_type(lines)
    logging.info(f"Detected printer type: {printer_type}")

    # First pass: analyze the file structure
    feature_types = set()
    for line in lines:
        if printer_type == "bambu" and "; FEATURE:" in line:
            feature_types.add(line.strip())
        elif printer_type == "prusa" and ";TYPE:" in line:
            feature_types.add(line.strip())
            
    if feature_types:
        logging.info(f"Detected features: {', '.join(feature_types)}")

    # Identify total layers
    total_layers = sum(1 for line in lines if line.startswith("G1 Z"))
    logging.info(f"Total layers: {total_layers}")

    # Process the G-code
    modified_lines = []
    shifted_blocks = 0
    
    for line_num, line in enumerate(lines):
        # Detect layer changes
        if line.startswith("G1 Z"):
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                current_z = float(z_match.group(1))
                current_layer = int(current_z / layer_height)
                perimeter_block_count = 0
            modified_lines.append(line)
            continue

        # Detect perimeter types based on printer type
        wall_type = get_wall_type(line, printer_type)
        if wall_type is not None:
            perimeter_type = wall_type
            inside_perimeter_block = False
            if wall_type == "internal":
                perimeter_found = True

        # Group lines into perimeter blocks
        if perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line:
            if "E" in line:  # Extrusion move
                # Start a new perimeter block if not already inside one
                if not inside_perimeter_block:
                    perimeter_block_count += 1
                    inside_perimeter_block = True
                    is_shifted = perimeter_block_count % 2 == 1

                    if is_shifted:  # Apply Z-shift to odd-numbered blocks
                        shifted_blocks += 1
                        adjusted_z = current_z + z_shift
                        modified_lines.append(f"G1 Z{adjusted_z:.3f} ; Shifted Z for block #{perimeter_block_count}\n")
                    else:  # Reset to the true layer height for even-numbered blocks
                        modified_lines.append(f"G1 Z{current_z:.3f} ; Reset Z for block #{perimeter_block_count}\n")

                # Adjust extrusion for shifted blocks
                if is_shifted:
                    e_match = re.search(r'E([-\d.]+)', line)
                    if e_match:
                        e_value = float(e_match.group(1))
                        if current_layer == 0:
                            new_e_value = e_value * 1.5
                        elif current_layer == total_layers - 1:
                            new_e_value = e_value * 0.5
                        else:
                            new_e_value = e_value * extrusion_multiplier
                        
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line.strip())
                        line += f" ; Adjusted E for block #{perimeter_block_count}\n"

            elif "F" in line and not "E" in line:  # Move without extrusion - end of block
                inside_perimeter_block = False

        modified_lines.append(line)

    if not perimeter_found:
        logging.warning(f"No internal perimeters found in the file. Printer type: {printer_type}")
    else:
        logging.info(f"Processing complete: Modified {shifted_blocks} blocks across {total_layers} layers")

    # Overwrite the input file with the modified G-code
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
    process_gcode(
        input_file=args.input_file,
        layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier,
    )
