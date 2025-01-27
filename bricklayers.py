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
import re
import sys
import logging
import os
import argparse
import subprocess
import shutil

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging to save in the script's directory
log_file_path = os.path.join(script_dir, "z_shift_log.txt")
logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Searches a line of the gcode file
# looking for comments which would indicate
# the layer height the model was sliced at
def get_layer_height_from_gcode(line):

    # Different printers specify layer heights in comments differently
    # so we'll need different regex patterns to search for
    layer_height_patterns = [
            r';\s*layer_height\s*=\s*(\d+(?:\.\d+)?)', # Prusa
            r';\s*Z_HEIGHT:\s*(\d+(?:\.\d+)?)', # Bambu Labs
            r';\s*layerHeight,(\d+(?:\.\d+)?)' # Simplify3D
        ]

    # Only check comments for matches
    if line.startswith(';'):
        for pattern in layer_height_patterns:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))

    # No matches found
    return None

# Checks to see if the input file is a binary G-code file
def is_binary_file(input_file, max_bytes=1024):
    try:
        with open(input_file,'rb') as file:
            return b'\x00' in file.read()
    except IOError:
        return False

# Checks if a utility such as bgcode is in the system path or current working directory
def get_utility_path(utility):
    # Check system path for the utility
    util_path = shutil.which(utility)
    if util_path:
        return util_path

    # Check current directory for the utility
    cwd_path = os.path.join(os.getcwd(), utility)
    logging.info(f"Current Working Directory path: {cwd_path}")
    if os.path.exists(cwd_path):
        return cwd_path

    # utility was not found on the system
    return False

# Convert a file from binary G-code format to regular G-code format and vice-versa
def convert_gcode(input_file, bgcode_path):
    try:
        # Convert the input file
        result = subprocess.run(
                                [bgcode_path, input_file], 
                                capture_output=True, 
                                text=True
        )

        # Conversion succeeded!
        if result.returncode == 0:
            logging.info('Binary G-code conversion was successful!')
            return True

        # Conversion failed
        raise Exception(f"Failed to perform binary G-code conversion. Error output: {result.stdout} {result.stderr}")
    except (subprocess.CalledProcessError, Exception) as e:
        error_message = f"Conversion failed: {e}"
        print(error_message, file=sys.stderr)
        logging.error(error_message)
        sys.exit(1)

def process_gcode(input_file, args_layer_height, extrusion_multiplier):
    generate_binary_gcode = False
    current_layer = 0
    current_z = 0.0
    perimeter_type = None
    perimeter_block_count = 0
    inside_perimeter_block = False
    logging.info("Starting G-code processing")
    logging.info(f"Input file: {input_file}")

    # If it's a binary G-code file, convert it back into a regular G-code file for modification
    if is_binary_file(input_file):
        logging.info('Binary G-code detected!')

        # Get the path to the bgcode binary
        bgcode_path = get_utility_path("bgcode")
        
        # bgcode not found. Cannot continue.
        # Tell the users and list out the paths where
        if not bgcode_path:
            error_message = "The bgcode utility is required in order to modify binary G-code files.\n\nPlease install in one of the paths below or disable binary G-code generation in your slicer.\n\n"
            print(error_message, file=sys.stderr)
            logging.error(error_message)

            # Tell the users where bgcode should be installed
            # for it to be found by this script
            paths = os.environ.get('PATH', '').split(os.pathsep)
            for path in paths:
                print(path, file=sys.stderr)
                logging.error(path)

            sys.exit(1)

        # File extension needs to be .bgcode for bgcode
        # to convert it correctly.
        if not input_file.endswith('.bgcode'):
            binary_input_file = f"{input_file}.bgcode"
            os.rename(input_file, binary_input_file)

            # Convert the binary G-code into regular G-code
            convert_gcode(binary_input_file, bgcode_path)

            # Rename the converted gcode file back to its original name
            os.rename(f"{input_file}.gcode", input_file)

        # Inform the script to perform G-code to binary G-code conversion at the end
        generate_binary_gcode = True

    # Read the input G-code.
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Try to pull layer_height from G-code comments
    # check each line for layer height information
    for line in lines:
        layer_height = get_layer_height_from_gcode(line)

        # Break the for loop once a valid layer_height has been found
        if layer_height is not None:
            logging.info(f"Found layer height in G-code comments")
            break

    # Use the value from command line args
    if layer_height is None:
        layer_height = args_layer_height

    z_shift = layer_height * 0.5
    logging.info(f"Z-shift: {z_shift} mm, Layer height: {layer_height} mm")

    # Identify the total number of layers by looking for `G1 Z` commands
    total_layers = sum(1 for line in lines if line.startswith("G1 Z"))

    # Process the G-code
    modified_lines = []
    for line in lines:
        # Detect layer changes
        if line.startswith("G1 Z"):
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                current_z = float(z_match.group(1))
                current_layer = int(current_z / layer_height)

                perimeter_block_count = 0  # Reset block counter for new layer
                logging.info(f"Layer {current_layer} detected at Z={current_z:.3f}")
            modified_lines.append(line)
            continue

        # Detect perimeter types from PrusaSlicer comments
        if ";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line:
            perimeter_type = "external"
            inside_perimeter_block = False
            logging.info(f"External perimeter detected at layer {current_layer}")
        elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
            perimeter_type = "internal"
            inside_perimeter_block = False
            logging.info(f"Internal perimeter block started at layer {current_layer}")
        elif ";TYPE:" in line:  # Reset for other types
            perimeter_type = None
            inside_perimeter_block = False

        # Group lines into perimeter blocks
        if perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line and "E" in line:
            # Start a new perimeter block if not already inside one
            if not inside_perimeter_block:
                perimeter_block_count += 1
                inside_perimeter_block = True
                logging.info(f"Perimeter block #{perimeter_block_count} detected at layer {current_layer}")

                # Insert the corresponding Z height for this block
                is_shifted = False  # Flag for whether this block is Z-shifted
                if perimeter_block_count % 2 == 1:  # Apply Z-shift to odd-numbered blocks
                    adjusted_z = current_z + z_shift
                    logging.info(f"Inserting G1 Z{adjusted_z:.3f} for shifted perimeter block #{perimeter_block_count}")
                    modified_lines.append(f"G1 Z{adjusted_z:.3f} ; Shifted Z for block #{perimeter_block_count}\n")
                    is_shifted = True
                else:  # Reset to the true layer height for even-numbered blocks
                    logging.info(f"Inserting G1 Z{current_z:.3f} for non-shifted perimeter block #{perimeter_block_count}")
                    modified_lines.append(f"G1 Z{current_z:.3f} ; Reset Z for block #{perimeter_block_count}\n")

            # Adjust extrusion (`E` values) for shifted blocks on the first and last layer
            if is_shifted:
                e_match = re.search(r'E([-\d.]+)', line)
                if e_match:
                    e_value = float(e_match.group(1))
                    if current_layer == 0:  # First layer
                        new_e_value = e_value * 1.5
                        logging.info(f"Multiplying E value by 1.5 on first layer (shifted block): {e_value:.5f} -> {new_e_value:.5f}")
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for first layer, block #{perimeter_block_count}\n"
                    elif current_layer == total_layers - 1:  # Last layer
                        new_e_value = e_value * 0.5
                        logging.info(f"Multiplying E value by 0.5 on last layer (shifted block): {e_value:.5f} -> {new_e_value:.5f}")
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for last layer, block #{perimeter_block_count}\n"
                    else: 
                        new_e_value = e_value * extrusion_multiplier
                        logging.info(f"Multiplying E value by extrusionMultiplier")
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for extrusionMultiplier, block #{perimeter_block_count}\n"
						
        elif perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line and "F" in line:  # End of perimeter block
            inside_perimeter_block = False

        modified_lines.append(line)

    # Overwrite the input file with the modified G-code
    with open(input_file, 'w') as outfile:
        outfile.writelines(modified_lines)

    # Convert the G-code back to binary G-code format
    if generate_binary_gcode:
        logging.info("Converting G-code back to binary G-code")
        if not input_file.endswith('.gcode'):
            tempfile = f"{input_file}.gcode"
            os.rename(input_file, tempfile)
            convert_gcode(tempfile, bgcode_path)
            os.remove(tempfile)
            os.rename(f"{input_file}.bgcode", input_file)

    logging.info("G-code processing completed")
    logging.info(f"Log file saved at {log_file_path}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process G-code for Z-shifting and extrusion adjustments.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    parser.add_argument("-layerHeight", type=float, default=0.2, help="Layer height in mm (default: 0.2mm)")
    parser.add_argument("-extrusionMultiplier", type=float, default=1, help="Extrusion multiplier for first layer (default: 1.5x)")
    args = parser.parse_args()

    process_gcode(
        input_file=args.input_file,
        args_layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier,
    )
