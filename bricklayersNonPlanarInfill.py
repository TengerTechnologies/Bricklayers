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
import math

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

# Add these constants from nonPlanarInfill.py
DEFAULT_AMPLITUDE = 0.6  # Default Z variation in mm
DEFAULT_FREQUENCY = 1.1  # Default frequency of the sine wave
SEGMENT_LENGTH = 1.0  # Split infill lines into segments of this length (mm)

# Add these helper functions from nonPlanarInfill.py
def segment_line(x1, y1, x2, y2, segment_length):
    """Divide a line into smaller segments."""
    segments = []
    total_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    num_segments = max(1, int(total_length // segment_length))
    
    for i in range(num_segments + 1):
        t = i / num_segments
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        segments.append((x, y))
    
    logging.debug(f"Segmented line ({x1}, {y1}) -> ({x2}, {y2}) into {len(segments)} segments.")
    return segments

def reset_modulation_state():
    """Reset parameters for Z-modulation to avoid propagating patterns."""
    global last_sx
    last_sx = 0

def update_layer_bounds(current_z, solid_infill_heights):
    """Update the bounds for non-planar processing based on current Z height."""
    global last_bottom_layer, next_top_layer
    lower_layers = [z for z in solid_infill_heights if z < current_z]
    upper_layers = [z for z in solid_infill_heights if z > current_z]
    if lower_layers:
        last_bottom_layer = max(lower_layers)
    if upper_layers:
        next_top_layer = min(upper_layers)

def process_nonplanar_infill(lines, current_z, amplitude, frequency, solid_infill_heights):
    """Process only the non-planar infill modifications."""
    modified_lines = []
    in_infill = False
    last_bottom_layer = 0
    next_top_layer = float('inf')
    processed_indices = set()

    def update_layer_bounds(current_z):
        nonlocal last_bottom_layer, next_top_layer
        lower_layers = [z for z in solid_infill_heights if z < current_z]
        upper_layers = [z for z in solid_infill_heights if z > current_z]
        if lower_layers:
            last_bottom_layer = max(lower_layers)
        if upper_layers:
            next_top_layer = min(upper_layers)

    for line_num, line in enumerate(lines):
        if line.startswith('G1') and 'Z' in line:
            z_match = re.search(r'Z([-+]?\d*\.?\d+)', line)
            if z_match:
                current_z = float(z_match.group(1))
                update_layer_bounds(current_z)

        if ';TYPE:Internal infill' in line:
            in_infill = True
            modified_lines.append(line)
            continue
        elif line.startswith(';TYPE:'):
            in_infill = False

        if in_infill and line_num not in processed_indices and line.startswith('G1') and 'E' in line:
            processed_indices.add(line_num)
            match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)\s*E([-+]?\d*\.?\d+)', line)
            if match:
                x1, y1, e = map(float, match.groups())
                next_line_index = line_num + 1
                
                if next_line_index < len(lines):
                    next_line = lines[next_line_index]
                    next_match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)', next_line)
                    if next_match:
                        x2, y2 = map(float, next_match.groups())
                        segments = segment_line(x1, y1, x2, y2, SEGMENT_LENGTH)
                        
                        distance_to_top = next_top_layer - current_z
                        distance_to_bottom = current_z - last_bottom_layer
                        total_distance = next_top_layer - last_bottom_layer
                        if total_distance > 0:
                            scaling_factor = min(distance_to_top, distance_to_bottom) / total_distance
                        else:
                            scaling_factor = 1.0

                        extrusion_per_segment = e / len(segments)
                        
                        for i, (sx, sy) in enumerate(segments):
                            z_mod = current_z + amplitude * scaling_factor * math.sin(frequency * sx)
                            
                            # Simple correction factor based on segment height difference
                            dz = abs(z_mod - current_z)
                            segment_2d = SEGMENT_LENGTH
                            segment_3d = math.sqrt(segment_2d**2 + dz**2)
                            correction_factor = segment_3d / segment_2d
                            
                            modified_lines.append(
                                f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} "
                                f"E{(extrusion_per_segment * correction_factor):.5f} ; Correction factor: {correction_factor:.3f} Original E: {extrusion_per_segment:.5f}\n"
                            )
                        continue
        
        modified_lines.append(line)
    
    return modified_lines

def process_wall_shifting(lines, layer_height, extrusion_multiplier, enable_wall_reorder=True):
    """Process only the wall shifting modifications."""
    current_layer = 0
    current_z = 0.0
    perimeter_type = None
    perimeter_block_count = 0
    inside_perimeter_block = False
    previous_g1_movement = None
    previous_f_speed = None
    z_shift = layer_height * 0.5
    
    # Add buffers for shifted and non-shifted walls (only used if wall_reorder is enabled)
    shifted_wall_buffer = []
    nonshifted_wall_buffer = []
    current_wall_buffer = []
    
    total_layers = sum(1 for line in lines if line.startswith(";AFTER_LAYER_CHANGE"))
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
            if enable_wall_reorder:
                # Output any buffered walls when switching to external perimeter
                if shifted_wall_buffer or nonshifted_wall_buffer:
                    # Output non-shifted walls first
                    for wall in nonshifted_wall_buffer:
                        modified_lines.extend(wall)
                    # Then output shifted walls
                    for wall in shifted_wall_buffer:
                        modified_lines.extend(wall)
                    # Clear buffers
                    shifted_wall_buffer = []
                    nonshifted_wall_buffer = []
            
            perimeter_type = "external"
            inside_perimeter_block = False
            logging.info(f"External perimeter detected at layer {current_layer}")
            modified_lines.append(line)
        elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
            perimeter_type = "internal"
            inside_perimeter_block = False
            if enable_wall_reorder:
                current_wall_buffer = []  # Start a new wall buffer
            logging.info(f"Internal perimeter block started at layer {current_layer}")
            modified_lines.append(line)
        elif ";TYPE:" in line:  # Reset for other types
            if enable_wall_reorder:
                # Output any remaining buffered walls
                if shifted_wall_buffer or nonshifted_wall_buffer:
                    for wall in nonshifted_wall_buffer:
                        modified_lines.extend(wall)
                    for wall in shifted_wall_buffer:
                        modified_lines.extend(wall)
                    shifted_wall_buffer = []
                    nonshifted_wall_buffer = []
            
            perimeter_type = None
            inside_perimeter_block = False
            modified_lines.append(line)

        # Group lines into perimeter blocks
        elif perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line and "E" in line:
            # Start a new perimeter block if not already inside one
            if not inside_perimeter_block:
                perimeter_block_count += 1
                inside_perimeter_block = True
                if enable_wall_reorder:
                    current_wall_buffer = []  # Start a new wall buffer
                
                # Add the cached movement command first
                if previous_g1_movement:
                    if enable_wall_reorder:
                        current_wall_buffer.append(f"{previous_g1_movement};Previous position\n")
                        current_wall_buffer.append(f"G1 F{previous_f_speed:.3f} ; F speed from previous G1 movement\n")
                    
                
                # Set Z height and determine if wall is shifted
                is_shifted = perimeter_block_count % 2 == 1
                if is_shifted:
                    adjusted_z = current_z + z_shift
                    z_command = f"G1 Z{adjusted_z:.3f} ; Shifted Z for block #{perimeter_block_count}\n"
                else:
                    z_command = f"G1 Z{current_z:.3f} ; Reset Z for block #{perimeter_block_count}\n"

                if enable_wall_reorder:
                    current_wall_buffer.append(z_command)
                else:
                    modified_lines.append(z_command)

            # Process the current line (including extrusion adjustments)
            if is_shifted:
                e_match = re.search(r'E([-\d.]+)', line)
                if e_match:
                    e_value = float(e_match.group(1))
                    original_line = line
                    if current_layer == 1:  # First layer
                        new_e_value = e_value * 1.5  # 50% more extrusion
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for first layer (1.5x), block #{perimeter_block_count}\n"
                    elif current_layer == total_layers - 1:  # Last layer
                        new_e_value = e_value * 0.5  # 50% less extrusion
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for last layer (0.5x), block #{perimeter_block_count}\n"
                    else:  # Regular layers
                        line += f" ; current layer: {current_layer} total layers: {total_layers} \n"
                        new_e_value = e_value * extrusion_multiplier
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for regular layer ({extrusion_multiplier}x), block #{perimeter_block_count}\n"
              

            if enable_wall_reorder:
                current_wall_buffer.append(line)
            else:
                modified_lines.append(line)



        elif perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line and "F" in line:
            # End of perimeter block
            if inside_perimeter_block:
                if enable_wall_reorder:
                    current_wall_buffer.append(line)
                    # Add Z reset for shifted blocks
                    if is_shifted:
                        current_wall_buffer.append(f"G1 Z{current_z:.3f} ; Reset Z after shifted block #{perimeter_block_count}\n")
                    # Add completed wall to appropriate buffer
                    if is_shifted:
                        shifted_wall_buffer.append(current_wall_buffer)
                    else:
                        nonshifted_wall_buffer.append(current_wall_buffer)
                else:
                    modified_lines.append(line)
                    if is_shifted:
                        modified_lines.append(f"G1 Z{current_z:.3f} ; Reset Z after shifted block #{perimeter_block_count}\n")
                inside_perimeter_block = False
                
        elif perimeter_type == "internal" and line.startswith("G1") and "F" in line:  #fix for Fspeed movements inside perimeter blocks
            if enable_wall_reorder:
                current_wall_buffer.append(line)
            else:
                modified_lines.append(line)
        # Cache G1 movements with X and Y coordinates and F speeds
        if line.startswith("G1"):
            if "X" in line and "Y" in line:
                previous_g1_movement = line.strip()
                logging.info(f"Cached G1 movement: {previous_g1_movement}")
            if "F" in line:
                f_match = re.search(r'F([\d.]+)', line)
                if f_match:
                    previous_f_speed = float(f_match.group(1))
                    logging.info(f"Cached F speed: {previous_f_speed}")

        # Add non-wall lines directly to output
        if not inside_perimeter_block and not perimeter_type == "internal":
            modified_lines.append(line)

    return modified_lines

def get_layer_height(gcode_lines):
    """Extract layer height from G-code header comments"""
    for line in gcode_lines:
        if "; layer_height =" in line.lower():
            match = re.search(r'layer_height = (\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None

def process_gcode(input_file, extrusion_multiplier, enable_nonplanar=False, enable_wall_reorder=True, amplitude=DEFAULT_AMPLITUDE, frequency=DEFAULT_FREQUENCY):
    logging.info("Starting G-code processing")
    logging.info(f"Input file: {input_file}")

    # Read the input G-code
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Get layer height from G-code
    layer_height = get_layer_height(lines)
    if layer_height is None:
        layer_height = 0.2  # Default fallback value
        logging.warning(f"Could not detect layer height from G-code, using default value: {layer_height}mm")
    else:
        logging.info(f"Detected layer height from G-code: {layer_height}mm")

    # First pass: Process non-planar infill if enabled
    if enable_nonplanar:
        logging.info("Processing non-planar infill modifications...")
        solid_infill_heights = []
        current_z = 0.0
        
        # Collect solid infill heights
        for line in lines:
            if line.startswith('G1') and 'Z' in line:
                z_match = re.search(r'Z([-+]?\d*\.?\d+)', line)
                if z_match:
                    current_z = float(z_match.group(1))
            if ';TYPE:Solid infill' in line:
                solid_infill_heights.append(current_z)
                logging.info(f"Found solid infill at Z={current_z}")

        # Process non-planar infill
        lines = process_nonplanar_infill(lines, current_z, amplitude, frequency, solid_infill_heights)
        logging.info("Non-planar infill processing completed")

    # Second pass: Process wall shifting
    logging.info("Processing wall shifting modifications...")
    modified_lines = process_wall_shifting(lines, layer_height, extrusion_multiplier, enable_wall_reorder)
    logging.info("Wall shifting processing completed")

    # Write the final modified G-code
    with open(input_file, 'w') as outfile:
        outfile.writelines(modified_lines)

    logging.info("G-code processing completed")
    logging.info(f"Log file saved at {log_file_path}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process G-code for Z-shifting, extrusion adjustments, and non-planar infill.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    parser.add_argument("-extrusionMultiplier", type=float, default=1, help="Extrusion multiplier (default: 1.0)")
    parser.add_argument("-nonPlanar", type=int, choices=[0, 1], default=0, help="Enable non-planar infill (0=off, 1=on)")
    parser.add_argument("-wallReorder", type=int, choices=[0, 1], default=1, help="Enable wall reordering (0=off, 1=on)")
    parser.add_argument("-amplitude", type=float, default=DEFAULT_AMPLITUDE, help=f"Amplitude of the Z modulation (default: {DEFAULT_AMPLITUDE})")
    parser.add_argument("-frequency", type=float, default=DEFAULT_FREQUENCY, help=f"Frequency of the Z modulation (default: {DEFAULT_FREQUENCY})")
    args = parser.parse_args()

    process_gcode(
        input_file=args.input_file,
        extrusion_multiplier=args.extrusionMultiplier,
        enable_nonplanar=bool(args.nonPlanar),
        enable_wall_reorder=bool(args.wallReorder),
        amplitude=args.amplitude,
        frequency=args.frequency,
    )
