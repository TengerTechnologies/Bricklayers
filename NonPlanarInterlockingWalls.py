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
import math
import sys
import logging
import argparse
from collections import Counter

def sine_wave(x):
    return math.sin(x)

def triangle_wave(x):
    # Create a sharp triangle wave
     # Normalized position t in [0,1) inside each 2π period:
    t = (x / (2 * math.pi)) % 1.0

    if t < 0.5:
        # first half of the 2π: ramp from −1 to +1
        #   at t=0    → −1
        #   at t=0.5  → +1
        return -1.0 + (4.0 * t)
    else:
        # second half of the 2π: ramp from +1 back down to −1
        #   at t=0.5  → +1
        #   at t=1.0  → −1
        return 3.0 - (4.0 * t)

def trapezoidal_wave(x):
 
    # t in [0,1) is the fractional position within each 2π:
    t = (x / (2 * math.pi)) % 1.0

    if t < 0.25:
        # Ramp from −1 up to +1 over the first quarter‐period
        #   at t=0    ⇒ −1
        #   at t=0.25 ⇒ +1
        return -1.0 + (t / 0.25) * 2.0

    elif t < 0.50:
        # Hold at +1 for the next quarter‐period
        return +1.0

    elif t < 0.75:
        # Ramp from +1 down to −1 over the third quarter‐period
        #   at t=0.50 ⇒ +1
        #   at t=0.75 ⇒ −1
        return +1.0 - ((t - 0.50) / 0.25) * 2.0

    else:
        # Hold at −1 for the final quarter‐period
        return -1.0

def sawtooth_wave(x):
   
    return 1.0 - ( (x % (2 * math.pi)) / math.pi )

# Dictionary mapping function names to their implementations
PERIODIC_FUNCTIONS = {
    "sine": sine_wave,
    "triangle": triangle_wave,
    "trapezoidal": trapezoidal_wave,
    "sawtooth": sawtooth_wave
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gcode_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

DEFAULT_AMPLITUDE = 0.3
DEFAULT_FREQUENCY = 1.1
DEFAULT_MAX_STEP = 0.1  # Default 10% step size per layer
DEFAULT_RESOLUTION = 0.2  # Default segment length in mm

# Lookup tables for different slicers
SLICER_TYPES = {
    "prusaslicer": {
        "infill": [";TYPE:Internal infill"],
        "solid_infill": [";TYPE:Solid infill", ";TYPE:Top solid infill", ";TYPE:Bridge infill"],
        "perimeter": [";TYPE:Perimeter"],
        "external_perimeter": [";TYPE:External perimeter"],
        "type_prefix": ";TYPE:"
    },
    "orcaslicer": {
        "infill": [";TYPE:Internal infill", ";TYPE:internal infill"],
        "solid_infill": [";TYPE:Solid infill", ";TYPE:solid infill", ";TYPE:Top surface", ";TYPE:top surface"],
        "perimeter": [";TYPE:Inner wall", ";TYPE:inner wall"],
        "external_perimeter": [";TYPE:Outer wall", ";TYPE:outer wall"],
        "type_prefix": ";TYPE:"
    },
    "bambustudio": {
        "infill": ["; FEATURE: Sparse infill", "; FEATURE: Internal infill"],
        "solid_infill": ["; FEATURE: Solid infill", "; FEATURE: Top surface", "; FEATURE: Bridge infill"],
        "perimeter": ["; FEATURE: Inner wall"],
        "external_perimeter": ["; FEATURE: Outer wall"],
        "type_prefix": "; FEATURE:"
    }
}


def segment_line(x1, y1, x2, y2, segment_length):
    segments = []
    total_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    num_segments = max(1, int(total_length // segment_length))
    for i in range(num_segments + 1):
        t = i / num_segments
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        segments.append((x, y))
    return segments


def reset_modulation_state():
    global last_sx
    last_sx = 0


def detect_slicer(gcode_lines):
    for line in gcode_lines[:10]:
        if 'PrusaSlicer' in line:
            return 'prusaslicer'
        elif 'OrcaSlicer' in line:
            return 'orcaslicer'
        elif 'BambuStudio' in line:
            return 'bambustudio'
    return None


def detect_gcode_flavor(gcode_lines):
    for line in gcode_lines:
        if line.startswith('; gcode_flavor ='):
            return line.split('=')[-1].strip()
    return None


def process_gcode(
    input_file,
    wall_amplitude, wall_frequency, wall_direction,
    infill_amplitude, infill_frequency, infill_direction,
    include_infill, include_perimeters, include_external_perimeters,
    max_step_size, alternate_loops,
    infill_function="sine", perimeter_function="sine",
    resolution=DEFAULT_RESOLUTION
):
    modified_lines = []
    current_z = 0
    current_region = None
    last_bottom_layer = 0
    next_top_layer = float('inf')
    processed_indices = set()

    # Variables for tracking nozzle state and detecting wall-loop starts.
    last_nozzle_position = None
    in_new_wall_region = False  # Set True when starting a new wall region
    loop_count   = 0
    phase_offset = 0.0

    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Detect slicer type and G-code flavor.
    slicer = detect_slicer(lines)
    gcode_flavor = detect_gcode_flavor(lines)
    if slicer and slicer.lower() in SLICER_TYPES:
        lookup = SLICER_TYPES[slicer.lower()]
        if slicer == 'orcaslicer' and gcode_flavor == 'marlin':
            lookup = SLICER_TYPES['bambustudio']
        logging.debug(f"Using lookup table for: {slicer.lower()}")
    else:
        lookup = SLICER_TYPES["prusaslicer"]
        logging.debug("Using default PrusaSlicer lookup table")

    # Extract markers.
    INFILL_MARKERS = lookup["infill"]
    SOLID_INFILL_MARKERS = lookup["solid_infill"]
    PERIMETER_MARKERS = lookup["perimeter"]
    EXTERNAL_PERIMETER_MARKERS = lookup["external_perimeter"]
    TYPE_PREFIX = lookup["type_prefix"]

    # Gather Z values for solid infill.
    solid_infill_heights = []
    for line in lines:
        if line.startswith('G1') and 'Z' in line:
            z_match = re.search(r'Z([-+]?[\d]*\.?[\d]+)', line)
            if z_match:
                current_z = float(z_match.group(1))
        if any(marker in line for marker in SOLID_INFILL_MARKERS):
            solid_infill_heights.append(current_z)

    def is_current_layer_solid_infill(z):
        return z in solid_infill_heights

    def update_layer_bounds(current_z):
        nonlocal last_bottom_layer, next_top_layer
        lower_layers = [z for z in solid_infill_heights if z < current_z]
        upper_layers = [z for z in solid_infill_heights if z > current_z]
        if lower_layers:
            last_bottom_layer = max(lower_layers)
        if upper_layers:
            next_top_layer = min(upper_layers)

    # Determine layer height.
    layer_heights = []
    last_z = None
    for line in lines:
        if line.startswith('G1') and 'Z' in line:
            z_match = re.search(r'Z([-+]?[\d]*\.?[\d]+)', line)
            if z_match:
                z = float(z_match.group(1))
                if last_z is not None:
                    layer_heights.append(z - last_z)
                last_z = z
    layer_height = 0.2
    if layer_heights:
        height_counter = Counter(round(h, 3) for h in layer_heights if h > 0.01)
        if height_counter:
            layer_height = height_counter.most_common(1)[0][0]

    def calculate_scaling_factor(current_z, last_bottom_layer, next_top_layer, max_step_size):
        distance_to_top = next_top_layer - current_z
        distance_to_bottom = current_z - last_bottom_layer
        total_distance = next_top_layer - last_bottom_layer
        raw_scaling_factor = min(distance_to_top, distance_to_bottom) / total_distance if total_distance > 0 else 1.0
        max_possible_scale = max_step_size * min(distance_to_bottom / layer_height, distance_to_top / layer_height)
        limited_scaling_factor = min(raw_scaling_factor, max_possible_scale)
        return limited_scaling_factor

    # Main processing loop.
    for line_num, line in enumerate(lines):
        if line_num in processed_indices:
            continue

        # --- New Travel Move Handling ---
        # If a G1 line contains X/Y but no E, we assume it is a travel move
        # (for example, a move to the start of a new wall loop). In that case, we simply update
        # the stored nozzle position and clear the "new wall" flag, outputting the line as-is.
        if line.startswith("G1") and ("X" in line or "Y" in line) and "E" not in line:
            pos_match = re.search(r'X([-+]?[\d]*\.?[\d]+).*?Y([-+]?[\d]*\.?[\d]+)', line)
            if pos_match:
                last_nozzle_position = (float(pos_match.group(1)), float(pos_match.group(2)))
            #in_new_wall_region = False  # Clear the flag: we've started a new loop.
            if current_region in ('internal_wall','external_wall'):
                in_new_wall_region = True
                if alternate_loops:
                    loop_count += 1
                    phase_offset = (loop_count % 2) * (math.pi / 2)
            modified_lines.append(line)
            processed_indices.add(line_num)
            
            continue

        if line.startswith('M73'):
            modified_lines.append(line)
            continue

        # Update Z and layer bounds on Z moves.
        if line.startswith('G1') and 'Z' in line:
            z_match = re.search(r'Z([-+]?[\d]*\.?[\d]+)', line)
            if z_match:
                current_z = float(z_match.group(1))
                reset_modulation_state()
                update_layer_bounds(current_z)

        # Set region based on markers.
        if any(marker in line for marker in INFILL_MARKERS) and include_infill:
            current_region = 'infill'
        elif any(marker in line for marker in PERIMETER_MARKERS) and include_perimeters:
            current_region = 'internal_wall'
            in_new_wall_region = True
            if alternate_loops:
                loop_count = 0
                phase_offset = 0.0 
        elif any(marker in line for marker in EXTERNAL_PERIMETER_MARKERS) and include_external_perimeters:
            current_region = 'external_wall'
            in_new_wall_region = True
            
        elif TYPE_PREFIX in line:
            current_region = None

        # Process modulated moves that have an extrusion value.
        if current_region in ['infill', 'internal_wall', 'external_wall'] and line.startswith('G1') and 'E' in line:
            # For walls, if we're at the very start of a new wall region,
            # check if bridging is needed.
            if current_region in ['internal_wall', 'external_wall'] and in_new_wall_region:
                in_new_wall_region = False
                match = re.search(r'X([-+]?[\d]*\.?[\d]+)\s*Y([-+]?[\d]*\.?[\d]+)\s*E([-+]?[\d]*\.?[\d]+)', line)
                if match:
                    wall_x = float(match.group(1))
                    wall_y = float(match.group(2))
                    e_val = float(match.group(3))
                    # Only bridge if a stored nozzle position exists and it is different
                    if last_nozzle_position is not None and (last_nozzle_position != (wall_x, wall_y)):
                        x1, y1 = last_nozzle_position
                        x2, y2 = wall_x, wall_y
                        segments = segment_line(x1, y1, x2, y2, resolution)
                        prev_pt = None
                        for i, (sx, sy) in enumerate(segments):
                            extrusion_per_segment = e_val / len(segments)
                            scaling_factor = calculate_scaling_factor(current_z, last_bottom_layer, next_top_layer, max_step_size)
                            # Use wall modulation parameters.
                            if wall_direction == "x":
                                sine_input = sx
                            elif wall_direction == "y":
                                sine_input = sy
                            elif wall_direction == "xy":
                                sine_input = sx + sy
                            elif wall_direction == "negx":
                                sine_input = -sx
                            elif wall_direction == "negy":
                                sine_input = -sy
                            elif wall_direction == "negxy":
                                sine_input = -(sx + sy)
                            else:
                                sine_input = sx
                            # compute raw angle
                            angle = wall_frequency * sine_input
                            # if we asked for alternation and this is a wall, tack on the per‐loop phase shift
                            if alternate_loops and current_region in ('internal_wall', 'external_wall'):
                                angle += phase_offset
                            
                            # finally modulate Z
                            wave_func = PERIODIC_FUNCTIONS[perimeter_function]
                            z_mod = current_z + wall_amplitude * scaling_factor * wave_func(angle)
                            if prev_pt is not None:
                                px, py, pz = prev_pt
                                dz         = z_mod - pz
                                # true 3D step length
                                seg3d      = math.hypot(resolution, dz)
                                # scale your original E
                                e_adj      = extrusion_per_segment * (seg3d / resolution)
                                # emit the move at the *previous* point
                                mod_line = f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{e_adj:.5f} ;Bridge\n"
                            else: mod_line = f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{extrusion_per_segment:.5f} ;Bridge no previous point\n"# stash current as "previous" for next iteration
                        
                            prev_pt = (sx, sy, z_mod)
                            
                            modified_lines.append(mod_line)
                        # Clear the "new wall" flag and update stored nozzle.
                        in_new_wall_region = False
                        last_nozzle_position = (wall_x, wall_y)
                        processed_indices.add(line_num)
                        continue
                    else:
                        # If no bridging is needed, clear the flag and update stored nozzle.
                        in_new_wall_region = False
                        last_nozzle_position = (wall_x, wall_y)
                else:
                    # If we cannot parse the coordinates, just pass the line on.
                    modified_lines.append(line +"; bridge didn't find a match\n")
                    processed_indices.add(line_num)
                    continue

            # For a standard move with extrusion, process normally.
            m = re.search(
                r'X([-+]?[\d]*\.?[\d]+)\s*Y([-+]?[\d]*\.?[\d]+)\s*E([-+]?[\d]*\.?[\d]+)',
                line
            )
            if not m:
                # no coords+E → passthrough
                modified_lines.append(line)
                processed_indices.add(line_num)
                continue
            x2, y2, e_total = map(float, m.groups())

            # 2) if we have no prior point, emit raw and set nozzle
            if last_nozzle_position is None:
                modified_lines.append(
                    line.rstrip() + " ;no prior point, raw emit\n"
                )
                processed_indices.add(line_num)
                last_nozzle_position = (x2, y2)
                continue

            # 3) segment from true start→end
            x1, y1 = last_nozzle_position
            segments = segment_line(x1, y1, x2, y2, resolution)
            prev_pt = None

            for i, (sx, sy) in enumerate(segments):
                if i == 0:
                    # seed prev_pt but don't emit
                    prev_pt = (sx, sy, current_z)
                    continue

                # compute per‑segment extrusion and modulation
                extrusion_per_seg = e_total / (len(segments) - 1)
                scaling_factor = calculate_scaling_factor(
                    current_z, last_bottom_layer, next_top_layer, max_step_size
                )
                # pick your amp/freq/direction based on region…
                if current_region == 'infill':
                    amp, freq, dirn = infill_amplitude, infill_frequency, infill_direction
                    wave_func = PERIODIC_FUNCTIONS[infill_function]
                else:
                    amp, freq, dirn = wall_amplitude, wall_frequency, wall_direction
                    wave_func = PERIODIC_FUNCTIONS[perimeter_function]

                if dirn == "x":
                    sine_input = sx
                elif dirn == "y":
                    sine_input = sy
                elif dirn == "xy":
                    sine_input = sx + sy
                elif dirn == "negx":
                    sine_input = -sx
                elif dirn == "negy":
                    sine_input = -sy
                elif dirn == "negxy":
                    sine_input = -(sx + sy)
                else:
                    sine_input = sx

                angle = freq * sine_input
                # if we asked for alternation and this is a wall, tack on the per‐loop phase shift
                if alternate_loops and current_region in ('internal_wall', 'external_wall'):
                    angle += phase_offset
                
                # finally modulate Z using the selected wave function
                z_mod = current_z + amp * scaling_factor * wave_func(angle)
                dz    = z_mod - prev_pt[2]
                seg3d = math.hypot(resolution, dz)
                e_adj = extrusion_per_seg * (seg3d / resolution)

                # 4) emit the slice, annotated so you can verify
                mod_line = (
                    f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} "
                    f"E{e_adj:.5f} "
                    f";seg {i}/{len(segments)-1} "
                    f"from ({x1:.3f},{y1:.3f})->({x2:.3f},{y2:.3f})\n"
                )
                modified_lines.append(mod_line)
                prev_pt = (sx, sy, z_mod)

            # 5) done—remember where we ended
            processed_indices.add(line_num)
            last_nozzle_position = (x2, y2)
            continue

        # For non-modulated moves with coordinates, update the stored nozzle position.
        if line.startswith('G1') and ('X' in line or 'Y' in line):
            pos_match = re.search(r'X([-+]?[\d]*\.?[\d]+).*?Y([-+]?[\d]*\.?[\d]+)', line)
            if pos_match:
                last_nozzle_position = (float(pos_match.group(1)), float(pos_match.group(2)))

        if line_num not in processed_indices:
            modified_lines.append(line+";not processed\n")

    return modified_lines


def save_gcode(output_file, lines):
    with open(output_file, 'w') as file:
        file.writelines(lines)
    logging.info(f"Saved modified G-code to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add non-planar modulation to G-code.")

    parser.add_argument("input_file", help="The input G-code file.")
    parser.add_argument("-include-infill", action="store_true", help="Apply modulation to infill.")
    parser.add_argument("-include-perimeters", action="store_true", help="Apply modulation to internal perimeters.")
    parser.add_argument("-include-external-perimeters", action="store_true", help="Include external perimeters in modulation.")
    parser.add_argument("-wall-amplitude", type=float, default=DEFAULT_AMPLITUDE,
                        help="Amplitude for wall modulation (default: 0.3).")
    parser.add_argument("-wall-frequency", type=float, default=DEFAULT_FREQUENCY,
                        help="Frequency for wall modulation (default: 1.1).")
    parser.add_argument("-infill-amplitude", type=float, default=DEFAULT_AMPLITUDE,
                        help="Amplitude for infill modulation (default: 0.3).")
    parser.add_argument("-infill-frequency", type=float, default=DEFAULT_FREQUENCY,
                        help="Frequency for infill modulation (default: 1.1).")
    parser.add_argument("-infill-direction", choices=["x", "y", "xy", "negx", "negy", "negxy"],
                        default="x", help="Direction of sine wave for infill (default: x)")
    parser.add_argument("-wall-direction", choices=["x", "y", "xy", "negx", "negy", "negxy"],
                        default="x", help="Direction of sine wave for walls (default: x)")
    parser.add_argument("-max-step-size", type=float, default=DEFAULT_MAX_STEP,
                        help="Max amplitude increase per layer as a percentage (0.0-1.0, default: 0.1)")
    parser.add_argument("-alternate-loops",action="store_true",
                        help="Alternate sine phase (low→low, high→high) on successive wall loops")
    parser.add_argument("-infill-function", choices=["sine", "triangle", "trapezoidal", "sawtooth"],
                        default="sine", help="Periodic function to use for infill modulation (default: sine)")
    parser.add_argument("-perimeter-function", choices=["sine", "triangle", "trapezoidal", "sawtooth"],
                        default="sine", help="Periodic function to use for perimeter modulation (default: sine)")
    parser.add_argument("-resolution", type=float, default=DEFAULT_RESOLUTION,
                        help="Resolution of wave segments in mm (default: 0.2)")

    args = parser.parse_args()

    modified_lines = process_gcode(
        args.input_file,
        args.wall_amplitude, args.wall_frequency, args.wall_direction,
        args.infill_amplitude, args.infill_frequency, args.infill_direction,
        args.include_infill, args.include_perimeters, args.include_external_perimeters,
        args.max_step_size, alternate_loops=args.alternate_loops,
        infill_function=args.infill_function,
        perimeter_function=args.perimeter_function,
        resolution=args.resolution
    )

    save_gcode(args.input_file, modified_lines)
