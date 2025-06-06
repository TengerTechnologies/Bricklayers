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
import numpy as np
import math
from collections import defaultdict

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging to save in the script's directory
log_file_path = os.path.join(script_dir, "zigzagWallCombiner.txt")
logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class GCodeMove:
    def __init__(self, line, x, y, e=None, is_travel=False):
        self.line = line
        self.x = x
        self.y = y
        self.e = e
        self.is_travel = is_travel
    
    def get_point(self):
        return Point(self.x, self.y)

def parse_gcode(line):
    """Parse a G-code line to extract the movement information"""
    try:
        x_match = re.search(r'X([-+]?\d*\.?\d+)', line)
        y_match = re.search(r'Y([-+]?\d*\.?\d+)', line)
        e_match = re.search(r'E([-+]?\d*\.?\d+)', line)
        
        if x_match and y_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            e = float(e_match.group(1)) if e_match else None
            is_travel = e_match is None
            
            return GCodeMove(line, x, y, e, is_travel)
    except Exception as e:
        logging.error(f"Error parsing G-code line: {line}")
        logging.error(f"Exception: {str(e)}")
    return None

def process_gcode(input_file, zigzag_length=2.0):
    """
    Process G-code to combine adjacent internal walls into zigzag patterns
    based on wall order (1st with 2nd, 3rd with 4th, etc.)
    
    Args:
        input_file: Path to the input G-code file
        zigzag_length: Length of each zigzag segment (mm)
    """
    logging.info("Starting G-code zigzag wall processing")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Zigzag segment length: {zigzag_length} mm")
    
    try:
        # Read the input G-code
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
        
        logging.info(f"Read {len(lines)} lines of G-code")
        
        # Parse G-code to extract walls by layer and preserve order
        layer_islands = defaultdict(list)
        external_walls = defaultdict(list)
        current_island=[]
        current_layer = 0  # Default to layer 0 if no layer marker found
        current_wall = []
        current_wall_type = None
        inside_perimeter_block = False
        perimeter_block_count = 0
        last_xy_move = None  # Track the last G1 move with X Y coordinates
        
        # First pass - analyze the G-code structure
        perimeter_markers = set()
        layer_markers = set()
        
        for line in lines[:5000]:  # Check first 5000 lines to identify markers
            if ';TYPE:' in line:
                perimeter_markers.add(line.strip())
            if ';LAYER:' in line or line.startswith(';LAYER'):
                layer_markers.add(line.strip())
            elif ';LAYER_CHANGE' in line:
                layer_markers.add(line.strip())
        
        logging.info(f"Detected perimeter markers: {perimeter_markers}")
        logging.info(f"Detected layer markers: {layer_markers}")

        # Second pass - extract walls
        for i, line in enumerate(lines):
            try:
                # Check for layer change - handle various formats
                if ';LAYER:' in line:
                    # Standard PrusaSlicer/SuperSlicer format
                    layer_match = re.search(r';LAYER:(\d+)', line)
                    if layer_match:
                        current_layer = int(layer_match.group(1))
                        logging.debug(f"Detected layer: {current_layer}")
                elif ';LAYER_CHANGE' in line:
                    # LAYER_CHANGE is often followed by the layer height
                    # Increment layer number when we see this
                    current_layer += 1
                    logging.debug(f"Layer change detected, now on layer: {current_layer}")
                
                # Detect perimeter types from slicer comments
                if ";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line:
                    # Save any current wall before starting a new section
                    if current_wall and (current_wall_type == "internal" or current_wall_type == "internal overhang") and inside_perimeter_block:
                        if len(current_wall) > 5:
                            current_island.append(current_wall)
                            logging.debug(f"Saved internal wall with {len(current_wall)} points before external perimeter")
                    
                    current_wall_type = "external"
                    inside_perimeter_block = False
                    current_wall = []
                    if len(current_island) > 0:
                        layer_islands.setdefault(current_layer, []).append(current_island)
                    current_island=[]
                    logging.info(f"Detected external perimeter")
                
                elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
                    # Save any current wall before starting a new section
                    if current_wall and current_wall_type == "internal" and inside_perimeter_block:
                        if len(current_wall) > 5:
                            current_island.append(current_wall)
                            logging.debug(f"{line}")
                            logging.debug(f"Saved internal wall with {len(current_wall)} points at perimeter type change")
                    
                    if current_wall_type=="internal overhang":
                        logging.info(f"Detected end of {current_wall_type} perimeter")
                        current_wall_type = "internal"
                        continue
                    else:
                        current_wall_type = "internal"
                        inside_perimeter_block = False
                        current_wall = []
                        logging.info(f"Detected internal perimeter")

                elif ";TYPE:Overhang perimeter" in line:
                    current_wall_type=current_wall_type+" overhang"
                    logging.info(f"Detected {current_wall_type} perimeter")

                elif ";TYPE:" in line:  # Reset for other types
                    # Save any current wall before starting a new section
                    if current_wall and current_wall_type == "internal" and inside_perimeter_block:
                        if len(current_wall) > 5:
                            current_island.append(current_wall)
                            logging.debug(f"{line}")
                            logging.debug(f"Saved internal wall with {len(current_wall)} points at type change")
                    
                    current_wall_type = None
                    inside_perimeter_block = False
                    current_wall = []
                
                # Group lines into perimeter blocks - only if we're in an internal perimeter section
                if (current_wall_type == "internal" or current_wall_type == "internal overhang") and line.startswith("G1") and "X" in line and "Y" in line and "E" in line:
                    # Start a new perimeter block if not already inside one
                    if not inside_perimeter_block:
                        perimeter_block_count += 1
                        inside_perimeter_block = True
                        current_wall = []  # Reset the wall at the start of each perimeter block
                        
                        # Add the last XY move to the beginning of this wall if one exists and it was a travel move
                        if last_xy_move and last_xy_move.is_travel:
                            current_wall.append(last_xy_move)
                            logging.info(f"Added last XY move to wall: {last_xy_move.line.strip()}")
                            logging.info(f"Last XY move coordinates: X={last_xy_move.x}, Y={last_xy_move.y}, is_travel={last_xy_move.is_travel}")
                        else:
                            logging.info(f"No suitable last XY move found or it wasn't a travel move")
                            if last_xy_move:
                                logging.info(f"Last XY move (not added): {last_xy_move.line.strip()}, is_travel={last_xy_move.is_travel}")
                        
                        logging.info(f"Starting internal perimeter block #{perimeter_block_count}")
                    
                    # Parse and add this point to the current wall
                    gcode_point = parse_gcode(line)
                    if gcode_point:
                        gcode_point.line = line  # Save the original line
                        current_wall.append(gcode_point)
                        logging.debug(f"Added extrusion point to wall: {line.strip()}")
                
                # Detect end of a perimeter block: M commands, travel moves, or comments
                elif inside_perimeter_block and (
                    line.startswith('M') or  # Any M command
                    (line.startswith('G1 ') and "X" in line and "Y" in line and ' E' not in line) or  # Travel move without extrusion
                    line.startswith(';TYPE')  # Comment line
                ):
                    # Special case - ignore progress reports that don't end blocks
                    if line.startswith('M73 ') or line.startswith('M204 ')or line.startswith('M106 '):  # M73 is a progress report - ignore it
                        pass
                    else:
                        # End of perimeter block - save the current wall if it's internal
                        if current_wall and current_wall_type == "internal":
                            if len(current_wall) > 0:  # Accept walls of any size
                                current_island.append(current_wall.copy())
                                logging.info(line)
                                # Log the complete wall block
                                logging.info(f"Saved internal wall (layer {current_layer}) with {len(current_wall)} points")
                                logging.info(f"Complete wall block G-code:")
                                for wall_point in current_wall:
                                    logging.info(f"  {wall_point.line.strip()}")
                                logging.info(f"End of perimeter block #{perimeter_block_count}")
                                if line.startswith("G1 "):
                                    gcode_point = parse_gcode(line)
                                    travel = gcode_point.get_point().distance(current_wall[-1])
                                    if travel>1:
                                        layer_islands.setdefault(current_layer, []).append(current_island)
                                        current_island=[]

                            # Reset for the next perimeter block
                            current_wall = []
                            inside_perimeter_block = False
                            
                            # Reset perimeter type if we're changing types
                            if line.startswith(';TYPE:'):
                                if ";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line:
                                    current_wall_type = "external"
                                    layer_islands.setdefault(current_layer, []).append(current_island)
                                    current_island=[]
                                elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
                                    current_wall_type = "internal"
                                elif ";TYPE:Overhang perimeter" in line:
                                    current_wall_type=current_wall_type
                                else:
                                    current_wall_type = None


                # IMPORTANT: Track any G1 move with X and Y coordinates at the END of processing each line
                # This ensures we have the correct last_xy_move for the NEXT line
                if line.startswith("G1") and "X" in line and "Y" in line:
                    # Store this as the last XY move regardless of type
                    xy_match = re.search(r'G1 X([-+]?\d*\.?\d+) Y([-+]?\d*\.?\d+)', line)
                    if xy_match:
                        x = float(xy_match.group(1))
                        y = float(xy_match.group(2))
                        is_travel = "E" not in line
                        e_val = None
                        if not is_travel:
                            e_match = re.search(r'E([-+]?\d*\.?\d+)', line)
                            if e_match:
                                e_val = float(e_match.group(1))
                        
                        last_xy_move = GCodeMove(line, x, y, e_val, is_travel)
                        logging.info(f"Updated last XY move: {line.strip()}, X={x}, Y={y}, is_travel={is_travel}")
            
            except Exception as e:
                logging.error(f"Error processing line {i}: {line.strip()}")
                logging.error(f"Exception: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Save any remaining wall
        if current_wall and current_wall_type == "internal" and inside_perimeter_block:
            if len(current_wall) > 0:  # Accept walls of any size
                current_island.append(current_wall.copy())
                logging.info(f"Saved final internal perimeter wall with {len(current_wall)} points")
        if len(current_island) > 0:
            layer_islands.setdefault(current_layer, []).append(current_island.copy())

        # Log wall statistics
        for layer, islands in layer_islands.items():
            for i in range(len(islands)):
                logging.info(f"Layer {layer}, Island {i+1}: {len(islands[i])} internal walls")
        
        # Process each layer to pair walls and create zigzags
        modified_lines = lines.copy()
        zigzag_segments = defaultdict(list)
        wall_line_indices = defaultdict(list)  # Store the line indices of walls to replace
        wall_start_end = defaultdict(list)     # Store the start/end line indices for each wall
        
        # First pass - locate wall locations in the original G-code
        current_layer = 0
        current_wall_start = None
        inside_internal_perimeter = False
        last_travel_index = None  # Track the line index of the last travel move
        
        for i, line in enumerate(lines):
            # Track layer changes
            if ';LAYER:' in line:
                layer_match = re.search(r';LAYER:(\d+)', line)
                if layer_match:
                    current_layer = int(layer_match.group(1))
            elif ';LAYER_CHANGE' in line:
                current_layer += 1
            
            # Track travel moves that could be part of walls
            if line.startswith("G1") and "X" in line and "Y" in line and "F9000" in line:
                last_travel_index = i
            
            # Track internal perimeter sections
            if ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
                inside_internal_perimeter = True
                # Set wall start to the last travel move if available, otherwise current line
                if last_travel_index is not None and i - last_travel_index <= 3:  # Travel move is close enough to be part of this wall
                    current_wall_start = last_travel_index
                else:
                    current_wall_start = i
            elif ";TYPE:" in line and inside_internal_perimeter:
                # End of internal perimeter section
                if current_wall_start is not None:
                    wall_start_end[current_layer].append((current_wall_start, i))
                inside_internal_perimeter = False
                current_wall_start = None
        
        # Create zigzags for each layer
        for layer, islands in layer_islands.items(): 
            for walls in islands:
                logging.info(f"Creating zigzags for layer {layer} with {len(walls)} internal walls")
                
                # Implement brick-layering pattern by alternating starting wall
                # Even layers start at wall 0, odd layers start at wall 1
                start_index = 1 if layer % 2 == 1 else 0
                
                # Handle the first wall in odd-numbered layers separately
                if layer % 2 == 1 and len(walls) > 0:
                    # Add the first wall as an individual wall
                    first_wall = walls[0]
                    original_lines = []
                    for move in first_wall:
                        original_lines.append(move.line)
                    zigzag_segments[layer].append(original_lines)
                    logging.info(f"Added first wall in odd layer {layer} as individual wall")
                
                # Process walls in pairs with the appropriate starting index
                l= len(walls)
                polarity = 0
                skip_next = False
                for i in range(start_index, l, 1):
                    if skip_next:
                        skip_next=False
                        continue
                    if i + 1 < l:  # Make sure we have a pair
                        wall = walls[i]
                        r1 = wall[0].get_point()
                        r2 = walls[i+1][0].get_point()
                        g1 = norm_point(sub_points(r2,r1))
                        g2 = norm_point(Point(-r1.y,r1.x))
                        dot = g1.x*g2.x+g1.y*g2.y
                        o = -0.45
                        new_polarity=-1;
                        if (dot>0):
                            o=-o
                            new_polarity=1
                        if new_polarity!=polarity:
                            if polarity!=0:
                                polarity=new_polarity
                                original_lines = []
                                for move in wall:
                                    original_lines.append(move.line)
                                zigzag_segments[layer].append(original_lines)
                                continue
                            else:
                                polarity=new_polarity
                        #logging.info(f"logging dot {dot}")
                        # Skip very short walls
                        if len(wall) < 3:
                            logging.info(f"Skipping short walls: Wall1={len(wall)} points")
                            # Add the original wall lines instead of skipping
                            original_lines = []
                            for move in wall:
                                original_lines.append(move.line)
                            zigzag_segments[layer].append(original_lines)
                            continue
                        
                        logging.info(f"Combining walls {i} and {i+1} in layer {layer}")
                        
                        # Create more segments for a visible zigzag effect
                        # More segments = more zigzag effect
                        
                        # Get evenly distributed points along both walls
                        wall2_points = distribute_points(wall,0.45)
                        wall1_points = offset(wall2_points,o)
                        
                        #Flip walls every other layer
                        if layer%2>0:
                            swap = wall1_points
                            wall1_points = wall2_points
                            wall2_points = swap

                        # Create true zigzag by connecting corresponding points between walls
                        zigzag = []
                        last_e = None
                        
                        # Add first move to position (travel move)
                        zigzag.append(f"G1 X{wall1_points[0].x:.3f} Y{wall1_points[0].y:.3f} F9000 ; Start zigzag\n")
                        
                        # Set initial extrusion value
                        if wall1_points[0].e is not None:
                            last_e = wall1_points[0].e
                        elif wall2_points[0].e is not None:
                            last_e = wall2_points[0].e
                        else:
                            # If no E value found, estimate a reasonable starting value
                            last_e = 1.5  # Typical starting E value
                        zigzag_points=[]
                        for j in range(0,len(wall2_points),1):
                            if j%2>0:
                                zigzag_points.append(wall1_points[j])
                                zigzag_points.append(wall2_points[j])
                            else:
                                zigzag_points.append(wall2_points[j])
                                zigzag_points.append(wall1_points[j])
                        if (len(wall2_points)%2==0):
                            zigzag_points.pop()
                            zigzag_points.pop()
                        zigzag_points.append(zigzag_points[1])
                        # Generate a zigzag pattern alternating between wall1 and wall2
                        for j in range(0,len(zigzag_points)-1):    
                                # Calculate distance and extrusion
                                start_point = zigzag_points[j].get_point()
                                end_point = zigzag_points[j+1].get_point()
                                distance = start_point.distance(end_point)
                                if distance>0:
                                    # Use consistent extrusion rate
                                    extrusion_rate = 0.033  # mm of filament per mm of travel
                                    new_e = (distance * extrusion_rate)
                                    
                                    # Add extrusion move
                                    zigzag.append(f"G1 X{end_point.x:.3f} Y{end_point.y:.3f} E{new_e:.5f} ; Zigzag segment {j}\n")
                            
                        # Important: Add a travel move to the end position of the original last wall
                        # This ensures the nozzle is positioned correctly for the next operation (e.g., external perimeter)
                        if len(wall) > 0:
                            # Use the last point of the second wall as the final position
                            final_pos = walls[i+1][-1]
                            zigzag.append(f"G1 X{final_pos.x:.3f} Y{final_pos.y:.3f} F9000 ; Travel to end position for next operation\n")
                            logging.info(f"Added final positioning move to X={final_pos.x:.3f} Y={final_pos.y:.3f}")
                        
                        # Store this zigzag pattern
                        zigzag_segments[layer].append(zigzag)
                        skip_next=True

                    else:
                        # Handle unpaired wall by adding its original lines
                        if i < l:  # Make sure the wall exists
                            unpaired_wall = walls[i]
                            original_lines = []
                            for move in unpaired_wall:
                                original_lines.append(move.line)
                            zigzag_segments[layer].append(original_lines)
                            logging.info(f"Added unpaired wall {i} with {len(unpaired_wall)} points as original lines")
        # Create the modified G-code with zigzag walls replacing original walls
        output_gcode = []
        current_layer = 0
        added_zigzag = False
        current_in_perimeter_block = False
        
        # Count how many layers actually got zigzags
        layers_with_zigzags = len(zigzag_segments)
        logging.info(f"Created zigzags for {layers_with_zigzags} layers")
        
        if layers_with_zigzags == 0:
            logging.warning("No zigzag patterns were created! Check for issues with wall detection.")
            output_gcode = lines  # Just use the original file
        else:
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Track layer changes
                if ';LAYER:' in line:
                    layer_match = re.search(r';LAYER:(\d+)', line)
                    if layer_match:
                        current_layer = int(layer_match.group(1))
                        output_gcode.append(line)
                        added_zigzag=False
                elif ';LAYER_CHANGE' in line:
                    current_layer += 1
                    added_zigzag=False
                    output_gcode.append(line)
                
                # Handle perimeter sections
                elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
                    output_gcode.append(line)  # Keep the perimeter type marker
                    # Check if we have zigzags for this layer
                    if current_layer in zigzag_segments and zigzag_segments[current_layer]:
                        # Make a copy of the zigzags to modify
                        zigzags_to_use = zigzag_segments[current_layer].copy()
                        
                        # Find next travel move or external perimeter marker
                        next_travel_moves = []
                        next_external_perimeter = None
                        j = i
                        while j < len(lines):
                            if lines[j].startswith("G1") and "X" in lines[j] and "Y" in lines[j] and not "E" in lines[j]:
                                if ";TYPE:External perimeter" in lines[j-1] or any(";TYPE:External perimeter" in lines[k] for k in range(j, j+5)):
                                    next_travel_moves.append(lines[j]+"\n")
                                    logging.info(f"Found next travel move for external perimeter at line {j}: {lines[j]}")
                            elif ";TYPE:External perimeter" in lines[j]:
                                next_external_perimeter = j
                                logging.info(f"Found external perimeter marker at line {j}")
                                break
                            j += 1
                        if added_zigzag==False:
                            # Insert zigzags instead of original perimeter
                            for zigzag in zigzags_to_use:
                                output_gcode.append(";ZIGZAG_PERIMETER_REPLACEMENT\n")
                                output_gcode.extend(zigzag)
                                output_gcode.append(";END_ZIGZAG_PERIMETER\n")
                        added_zigzag=True
                        output_gcode.extend(next_travel_moves)

                        # Skip all lines until external perimeter or next type
                        skip_until = False
                        j = i + 1
                        while j < len(lines):
                            if (";TYPE:External perimeter" in lines[j] or 
                                (";TYPE:" in lines[j] and not (";TYPE:Perimeter" in lines[j] or ";TYPE:Overhang perimeter" in lines[j]))):
                                # We've found the end of the internal perimeter section
                                i = j - 1  # -1 because we'll increment i at the end of the loop
                                skip_until = True
                                break
                            j += 1
                        
                        if not skip_until:  # If we didn't find a type change, go to end of file
                            i = len(lines) - 1
                        
                    else:
                        # No zigzags for this layer, keep original
                        current_in_perimeter_block = True
                
                # For all other lines, just copy them
                else:
                    output_gcode.append(line)
                
                i += 1
        
        # Write the modified G-code to a file
        output_file = input_file
        with open(output_file, 'w') as outfile:
            outfile.writelines(output_gcode)
            logging.info(f"Wrote {len(output_gcode)} lines to output file: {output_file}")
        
        # Optional: Write a debug copy
        debug_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zigzag_debug_output.gcode")
        with open(debug_file, 'w') as debugfile:
            debugfile.writelines(output_gcode)
            logging.info(f"Wrote {len(output_gcode)} lines to debug file: {debug_file}")
        
        return output_file
    except Exception as e:
        logging.error(f"Error processing G-code: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Return the original file unchanged
        return input_file

def calculate_wall_length(wall):
    """Calculate the total length of a wall path"""
    if len(wall) < 2:
        return 0
    
    total_length = 0
    for i in range(len(wall) - 1):
        p1 = wall[i].get_point()
        p2 = wall[i+1].get_point()
        total_length += p1.distance(p2)
    
    return total_length
def sum_points(p1,p2):
    return Point(p1.x+p2.x,p1.y+p2.y)
def sub_points(p1,p2):
    return Point(p1.x-p2.x,p1.y-p2.y)
def norm_point(p1):
    length = math.sqrt(p1.x*p1.x+p1.y*p1.y)
    if length>0:
        return Point(p1.x/length,p1.y/length)
    else:
        return Point(0,0)
def offset(points,distance):
    weighted_normals=[]
    l=len(points)
    for i in range(l):
        p1 = points[i].get_point()
        p2 = points[(i+1)%l].get_point()
        weighted_normals.append(sub_points(p1,p2))
    logging.info(f"calculated {weighted_normals} normals of {l}")
        
    vertex_normals=[]
    for i in range(l):
        j=(i+l-1)%l
        p1=sum_points(weighted_normals[i],weighted_normals[j])
        length=Point(0,0).distance(p1)
        vertex_normals.append(Point(-p1.y/length,p1.x/length))
    result=[]
    for i in range(l):
        point=points[i].get_point()
        new_x=vertex_normals[i].x*distance+point.x
        new_y=vertex_normals[i].y*distance+point.y
        new_e=points[i].e

        # Create a new point
        new_line = f"G1 X{new_x:.3f} Y{new_y:.3f}" + (f" E{new_e:.5f}" if new_e else "") + "\n"
        new_point = GCodeMove(new_line, new_x, new_y, new_e, False)
        result.append(new_point)
    return result
        
def evenly_distribute_points(wall, num_points):
    """Distribute points evenly along the wall path based on distance"""
    if len(wall) < 2 or num_points < 2:
        return wall
    
    total_length = calculate_wall_length(wall)
    segment_length = total_length / (num_points - 1)
    
    result = [wall[0]]  # Always include first point
    current_distance = 0
    target_distance = segment_length
    
    for i in range(len(wall) - 1):
        p1 = wall[i].get_point()
        p2 = wall[i+1].get_point()
        segment_dist = p1.distance(p2)
        
        if current_distance + segment_dist >= target_distance:
            # Need to insert a point in this segment
            while current_distance + segment_dist >= target_distance:
                # Calculate how far along this segment the point should be
                ratio = (target_distance - current_distance) / segment_dist
                
                # Interpolate the point
                new_x = p1.x + ratio * (p2.x - p1.x)
                new_y = p1.y + ratio * (p2.y - p1.y)
                
                # Interpolate E value if available
                new_e = None
                if wall[i].e is not None and wall[i+1].e is not None:
                    new_e = wall[i].e + ratio * (wall[i+1].e - wall[i].e)
                
                # Create a new point
                new_line = f"G1 X{new_x:.3f} Y{new_y:.3f}" + (f" E{new_e:.5f}" if new_e else "") + "\n"
                new_point = GCodeMove(new_line, new_x, new_y, new_e, False)
                result.append(new_point)
                
                # Update for next point
                current_distance = 0
                target_distance = segment_length
                
                # Update segment for next iteration
                p1 = Point(new_x, new_y)
                segment_dist = p1.distance(p2)
        else:
            current_distance += segment_dist
    
    # Always include last point
    if len(result) < num_points:
        result.append(wall[-1])
    
    return result

def distribute_points(wall, segment_length):
    """Distribute points evenly along the wall path based on distance"""
    
    
    total_length = calculate_wall_length(wall)
    num_points = segment_length/total_length
    
    #if len(wall) < 2 or num_points < 2:
    #    return wall
    
    result = [wall[0]]  # Always include first point
    current_distance = 0
    target_distance = segment_length
    
    for i in range(len(wall) - 1):
        p1 = wall[i].get_point()
        p2 = wall[i+1].get_point()
        segment_dist = p1.distance(p2)
        
        if current_distance + segment_dist >= target_distance:
            # Need to insert a point in this segment
            while current_distance + segment_dist >= target_distance:
                # Calculate how far along this segment the point should be
                ratio = (target_distance - current_distance) / segment_dist
                
                # Interpolate the point
                new_x = p1.x + ratio * (p2.x - p1.x)
                new_y = p1.y + ratio * (p2.y - p1.y)
                
                # Interpolate E value if available
                new_e = None
                if wall[i].e is not None and wall[i+1].e is not None:
                    new_e = wall[i].e + ratio * (wall[i+1].e - wall[i].e)
                
                # Create a new point
                new_line = f"G1 X{new_x:.3f} Y{new_y:.3f}" + (f" E{new_e:.5f}" if new_e else "") + "\n"
                new_point = GCodeMove(new_line, new_x, new_y, new_e, False)
                result.append(new_point)
                
                # Update for next point
                current_distance = 0
                target_distance = segment_length
                
                # Update segment for next iteration
                p1 = Point(new_x, new_y)
                segment_dist = p1.distance(p2)
        else:
            current_distance += segment_dist
    
    # Always include last point
    if len(result) < num_points:
        result.append(wall[-1])
    
    return result


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process G-code to combine adjacent walls into zigzag patterns.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    parser.add_argument("--zigzag-length", type=float, default=0.4, 
                        help="Length of each zigzag segment (mm, default: 2.0)")
    
    args = parser.parse_args()
    
    modified_file = process_gcode(
        input_file=args.input_file,
        zigzag_length=args.zigzag_length
    )
    
    print(f"G-code file modified in place: {modified_file}")