import re
import sys
import logging
import os
import argparse
from tempfile import NamedTemporaryFile
import io
from pathlib import Path

class GCodeProcessor:
    def __init__(self, layer_height, extrusion_multiplier):
        self.layer_height = layer_height
        self.extrusion_multiplier = extrusion_multiplier
        self.current_layer = 0
        self.current_z = 0.0
        self.perimeter_type = None
        self.perimeter_block_count = 0
        self.inside_perimeter_block = False
        self.z_shift = layer_height * 0.5
        self.total_layers = 0
        self.buffer_size = 1024 * 1024  # 1MB buffer size
        self.printer_type = None
        self.in_object = False
        self.shifted_blocks = 0
        self.perimeter_found = False

    def detect_printer_type(self, lines):
        """Detect printer type based on G-code features"""
        logging.info("Starting printer type detection")
        
        for i, line in enumerate(lines):
            if "; FEATURE:" in line:
                logging.info(f"Detected Bambu/Orca printer from feature marker in line {i}: {line.strip()}")
                return "bambu"
            elif ";TYPE:" in line:
                logging.info(f"Detected Prusa printer from type marker in line {i}: {line.strip()}")
                return "prusa"
        
        logging.warning("No printer type markers found - defaulting to Prusa")
        return "prusa"

    def get_z_height_from_comment(self, line):
        """Extract Z height from comment if present"""
        if "; Z_HEIGHT:" in line:
            match = re.search(r'; Z_HEIGHT: ([\d.]+)', line)
            if match:
                return float(match.group(1))
        return None

    def count_layers(self, input_file):
        """Count layers using buffered reading"""
        count = 0
        with open(input_file, 'r') as f:
            buffer = io.StringIO()
            while True:
                chunk = f.read(self.buffer_size)
                if not chunk:
                    break
                buffer.write(chunk)
                buffer.seek(0)
                
                for line in buffer:
                    if line.startswith("G1 Z"):
                        count += 1
                        
                buffer.seek(0)
                buffer.truncate(0)
        return count

    def process_line(self, line, line_num, lines, is_last_layer):
        """Process a single line of G-code"""
        modified_lines = []

        # Handle object printing sections (Bambu/Orca specific)
        if "M624" in line:  # Start printing object
            self.in_object = True
            self.perimeter_block_count = 0
        elif "M625" in line:  # Stop printing object
            self.in_object = False
            if self.inside_perimeter_block:
                modified_lines.append(f"G1 Z{self.current_z:.3f} F1200 ; Reset Z at object end\n")
                self.inside_perimeter_block = False

        # Handle layer changes
        if line.startswith("G1 Z") or "; CHANGE_LAYER" in line:
            if "; CHANGE_LAYER" in line:
                z_height = self.get_z_height_from_comment(lines[line_num + 1] if line_num + 1 < len(lines) else "")
                if z_height is not None:
                    self.current_z = z_height
                    self.current_layer += 1
            else:
                z_match = re.search(r'Z([-\d.]+)', line)
                if z_match:
                    self.current_z = float(z_match.group(1))
                    self.current_layer = int(self.current_z / self.layer_height)
            
            self.perimeter_block_count = 0
            logging.info(f"Layer {self.current_layer} detected at Z={self.current_z:.3f}")
            modified_lines.append(line)
            return modified_lines

        # Handle perimeter detection based on printer type
        if self.printer_type == "bambu":
            if "; FEATURE:" in line:
                if self.inside_perimeter_block:
                    modified_lines.append(f"G1 Z{self.current_z:.3f} F1200 ; Reset Z for feature transition\n")
                    self.inside_perimeter_block = False
                
                if "; FEATURE: Inner wall" in line:
                    self.perimeter_type = "internal"
                    self.perimeter_found = True
                elif "; FEATURE: Outer wall" in line:
                    self.perimeter_type = "external"
                else:
                    self.perimeter_type = None
        else:  # Prusa
            if ";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line:
                self.perimeter_type = "external"
                self.inside_perimeter_block = False
            elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line or ";TYPE:Internal perimeter" in line:
                self.perimeter_type = "internal"
                self.perimeter_found = True
                self.inside_perimeter_block = False
            elif ";TYPE:" in line:
                self.perimeter_type = None
                self.inside_perimeter_block = False

        # Process perimeter blocks
        should_process = (self.printer_type == "bambu" and self.in_object) or self.printer_type == "prusa"
        
        if should_process and self.perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line:
            if "E" in line:  # Extrusion move
                if not self.inside_perimeter_block:
                    self.perimeter_block_count += 1
                    self.inside_perimeter_block = True
                    
                    # Apply Z shift for odd-numbered blocks
                    if self.perimeter_block_count % 2 == 1:
                        adjusted_z = self.current_z + self.z_shift
                        modified_lines.append(f"G1 Z{adjusted_z:.3f} F1200 ; Z shift for internal perimeter\n")
                        self.shifted_blocks += 1
                        
                        # Adjust extrusion
                        e_match = re.search(r'E([-\d.]+)', line)
                        if e_match:
                            e_value = float(e_match.group(1))
                            if self.current_layer == 0:
                                new_e_value = e_value * 1.5
                                comment = "first layer"
                            elif is_last_layer:
                                new_e_value = e_value * 0.5
                                comment = "last layer"
                            else:
                                new_e_value = e_value * self.extrusion_multiplier
                                comment = "internal perimeter"
                            
                            line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line.strip())
                            line += f" ; Adjusted E for {comment}\n"
                    else:
                        modified_lines.append(f"G1 Z{self.current_z:.3f} F1200 ; Reset Z for even block\n")
            
            elif "F" in line and not "E" in line and self.inside_perimeter_block:
                modified_lines.append(f"G1 Z{self.current_z:.3f} F1200 ; Reset Z after internal perimeter\n")
                self.inside_perimeter_block = False

        modified_lines.append(line)
        return modified_lines

    def process_gcode(self, input_file, is_bgcode=False):
        """Process G-code file using buffered reading and writing"""
        # Set up logging
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(script_dir, "z_shift_log.txt")
        logging.basicConfig(
            filename=log_file_path,
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(message)s"
        )
        
        logging.info("Starting G-code processing")
        logging.info(f"Input file: {input_file}")
        logging.info(f"Z-shift: {self.z_shift} mm, Layer height: {self.layer_height} mm")

        # Handle bgcode conversion if necessary
        input_path = Path(input_file)
        if is_bgcode:
            if input_path.suffix != '.bgcode':
                input_path = input_path.rename(str(input_path) + '.bgcode')
            if os.system('bgcode ' + str(input_path)) == 0:
                input_path = input_path.parent / (input_path.stem + '.gcode')

        # First pass to count layers and detect printer type
        with open(input_path, 'r') as infile:
            content = infile.read()
            lines = content.splitlines(True)
            self.printer_type = self.detect_printer_type(lines)
            self.total_layers = sum(1 for line in lines if line.startswith("G1 Z"))
        
        logging.info(f"Detected printer type: {self.printer_type}")
        logging.info(f"Total layers detected: {self.total_layers}")

        # Process the file using a temporary file
        temp_file = NamedTemporaryFile(mode='w', delete=False)
        try:
            with open(input_path, 'r') as infile:
                buffer = io.StringIO()
                while True:
                    chunk = infile.read(self.buffer_size)
                    if not chunk:
                        break
                    
                    buffer.write(chunk)
                    buffer.seek(0)
                    
                    lines = buffer.readlines()
                    for line_num, line in enumerate(lines):
                        is_last_layer = self.current_layer == self.total_layers - 1
                        modified_lines = self.process_line(line, line_num, lines, is_last_layer)
                        temp_file.writelines(modified_lines)
                    
                    buffer.seek(0)
                    buffer.truncate(0)
                    
            temp_file.close()
            
            # Replace original file with processed file
            os.replace(temp_file.name, input_path)
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise

        # Re-encode to bgcode if necessary
        if is_bgcode:
            os.system('bgcode ' + str(input_path))
            input_path.unlink()
            input_path = input_path.parent / (input_path.stem + '.bgcode')
            if input_path.suffix != '.bgcode':
                input_path.rename(str(input_path)[:-6])

        if not self.perimeter_found:
            logging.warning("No internal perimeters found in the file.")
        else:
            logging.info(f"Processing complete: Modified {self.shifted_blocks} blocks across {self.current_layer} layers")
        
        logging.info("G-code processing completed")
        logging.info(f"Log file saved at {log_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient G-code post-processor for Z-shifting and extrusion adjustments.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    parser.add_argument("-layerHeight", type=float, default=0.2, help="Layer height in mm (default: 0.2mm)")
    parser.add_argument("-extrusionMultiplier", type=float, default=1, help="Extrusion multiplier (default: 1.0)")
    parser.add_argument("-bgcode", action='store_true', help="Input file is binary gcode (PrusaSlicer default from v2.7)")
    args = parser.parse_args()

    processor = GCodeProcessor(
        layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier
    )
    processor.process_gcode(args.input_file, args.bgcode)

if __name__ == "__main__":
    main()