import re
import sys
import logging
import os
import argparse
from tempfile import NamedTemporaryFile
import io

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

    def process_line(self, line, is_last_layer):
        """Process a single line of G-code"""
        modified_lines = []
        
        # Detect layer changes
        if line.startswith("G1 Z"):
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                self.current_z = float(z_match.group(1))
                self.current_layer = int(self.current_z / self.layer_height)
                self.perimeter_block_count = 0
                logging.info(f"Layer {self.current_layer} detected at Z={self.current_z:.3f}")
            modified_lines.append(line)
            return modified_lines

        # Detect perimeter types
        if ";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line:
            self.perimeter_type = "external"
            self.inside_perimeter_block = False
        elif ";TYPE:Perimeter" in line or ";TYPE:Inner wall" in line:
            self.perimeter_type = "internal"
            self.inside_perimeter_block = False
        elif ";TYPE:" in line:
            self.perimeter_type = None
            self.inside_perimeter_block = False

        # Process perimeter blocks
        if self.perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line and "E" in line:
            if not self.inside_perimeter_block:
                self.perimeter_block_count += 1
                self.inside_perimeter_block = True
                
                is_shifted = False
                if self.perimeter_block_count % 2 == 1:
                    adjusted_z = self.current_z + self.z_shift
                    modified_lines.append(f"G1 Z{adjusted_z:.3f} ; Shifted Z for block #{self.perimeter_block_count}\n")
                    is_shifted = True
                else:
                    modified_lines.append(f"G1 Z{self.current_z:.3f} ; Reset Z for block #{self.perimeter_block_count}\n")

                if is_shifted:
                    e_match = re.search(r'E([-\d.]+)', line)
                    if e_match:
                        e_value = float(e_match.group(1))
                        if self.current_layer == 0:
                            new_e_value = e_value * 1.5
                        elif is_last_layer:
                            new_e_value = e_value * 0.5
                        else:
                            new_e_value = e_value * self.extrusion_multiplier
                        
                        line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line).strip()
                        line += f" ; Adjusted E for layer {self.current_layer}, block #{self.perimeter_block_count}\n"

        elif self.perimeter_type == "internal" and line.startswith("G1") and "X" in line and "Y" in line and "F" in line:
            self.inside_perimeter_block = False

        modified_lines.append(line)
        return modified_lines

    def process_gcode(self, input_file):
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

        # Count total layers first
        self.total_layers = self.count_layers(input_file)
        logging.info(f"Total layers detected: {self.total_layers}")

        # Process the file using a temporary file
        temp_file = NamedTemporaryFile(mode='w', delete=False)
        try:
            with open(input_file, 'r') as infile:
                buffer = io.StringIO()
                while True:
                    chunk = infile.read(self.buffer_size)
                    if not chunk:
                        break
                    
                    buffer.write(chunk)
                    buffer.seek(0)
                    
                    for line in buffer:
                        is_last_layer = self.current_layer == self.total_layers - 1
                        modified_lines = self.process_line(line, is_last_layer)
                        temp_file.writelines(modified_lines)
                    
                    buffer.seek(0)
                    buffer.truncate(0)
                    
            temp_file.close()
            
            # Replace original file with processed file
            os.replace(temp_file.name, input_file)
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
        
        logging.info("G-code processing completed")
        logging.info(f"Log file saved at {log_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Memory-efficient G-code post-processor for Z-shifting and extrusion adjustments.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    parser.add_argument("-layerHeight", type=float, default=0.2, help="Layer height in mm (default: 0.2mm)")
    parser.add_argument("-extrusionMultiplier", type=float, default=1, help="Extrusion multiplier (default: 1.0)")
    args = parser.parse_args()

    processor = GCodeProcessor(
        layer_height=args.layerHeight,
        extrusion_multiplier=args.extrusionMultiplier
    )
    processor.process_gcode(args.input_file)

if __name__ == "__main__":
    main()