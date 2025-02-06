import re
import math
import sys
import logging
import argparse

# Modified parameters for bricklayering
DEFAULT_LAYER_HEIGHT = 0.2  # Will be auto-detected
SEGMENT_LENGTH = 1.0  # Split infill lines into segments of this length (mm)


class BricklayerProcessor:
    def __init__(self):
        self.layer_height = DEFAULT_LAYER_HEIGHT
        self.infill_spacing = None
        self.current_z = 0
        self.layer_num = 0
        self.in_infill = False
        self.phase = 0.0
        self.x_positions = []
        self.first_layer_detected = False

    def detect_parameters(self, lines):
        """Auto-detect layer height and infill spacing from G-code"""
        # Detect layer height
        z_values = []
        for line in lines:
            if line.startswith("G1") and "Z" in line:
                z_match = re.search(r"Z([-+]?\d*\.?\d+)", line)
                if z_match:
                    z_values.append(float(z_match.group(1)))

        if len(z_values) > 1:
            self.layer_height = round(z_values[1] - z_values[0], 2)
            logging.info(f"Auto-detected layer height: {self.layer_height}mm")

        # Detect infill spacing (crude approximation)
        infill_x = []
        for line in lines:
            if ";TYPE:Internal infill" in line:
                x_match = re.search(r"X([-+]?\d*\.?\d+)", line)
                if x_match:
                    infill_x.append(float(x_match.group(1)))

        if len(infill_x) > 1:
            self.infill_spacing = abs(infill_x[1] - infill_x[0])
            logging.info(f"Auto-detected infill spacing: {self.infill_spacing}mm")

    def calculate_wave_parameters(self):
        """Calculate wave parameters based on detected geometry"""
        # Amplitude = 0.5 * layer_height for non-planar effect
        self.amplitude = self.layer_height * 0.5

        # Wavelength = 2 * infill spacing (if detected)
        if self.infill_spacing:
            self.wavelength = 2 * self.infill_spacing
        else:
            # Fallback to conservative default
            self.wavelength = 10.0  # mm

        logging.info(
            f"Using amplitude: {self.amplitude}mm, wavelength: {self.wavelength}mm"
        )

    def process_line(self, line):
        """Process individual G-code line"""
        # Track layer changes
        if line.startswith("G1") and "Z" in line:
            z_match = re.search(r"Z([-+]?\d*\.?\d+)", line)
            if z_match:
                new_z = float(z_match.group(1))
                if new_z != self.current_z:
                    self.layer_num += 1
                    self.phase = math.pi * self.layer_num  # Phase shift per layer
                    self.current_z = new_z

        # Track infill sections
        if ";TYPE:Internal infill" in line:
            self.in_infill = True
            self.x_positions = []  # Reset x tracking for new infill section
        elif line.startswith(";TYPE:"):
            self.in_infill = False

        # Process infill moves
        if self.in_infill and "E" in line and "X" in line:
            return self.modify_infill(line)

        return line

    def modify_infill(self, line):
        """Apply bricklayering modifications to infill"""
        # For first layer: modify extrusion
        if self.layer_num == 1:
            return self.modify_first_layer(line)

        # For other layers: modify Z-height
        return self.modify_upper_layers(line)

    def modify_first_layer(self, line):
        """First layer: 150% extrusion with original Z"""
        e_match = re.search(r"E([-+]?\d*\.?\d+)", line)
        if e_match:
            e_value = float(e_match.group(1)) * 1.5
            return re.sub(r"E[-+]?\d*\.?\d+", f"E{e_value:.5f}", line)
        return line

    def modify_upper_layers(self, line):
        """Upper layers: sinusoidal Z modulation"""
        x_match = re.search(r"X([-+]?\d*\.?\d+)", line)
        if x_match:
            x_pos = float(x_match.group(1))
            z_mod = self.current_z + self.amplitude * math.sin(
                (2 * math.pi * x_pos / self.wavelength) + self.phase
            )
            return re.sub(r"Z[-+]?\d*\.?\d+", f"Z{z_mod:.3f}", line)
        return line


def process_gcode(input_file, output_file):
    processor = BricklayerProcessor()

    with open(input_file, "r") as f:
        lines = f.readlines()

    # Auto-detect parameters from initial lines
    processor.detect_parameters(lines[:500])  # Check first 500 lines
    processor.calculate_wave_parameters()

    # Process all lines
    modified_lines = []
    for line in lines:
        modified_lines.append(processor.process_line(line))

    # Save output
    with open(output_file, "w") as f:
        f.writelines(modified_lines)
    logging.info(f"Saved modified G-code to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bricklayering G-code Processor")
    parser.add_argument("input_file", help="Input G-code file")
    parser.add_argument("-o", "--output", help="Output file", default="output.gcode")

    args = parser.parse_args()
    process_gcode(args.input_file, args.output)
