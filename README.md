# Bricklayers G-code Post Processor

Open-source tool for modifying 3D printing G-code files to alternate Z-axis layer heights on internal perimeters.

[![Demonstration Video](https://img.youtube.com/vi/EqRdQOoK5hc/0.jpg)](https://www.youtube.com/watch?v=EqRdQOoK5hc)

## Key Features

- Compatible with common slicing software output
- Adjustable Z-axis layer shift patterns
- Configurable extrusion multipliers
- Automatic layer height detection
- Basic geometry validation

## Requirements

- Python 3.9+ ([python.org](https://www.python.org/))
- 64-bit operating system recommended
- 2X free RAM relative to G-code file size

## Installation

1. Install Python dependencies:

   ```bash
   pip install shapely
   pip install psutil # Optional, for memory usage logging
   ```
2. Download script:

   ```
   wget https://example.com/bricklayers.py
   ```


## Usage Instructions

### Basic Configuration

bash

```
python bricklayers.py input.gcode -layerHeight 0.2
```

### Full Parameter List

bash

```
python bricklayers.py input.gcode \
  -layerHeight 0.2 \
  -extrusionMultiplier 1.2 \
  -simplifyTolerance 0.03 \
  -bgcode \
  --logLevel INFO
```

## Technical Notes

* Processes standard G-code (ASCII) and Prusa binary formats
* Creates backup files with timestamped logs
* Detailed processing reports in `bricklayers.log`

## Important Disclaimers

This software is provided **as-is** without any warranties. By using this tool, you agree:

1. To validate all processed files in simulation software before printing
2. That improper use may damage printers or create hazardous conditions
3. The authors are not liable for material losses or printer damage

> Test on non-critical prints first. Not certified for medical/safety-critical applications.

## License

GNU General Public License v3.0

Copyright © 2025 Roman Tenger

---

*Not affiliated with or endorsed by Prusa Research, Bambu Lab, or Anycubic.

All trademarks remain property of their respective owners.*
