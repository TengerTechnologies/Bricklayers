# Brick Layers a G-code Z-Shift Processor

A Orca Slicer or PrusaSlicer Python script for post-processing G-code files to enhance wall strength in 3D prints by applying Z-height shifts and extrusion adjustments to inner perimeters.

These videos show off the concept well:

[TenTech](https://www.youtube.com/watch?v=EqRdQOoK5hc&t=1s)      
[CNC Kitchen](https://www.youtube.com/watch?v=5hGm6cubFVs&t=1s)  
[Geek Detour](https://www.youtube.com/watch?v=9IdNA_hWiyE)

## Features

- Automatic layer height detection from G-code
- Printer type detection (Bambu/Orca and Prusa supported)
- Z-height shifting for inner walls
- Extrusion multiplier adjustment
- Detailed logging of all modifications
- Support for multi-object printing (M624/M625 commands)

## Installation

```bash
git clone [https://github.com/drkpxl/Bricklayers](https://github.com/drkpxl/Bricklayers)
cd Bricklayers
```

## Usage

### In Your Slicer

1. Enable post-processing scripts in your slicer
2. Add the script with parameters (on a modern Mac running OSX:
```
"/usr/bin/python3" "/path/to/script/gcode_processor.py" -layerHeight 0.2 -extrusionMultiplier 1.1
```

### Parameters

- `-layerHeight`: Layer height in mm (default: auto-detect)
- `-extrusionMultiplier`: Extrusion multiplier for inner walls (default: 1.0)

## How It Works

1. Detects printer type from G-code comments
2. Applies Z-shift of 50% layer height to inner walls
3. Adjusts extrusion rates for shifted sections
4. Maintains proper Z transitions between features

## Logging

Creates `z_shift_log.txt` in script directory with:
- Detected settings and printer type
- Layer changes
- Z-shift operations
- Processing statistics

## License

GNU General Public License v3.0
