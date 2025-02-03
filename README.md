
# Bricklayers Adaptive G-code Optimizer

Advanced post-processor for structural 3D printing optimization leveraging "bricklayers" with intelligent Z-axis patterning and precision-aware geometry processing.

[![Demonstration Video](https://img.youtube.com/vi/EqRdQOoK5hc/0.jpg)](https://www.youtube.com/watch?v=EqRdQOoK5hc)

## 🚀 Key Enhancements

- **Adaptive Precision Engine** - Auto-detects optimal processing mode based on model characteristics
- **Structural Layer Shifting** - Full-layer or per-perimeter Z-axis alternation patterns
- **Smart Feature Preservation** - Maintains critical details while optimizing print geometry
- **Industrial-Grade Parameters** - Configurable for professional-grade printing requirements
- **Advanced Diagnostics** - Detailed path analysis and memory optimization

## 📋 Requirements

- Python 3.10+ (64-bit mandatory)
- 4GB+ free RAM (8GB recommended for complex models)
- SSD storage recommended for large files (>500MB)

## ⚙️ Installation

1. Install core dependencies:

```bash
pip install "shapely>=2.0"
pip install "psutil>=5.9"  # Optional for memory usage tracking
```

3. Download latest version by cloning the repo.

## 🛠 Quick Start Guide

### Basic Structural Optimization

```bash
python bricklayers.py input.gcode --precision balanced
```

### Professional Quality Profile

```bash
python bricklayers.py aerospace_part.gcode \
  --precision high_precision \
  --minDetail 0.15 \
  --criticalAngle 30 \
  --maxZSpeed 4500 \
  --hausdorffMult 0.2
```

### Production Speed Profile

```bash
python bricklayers.py prototype.gcode \
  --precision draft \
  --fullLayerShifts \
  --minDetail 0.4 \
  --maxZSpeed 12000
```

## 🔧 Advanced Configuration

### Core Parameters

| Parameter             | Values                            | Description                               |
| --------------------- | --------------------------------- | ----------------------------------------- |
| `--precision`       | auto/high/balanced/draft/disabled | Processing rigor level                    |
| `--minDetail`       | 0.1-1.0 (mm)                      | Minimum preserved feature size            |
| `--criticalAngle`   | 15-45 (degrees)                   | Sharpest angle to maintain                |
| `--fullLayerShifts` | Flag                              | Enable structural full-layer Z patterning |

### ⚙️ Complete Parameter Reference

### Layer Configuration

| Parameter               | Type  | Default | Description                               |
| ----------------------- | ----- | ------- | ----------------------------------------- |
| `-layerHeight`        | float | Auto    | Manual layer height override (mm)         |
| `-fullLayerShifts`    | flag  | On      | Structural full-layer Z patterning        |
| `-perPerimeterShifts` | flag  | Off     | Legacy per-perimeter Z shifts             |
| `-minZMoveTime`       | float | 0.5     | Minimum Z move duration (seconds)         |
| `-safeZDistance`      | float | 1.0     | Safety threshold for Z speed ramping (mm) |

### Extrusion Control

| Parameter                 | Type  | Default | Description                        |
| ------------------------- | ----- | ------- | ---------------------------------- |
| `-extrusionMultiplier`  | float | 1.0     | Global extrusion multiplier        |
| `-firstLayerMultiplier` | float | 1.5×   | First layer flow boost             |
| `-lastLayerMultiplier`  | float | 0.5×   | Final layer reduction              |
| `-simplifyTolerance`    | float | 0.03    | Base simplification tolerance (mm) |

### Geometry Processing

| Parameter                 | Type  | Default | Description                             |
| ------------------------- | ----- | ------- | --------------------------------------- |
| `-minDetail`            | float | 0.2     | Minimum preserved feature size (mm)     |
| `-criticalAngle`        | float | 25      | Minimum preserved angle (degrees)       |
| `-hausdorffMult`        | float | 0.3     | Simplification aggressiveness (0.1-1.0) |
| `-maxAreaDev`           | float | 0.06    | Maximum area deviation (1=100%)         |
| `-min_perimeter_points` | int   | 3       | Minimum points to validate perimeter    |

### Speed Configuration

| Parameter      | Type  | Default | Description                   |
| -------------- | ----- | ------- | ----------------------------- |
| `-maxZSpeed` | float | 6000    | Maximum Z-axis speed (mm/min) |
| `-zSpeed`    | float | Auto    | Manual Z move speed override  |

### Precision Controls

| Parameter       | Values                            | Description                        |
| --------------- | --------------------------------- | ---------------------------------- |
| `--precision` | auto/high/balanced/draft/disabled | Processing rigor level             |
| `-bgcode`     | flag                              | Enable Prusa binary G-code support |

### Diagnostic Parameters

| Parameter      | Values                            | Description      |
| -------------- | --------------------------------- | ---------------- |
| `--logLevel` | DEBUG/INFO/WARNING/ERROR/CRITICAL | Output verbosity |

---

## Example Full Configuration

bash

```
python bricklayers2.py high_precision_part.gcode \
  -layerHeight 0.15 \
  -extrusionMultiplier 1.05 \
  -firstLayerMultiplier 1.8 \
  -lastLayerMultiplier 0.4 \
  -simplifyTolerance 0.02 \
  -minDetail 0.12 \
  -criticalAngle 30 \
  -hausdorffMult 0.25 \
  -maxAreaDev 0.04 \
  -maxZSpeed 8000 \
  -minZMoveTime 0.75 \
  -safeZDistance 1.5 \
  -minPerimeterPoints 4 \
  --precision high_precision \
  --logLevel DEBUG
```


### Optimization Controls

```bash
# Geometry Processing
--hausdorffMult 0.3      # Simplification aggressiveness (0.1-1.0)
--maxAreaDev 0.06        # Maximum allowed geometry deviation (1-10%)

# Speed Configuration
--maxZSpeed 6000         # Maximum Z-axis speed (mm/min)
--minZMoveTime 0.5       # Minimum layer shift duration (seconds)

# Advanced Features
--perPerimeterShifts   # Legacy per-perimeter mode (not recommended)
--minPerimeterPoints 3 # Minimum points to consider valid perimeter
```

## 📊 Diagnostic Features

* Layer-by-layer processing statistics
* Path closure validation tracking
* Small feature preservation reports
* Memory optimization profiling
* Error resilience logging

## ⚠️ Critical Safety Protocols

**By using this software, you acknowledge and agree:**

1. **Mandatory Pre-Validation**

   All processed files must undergo:

   * Full G-code simulation
   * Thermal stress analysis
   * Mechanical property verification
2. **Safety Certification**

   Not approved for:

   * Medical implants
   * Aerospace components
   * Load-bearing structures
   * Safety-critical applications
3. **Liability Disclaimer**

   The developers assume **no responsibility** for:

   * Printer damage
   * Material losses
   * Production downtime
   * Secondary damages from printed objects

> **WARNING:** Always conduct test prints at 50% scale before full production. Monitor first 10 layers manually.

## 📜 License Compliance

GNU GPLv3 - Full text included with distribution. By using this software:

* You must disclose source code modifications
* Commercial use requires explicit authorization
* No warranty protection provided

---

*This project is not affiliated with Bambu Lab, Prusa Research, UltiMaker, or any commercial 3D printing vendor.

All trademarks remain property of their respective owners. Use of vendor logos strictly prohibited.*
