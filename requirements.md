I want to write a script to implement "bricklayering" in GCode. The intended design is as follows, assuming a 0.2 mm layer/extrusion height (the height of the actual plastic being deposited in the layer):

| Layer Num | Perimeter Extrusion Height (% of Normal) | Perimeter Z | Non-Perimeter Extrusion Height (% of Normal) | Non-Perimeter Z |
| --------- | ---------------------------------------- | ----------- | -------------------------------------------- | --------------- |
| 1         | 100                                      | 0.2 mm      | 150                                          | 0.2 mm          |
| 2         | 100                                      | 0.4 mm      | 100                                          | 0.5 mm          |
| 3         | 100                                      | 0.6 mm      | 100                                          | 0.7 mm          |
| 4         | 100                                      | 0.8 mm      | 100                                          | 0.9 mm          |
| 5         | 100                                      | 1.0 mm      | 100                                          | 1.1 mm          |

And so on.

The script needs to have these features by the end of development:

- Automatic perimeter and non-perimeter classification
- Fallback Shapely-based polygon processing to handle complex geometries like the 3DBenchy where multiple 'outer' walls exist in the same layer
- Layer boundary proximity detection
- Spatial containment hierarchy analysis
- Travel move detection with nozzle-diameter awareness
- Bambu Studio and PrusaSlicer compatibility
- Automatic printer type detection
- Slicer-specific comment parsing
- BGCode file decoding support
- Precision Processing
- Path closure validation
- Extrusion mode detection (absolute/relative)
- Failed geometry fallback handling
- Collision Prevention
- Layer height validation
- Logging with detailed layer-by-layer reports, memory usage tracking, processing time statistics, error classification (failed simplifications, decoding errors), what is found by parsers, what decisions were made by the script on that layer, whether or not a z-shift is applied, if so, the magnitude and direction thereof, if not, why the z-shift was not applied, the pre-modification gcode, and the post-modification gcode for help with close debugging
- Multi-handler logging (file + console) that is RFC-5424 compliant logging system
- Automatic layer height detection
- Automatic nozzle diameter detection
- Configurable extrusion multiplier parameters (overall/first layer/last layer)
- Shift comment annotation in G-code
- Comprehensive final report in log
- Log file rotation (5x10MB)
- Timestamped logs for each run
- Temporary file safety handling
- MMap-based file processing
- STRtree spatial indexing
- Geometry validation (make_valid)
- Parallel path processing
- Unicode error fallback handling
- Geometry processing try/catch
- Buffer overflow protection
- Layer processing fail-safes
- configurable Log level control
- configurable Layer height override
