# Only Bricklayers
This is a script to add Brick layers to Prusaslicer and Orcaslicer.
(As of now it doesn't work with Bambu printers)

To use it you need to have Python installed. (www.python.org) 

In Prusaslicer's printsettings go to "Output options". There you will find a section called "Post processing scripts". 
You can add the following to run the script:

```"C:\Your\Path\To\Python\python.exe" "C:\Your\Path\To\Script\bricklayers.py"```

This will run it with a default layerheight of 0.2.

There are two parameters you can add. -layerHeight and -extrusionMultiplier

The layerHeight has to match the settings in the slicer to work as intended,
The extrusionmultiplier multiplies the extrusions of the shifted layers so you can use it to probably increase strenght(has yet to be tested).

Sample: 

```"C:\Your\Path\To\Python\python.exe" "C:\Your\Path\To\Script\bricklayers.py" -layerHeight 0.2 -extrusionMultiplier 1.3```

Thanks to all of you who opened issues and made pullrequests. I'm not ignoring you, I just didn't have the time to review yet. I will do on the weekend!<3
(I will also make a good readme then)

Here is a video about the script.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/EqRdQOoK5hc/0.jpg)](https://www.youtube.com/watch?v=EqRdQOoK5hc)


# Bricklayers + NonPlanar infill (Currently only tested in Prusaslicer!)

(For better results turn on External perimeters first)

In Prusaslicer's printsettings go to "Output options". There you will find a section called "Post processing scripts". 

You can add the following to run the script:

```"C:\Your\Path\To\Python\python.exe" "C:\Your\Path\To\Script\bricklayersNonPlanarInfill.py" -extrusionMultiplier 1.05 -wallReorder 0 -nonPlanar 1 -amplitude 0.6 -frequency 1.1;```

The parameters are
-extrusionMultiplier {number} -> Increase the extrusion for shifted walls

-wallReorder {1 or 0} -> enable/disable wall loop reordering.

-nonPlanar {1 or 0} -> enable/disable non-planar infill

-amplitude {number} -> amplitude of the infill sine wave

-frequency {number} -> frequency of the infill sine wave 

Here is a video about this version.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/DosU-M0g-QU/0.jpg)](https://www.youtube.com/watch?v=DosU-M0g-QU)

# Non-Planar Interlocking walls

Please make sure to disable Arcfitting and .bgcode for this to work. 

Parameters: 

-include-infill   -> process the infill

-include-perimeters -> process the perimeters

-include-external-perimeters -> process the external perimeters

-wall-amplitude + number -> amplitude of sinewave used on walls

-wall-frequency + number -> frequency of sinewave used on walls

-infill-amplitude + number -> amplitude of sinewave used on infill

-infill-frequency + number -> frequency of sinewave used on infill

-infill-direction + one of y, x ,xy , negx, negy, negxy ->direction of sinewave for infill

-wall-direction + one of y, x ,xy , negx, negy, negxy ->direction of sinewave for walls

-max-step-size + number -> maximal amplitude increase per layer

-alternate-loops -> activate the alternating sinewaves of walls for interlocking effect

-perimeter-function + one of sine, triangular, trapezoidal, sawtooth -> to select a peridoic function (default: sine)

-infill-function + one of sine, triangular, trapezoidal, sawtooth -> to select a peridoic function (default: sine)

-resolution + number -> set the segmenting resolution in mm. (How many times your G1 movement gets segmented to apply deformation)

Sample usage:

```"C:\Path\To\Python\python.exe" "C:\Path\To\Script\InterlockingWalls.py" -include-infill -infill-amplitude 0.3 -include-external-perimeter -infill-frequency 2 -include-perimeters -infill-direction y -wall-direction xy -max-step-size 0.1 -wall-amplitude 0.4 -wall-frequency 2 -infill-function sawtooth -perimeter-function triangle -resolution 0.3;```

More info in these video:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/r9YdJhN6jWQ/0.jpg)](https://www.youtube.com/watch?v=r9YdJhN6jWQ)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/WirZadHmQ3A/0.jpg)](https://www.youtube.com/watch?v=WirZadHmQ3A)
