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
