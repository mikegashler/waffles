Build Instructions:
	Linux:
		1- Install the following dependency packages:
			g++
			make
			libpng3-dev
			libsdl1.2-dev (needed only for some demo apps)
			freeglut-dev (needed only for some demo apps)
		2- cd src
		3- sudo make install

		If you also want to build the demo apps:
		4- cd ../demos
		5- make opt

	Windows:
		1- Install Microsoft Visual C++ 2008 or 2010.
		   (The free Express Edition is sufficient.)
		2- File->Open->Project/Solution
		3- Open src/waffles.sln or demos/demos.sln
		4- Change to Release mode.
		5- Build (F7)
		6- Set the startup app to the app you want to try.
		7- Set any arguments in Project->Properties->Debugging->
		   Command Arguments
		8- Run it (F5)

	OSX
		1- Install Fink (a unix package manager).
		1- Install the following packages.
			g++
			make
			libpng3-dev
		2- cd waffles/src
		3- make opt

For more detailed instructions, troubleshooting help, an overview of this
toolkit, and instructions for using it, see web/docs.html.

