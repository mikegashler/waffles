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
		1- Install Microsoft Visual C++ 2008 Express Edition.
		2- File->Open->Project/Solution
		3- Open waffles/src/waffles.sln or waffles/demos/demos.sln
		4- Build (F7)
		5- Set the startup app to the app you want to try.
		6- Set arguments in Project->Properties->Debugging->
		   Command Arguments
		7- Run it (F5)

	OSX
		1- Install Fink (a unix package manager).
		1- Install the following packages.
			g++
			make
			libpng3-dev
			libsdl1.2-dev
		2- cd waffles/src
		3- make opt

		If you also want to build the demo apps:
		4- cd ../demos
		5- make opt

For more detailed instructions, troubleshooting help, an overviews of this
toolkit, and instructions for using it, see web/docs.html.
