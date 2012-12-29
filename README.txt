Build Instructions:

	Linux:
		1- Install g++ and make (if they are not already installed)
			On Debian, Ubuntu, and derivatives:
				sudo apt-get install g++ make
			On Red Hat, Fedora, and derivatives:
				sudo yum install g++ make
		2- cd src
		3- sudo make install

		To uninstall:
			sudo make uninstall

		To build optimized binaries without installing them:
			make opt

		To build demo apps and additional tools:
			see web/docs.html

		If you also want to build the demo apps:
		4- Install the following dependency packages:
			libpng12-dev ( libpng-devel )
			libsdl1.2-dev ( SDL-devel )
			freeglut3-dev ( freeglut-devel )
			libx11-dev ( libX11-devel )
		5- cd ../demos
		6- make opt

	Windows:
		1- Install Microsoft Visual C++ 2010 Express Edition.
		2- File->Open->Project/Solution
		3- Open waffles\src\waffles.sln
		4- Change to Release mode.
		5- Build (F7).
		6- Set the startup app to the app you want to try.
		7- Set any relevant arguments in Project->Properties->
		   Debugging->Command Arguments
		8- Run it (F5).

		If you also want to try the demo apps:
		9- Open waffles\demos\demos.sln in Visual C++.
		10- Build (F7).
		11- Set the startup app to the app you want to try.
		12- Run it (F5).

	OSX:
		1- Install Fink or some other UNIX package manager.
		2- Install the following packages.
			g++
			make
			libpng3-dev
		3- cd waffles/src
		4- make opt

		If you also want to build the demo apps:
		5- Install the following dependency packages:
			libsdl1.2-dev
			freeglut3-dev
		6- cd ../demos
		7- make opt

For more detailed instructions, troubleshooting help, an overview of this
toolkit, and instructions for using it, see web/docs.html.

