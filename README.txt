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

		To build unoptimized binaries with debug symbols:
			make dbg

		To build demo apps and additional tools:
			see web/docs/linux.html

	OSX:
		1- Install clang or g++. Also install Gnu Make.
		   a- Install Xcode
		   b- Click Xcode->Preferences->Downloads
		   c- Click the down arrow to install the "Command Line Tools".
		   (If you do not want to install Xcode, there are other ways
		    to install g++, such as with Fink or MacPorts.)
		2- Open a terminal
		3- cd waffles/src
		4- make opt

		To build demo apps and additional tools:
			see web/docs/mac.html

	Windows:
		1- Install Microsoft Visual C++ 2013 Express Edition.
		2- File->Open->Project/Solution
		3- Open waffles\src\waffles.sln
		4- Change to Release mode.
		5- Build (F7).
		6- Set the startup app to the app you want to try.
		7- Set any relevant arguments in Project->Properties->
		   Debugging->Command Arguments
		8- Run it (F5).

		If you also want to try the demo apps:
			see web/docs/windows.html.

For more detailed instructions, troubleshooting help, an overview of this
toolkit, and instructions for using it, see web/index.html.

