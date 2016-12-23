# Waffles
Waffles is a comprehensive machine learning and data mining library. It includes many tools and algorithms from neural networks to recommender systems.

# Build Instructions
You may be interested in a more stable but older build of Waffles, check out the release for Waffles v1.0.

## Linux and OS X

### To install

1. Install g++ and make (if they are not already installed)
    1. On Debian, Ubuntu, and derivatives:

            sudo apt-get install g++ make

    2. On Red Hat, Fedora, and derivatives:

            sudo yum install g++ make

    3. On OS X:

            xcode-select --install

2. cd src
3. sudo make install

### To uninstall

	sudo make uninstall

### To build optimized binaries without installing them

	make opt

### To build unoptimized binaries with debug symbols

	make dbg

To build demo apps and additional tools, see web/docs/linux.html.

## Windows

1.	Install Microsoft Visual C++ 2013 Express Edition.
2.	File->Open->Project/Solution
3.	Open waffles\src\waffles.sln
4.	Change to Release mode.
5.	Build (F7).
6.	Set the startup app to the app you want to try.
7.	Set any relevant arguments in Project->Properties->Debugging->Command Arguments
8.	Run it (F5).

If you also want to try the demo apps, see web/docs/windows.html.

***

For more detailed instructions, troubleshooting help, an overview of this toolkit, and instructions for using it, see web/index.html.
