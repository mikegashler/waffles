#include "Body.h"
#include "GlutStuff.h"
#include <GClasses/GError.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GKNN.h>
#include <GClasses/GRand.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GRand.h>
#include <GClasses/GImage.h>
#include "PolicyLearner.h"
#include <iostream>

using namespace GClasses;
using std::cerr;
using std::cout;

void DoIt(int argc,char* argv[])
{
	if(argc < 2)
	{
		cout << "Jumper is a demo of a robot that learns to jump. It uses\n";
		cout << "a neural network to represent a policy that tells it how\n";
		cout << "to move its joints based on the time, and the orientation\n";
		cout << "of its spine. It improves the policy over time by\n";
		cout << "using an evolutionary optimizer to adjust the weights, and\n";
		cout << "computes fitness by the height it obtains.\n\n";

		cout << "Usage:\n";
		cout << "	jumper				see this info\n";
		cout << "	jumper train			generate policies\n";
		cout << "	jumper [policy_file]		see it jump\n";
		return;
	}
	else if(_stricmp(argv[1], "train") == 0)
	{
		cout << "Training the jumper. This takes some time, so let it run\n";
		cout << "until you have several policy files. When you've got enough,\n";
		cout << "press ctrl-c to stop. Then, run\n\n";

		cout << "	jumper [policy_file]\n\n";
		
		cout << "to see it jump with one of the policies. The longer you let\n";
		cout << "it train, the better they should get. You might want to\n";
		cout << "open another shell, so you can let this one train while you\n";
		cout << "view results in the other one.\n";

		Train();

		cout << "Well, that's enough. I think I'll stop now.\n";
	}
	else if(_stricmp(argv[1], "manual") == 0)
	{
		GRand prng(0);
		RagdollDemo demoApp;
		glutmain(argc, argv, 800, 600, "Demo of a robot that learns to jump", &demoApp);
	}
	else
	{
#ifdef NOGUI
		cout << "This binary was compiled with NOGUI (which means it isn't\n";
		cout << "linked with the GL libraries. This feature might be useful\n";
		cout << "for someone that wants to train on a fast cluster that doesn't\n";
		cout << "have display stuff installed.) Obviously you can't view the\n";
		cout << "jumper without a GUI, so if you want to see it, you'll have\n";
		cout << "to rebuild without the NOGUI switch.\n";
#else
		cout << "Press space-bar to see it jump again. (Note that very small differences in when your computer decides to schedule various threads can affect the frame rate, which can affect the physics simulator, which can affect how the robot jumps, so you may want to try several times)\n";
		GRand prng(0);
		RagdollDemo demoApp;
		GNeuralNet* pNN = LoadPolicy(argv[1], &prng);
		Holder<GNeuralNet> hNN(pNN);
		demoApp.SetPolicy(pNN);
		glutmain(argc, argv, 800, 600, "Demo of a robot that learns to jump", &demoApp);
#endif
	}
}

int main(int argc,char* argv[])
{
	int ret = 0;
	try
	{
		DoIt(argc, argv);
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		ret = 1;
	}
	return ret;
}
