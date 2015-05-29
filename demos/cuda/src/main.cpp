/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include <exception>
#include <iostream>
#include <GClasses/GApp.h>
#include <GClasses/GError.h>
#include <GClasses/GVec.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GTime.h>
#include <depends/GCuda/GCudaMatrix.h>
#include <depends/GCuda/GCudaLayers.h>

using namespace GClasses;
using std::cerr;
using std::cout;

void test_GCudaMatrix(GCudaEngine& e)
{
	GVec v1(2);
	v1.v[0] = 1.0;
	v1.v[1] = 2.0;
	GMatrix m(2, 3);
	m[0][0] = 1.0; m[0][1] = 2.0; m[0][2] = 3.0;
	m[1][0] = 4.0; m[1][1] = 5.0; m[1][2] = 6.0;

	GCudaVector cv1;
	cv1.upload(v1.v, 2);
	GCudaMatrix cm;
	cm.upload(m);
	GCudaVector cv2;
	cm.rowVectorTimesThis(e, cv1, cv2);

	cout << "Testing Matrix-vector multiplication...\n";
	GVec v2(3);
	cv2.download(v2.v);
	std::cout << "Expected: 9, 12, 15\n";
	std::cout << "  Actual: " << to_str(v2.v[0]) << ", " << to_str(v2.v[1]) << ", " << to_str(v2.v[2]) << "\n";

	cout << "Testing the tanh activation kernel...\n";
	v2.v[0] = 1.0; v2.v[1] = 2.0; v2.v[2] = 3.0;
	cv2.upload(v2.v, 3);
	cv2.activateTanh(e);
	cv2.download(v2.v);
	std::cout << "Expected: 0.76159415595576, 0.96402758007582, 0.99505475368673\n";
	std::cout << "  Actual: " << to_str(v2.v[0]) << ", " << to_str(v2.v[1]) << ", " << to_str(v2.v[2]) << "\n";
}

void test_GCudaLayer(GCudaEngine& e)
{
	GUniformRelation rel(3);
	size_t width = 1000;
	size_t epochs = 2000;

	cout << "Making a classic neural net...\n";
	GNeuralNet nn1;
	nn1.addLayer(new GLayerClassic(3, width));
	nn1.addLayer(new GLayerClassic(width, width));
	nn1.addLayer(new GLayerClassic(width, width));
	nn1.addLayer(new GLayerClassic(width, 3));
	nn1.beginIncrementalLearning(rel, rel);

	cout << "Making a parallel neural net...\n";
	GNeuralNet nn2;
	nn2.addLayer(new GLayerClassicCuda(e, 3, width));
	nn2.addLayer(new GLayerClassicCuda(e, width, width));
	nn2.addLayer(new GLayerClassicCuda(e, width, width));
	nn2.addLayer(new GLayerClassicCuda(e, width, 3));
	nn2.beginIncrementalLearning(rel, rel);

	cout << "Making another parallel neural net...\n";
	GNeuralNet nn3;
	nn3.addLayer(new GLayerClassicCuda(e, 3, width));
	nn3.addLayer(new GLayerClassicCuda(e, width, width));
	nn3.addLayer(new GLayerClassicCuda(e, width, width));
	nn3.addLayer(new GLayerClassicCuda(e, width, 3));
	nn3.beginIncrementalLearning(rel, rel);

	cout << "Copying (so they will have identical weights)...\n";
	for(size_t i = 0; i < nn1.layerCount(); i++)
		((GLayerClassicCuda*)&nn2.layer(i))->upload(*(GLayerClassic*)&nn1.layer(i));
	for(size_t i = 0; i < nn1.layerCount(); i++)
		((GLayerClassicCuda*)&nn3.layer(i))->upload(*(GLayerClassic*)&nn1.layer(i));

	cout << "Testing to make sure they make identical predictions (before training)...\n";
	double vec[3];
	vec[0] = 0.2;
	vec[1] = 0.4;
	vec[2] = 0.6;
	double out[3];
	cout << "  Input: " << to_str(vec[0]) << ",	" << to_str(vec[1]) << ",	" << to_str(vec[2]) << "\n";
	nn1.predict(vec, out);
	cout << "Classic: " << to_str(out[0]) << ",	" << to_str(out[1]) << ",	" << to_str(out[2]) << "\n";
	nn2.predict(vec, out);
	cout << "  Cuda1: " << to_str(out[0]) << ",	" << to_str(out[1]) << ",	" << to_str(out[2]) << "\n";
	nn3.predict(vec, out);
	cout << "  Cuda2: " << to_str(out[0]) << ",	" << to_str(out[1]) << ",	" << to_str(out[2]) << "\n";


	cout << "Training the classic network...\n";
	GRand r(0);
	double timeBef1 = GTime::seconds();
	for(size_t i = 0; i < epochs; i++)
	{
		vec[0] = 0.2 * r.normal();
		vec[1] = 0.2 * r.normal();
		vec[1] = 0.2 * r.normal();
		nn1.trainIncremental(vec, vec);
	}
	double timeAft1 = GTime::seconds();


	cout << "Training the first parallel network...\n";
	r.setSeed(0);
	double timeBef2 = GTime::seconds();
	for(size_t i = 0; i < epochs; i++)
	{
		vec[0] = 0.2 * r.normal();
		vec[1] = 0.2 * r.normal();
		vec[1] = 0.2 * r.normal();
		nn2.trainIncremental(vec, vec);
	}
	double timeAft2 = GTime::seconds();

	cout << "Training the second parallel network without synchronization...\n";
	e.setHogWild(true);
	r.setSeed(0);
	double timeBef3 = GTime::seconds();
	for(size_t i = 0; i < epochs; i++)
	{
		vec[0] = 0.2 * r.normal();
		vec[1] = 0.2 * r.normal();
		vec[1] = 0.2 * r.normal();
		nn3.trainIncremental(vec, vec);
	}
	double timeAft3 = GTime::seconds();

	cout << "Classic training time: " << to_str(timeAft1 - timeBef1) << " seconds\n";
	cout << "  Cuda1 training time: " << to_str(timeAft2 - timeBef2) << " seconds\n";
	cout << "  Cuda2 training time: " << to_str(timeAft3 - timeBef3) << " seconds\n";
	cout << "Speedup1: " << to_str((timeAft1 - timeBef1) / (timeAft2 - timeBef2)) << "\n";
	cout << "Speedup2: " << to_str((timeAft1 - timeBef1) / (timeAft3 - timeBef3)) << "\n";

	cout << "Testing to make sure both networks still make the same predictions...\n";
	vec[0] = 0.2;
	vec[1] = 0.4;
	vec[2] = 0.6;
	cout << "  Input: " << to_str(vec[0]) << ",	" << to_str(vec[1]) << ",	" << to_str(vec[2]) << "\n";
	nn1.predict(vec, out);
	cout << "Classic: " << to_str(out[0]) << ",	" << to_str(out[1]) << ",	" << to_str(out[2]) << "\n";
	nn2.predict(vec, out);
	cout << "  Cuda1: " << to_str(out[0]) << ",	" << to_str(out[1]) << ",	" << to_str(out[2]) << "\n";
	nn3.predict(vec, out);
	cout << "  Cuda2: " << to_str(out[0]) << ",	" << to_str(out[1]) << ",	" << to_str(out[2]) << "\n";

}

void doit()
{
	GCudaEngine e;
	test_GCudaMatrix(e);
	test_GCudaLayer(e);
}

int main(int argc, char *argv[])
{
#ifdef _DEBUG
	GApp::enableFloatingPointExceptions();
#endif
	int nRet = 0;
	try
	{
		doit();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

