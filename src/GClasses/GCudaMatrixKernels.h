#ifndef GCUDAMATRIXKERNELS_H
#define GCUDAMATRIXKERNELS_H


namespace GClasses {

class GCudaEngine;
class GCudaVector;

void Conv2D_feedForward(GCudaEngine& engine, GCudaVector& activation, GCudaVector& net, GCudaVector& incoming, const GCudaMatrix& kernels, const GCudaVector& bias, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputRows, size_t inputCols, size_t inputChannels, size_t padding, size_t stride);

void Conv2D_deactivate(GCudaEngine& engine, GCudaVector& error, const GCudaVector& net, const GCudaVector& activation, size_t outputs);

void Conv2D_backPropError(GCudaEngine& engine, GCudaVector& upStreamError, const GCudaVector& err, const GCudaMatrix& kernels, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride);

void Conv2D_updateDeltas(GCudaEngine& engine, GCudaMatrix& delta, GCudaVector& biasDelta, const GCudaVector& upStreamActivation, const GCudaVector& err, double momentum, size_t kernelCount, size_t kernelRows, size_t kernelCols, size_t outputRows, size_t outputCols, size_t inputChannels, size_t inputRows, size_t inputCols, size_t padding, size_t stride);

}

#endif // GCUDAMATRIXKERNELS_H

