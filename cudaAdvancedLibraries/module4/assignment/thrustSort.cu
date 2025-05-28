/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include "helper/helper_cuda.h"

#include <algorithm>
#include <time.h>
#include <limits.h>

__host__ std::tuple<thrust::device_vector<float>, thrust::device_vector<unsigned int>, thrust::host_vector<unsigned int>> 
                                                                          generateDeviceMemory(int numElements, int keybits) {
  
  // TODO Generate thrust host_vectors for keys (data type float) and values (data type unsigned int) of length numELements
  thrust::host_vector<float> h_keys(numElements);
  thrust::host_vector<unsigned int> h_values(numElements);
  thrust::host_vector<float> h_keysSorted(numElements); // result vector

  // TODO Initialize thrust's default_random_engine random number generator
  thrust::default_random_engine rng(clock());
  
  // TODO Fill keys vector with random floats using thrust::uniform_real_distribution
  thrust::uniform_real_distribution<float> u01(0, 1);
  for (unsigned int i = 0; i < static_cast<unsigned int>(numElements); i++) {
    h_keys[i] = u01(rng);
  }
  
  // TODO Fill values vector with random integers
  for (unsigned int j = 0; j < static_cast<unsigned int>(numElements); j++) {
    h_values[j] = static_cast<unsigned int>(u01(rng));
  }

  // Copy data onto the GPU in thrust device_vectors for keys(d_keys of type float) and values(d_values of type unsigned int)
  thrust::device_vector<float> d_keys = h_keys;
  thrust::device_vector<unsigned int> d_values = h_values;

  // at this point, we forget the data on the host (h_keys and h_values)
  return {d_keys, d_values, h_keysSorted};
}

__host__ float runSortIteration(int numElements, int keybits, cudaEvent_t& start_event, cudaEvent_t& stop_event) {

    auto tuple = generateDeviceMemory(numElements, keybits);
    auto d_keys = std::get<0>(tuple);
    auto d_values = std::get<1>(tuple);
    auto h_keysSorted = std::get<2>(tuple);

    checkCudaErrors(cudaEventRecord(start_event, 0));

    // TODO sort using the thrust sort function on d_keys and d_values
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    // TODO copy sorted keys and values from GPU to host memory
    thrust::copy(d_keys.begin(), d_keys.end(), h_keysSorted.begin());

    // TODO check that keys are sorted and then check values are sorted 
    //      if both are sorted correctly (use the thrust::is_sorted function) then set bTestResult = true and false otherwise
    bool bTestResult = thrust::is_sorted(h_keysSorted.begin(), h_keysSorted.end());

    if(bTestResult)
      printf("Iteration sorted data successfully.\n");
    else
      printf("Iteration failed in sorting data.\n");

    checkCudaErrors(cudaEventRecord(stop_event, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event));

    float time = 0;
    checkCudaErrors(cudaEventElapsedTime(&time, start_event, stop_event));
    return time;
}

__host__ void testRadixSort(int numElements, int keybits, int numIterations) {

  // run multiple iterations to compute an average sort time
  cudaEvent_t start_event, stop_event;
  checkCudaErrors(cudaEventCreate(&start_event));
  checkCudaErrors(cudaEventCreate(&stop_event));

  float totalTime = 0;

  for (unsigned int i = 0; i < numIterations; i++) {
      float time = runSortIteration(numElements, keybits, start_event, stop_event);
      totalTime += time;
  }

  totalTime /= (1.0e3f * numIterations);
  printf(
      "radixSortThrust, Throughput = %.4f MElements/s, Time = %.5f s, Size = "
      "%u elements\n",
      1.0e-6f * numElements / totalTime, totalTime, numElements);

  getLastCudaError("after radixsort");
  getLastCudaError("copying results to host memory");

  checkCudaErrors(cudaEventDestroy(start_event));
  checkCudaErrors(cudaEventDestroy(stop_event));
}

int main(int argc, char **argv) {
  // Start logs
  printf("%s Starting...\n\n", argv[0]);

  findCudaDevice(argc, (const char **)argv);

  int numElements = 1024;
  int keybits = 32;
  int numIterations = 10;

  testRadixSort(numElements, keybits, numIterations);

}