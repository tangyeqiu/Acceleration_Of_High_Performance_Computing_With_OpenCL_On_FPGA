#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <cassert>
#include <map>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device;
cl_context context = NULL;
scoped_array<cl_command_queue> queue;
cl_program program = NULL;

bool init_opencl();
size_t GetPaddedRowSize(size_t input_row_size);
void ZeroRow(cl_mem row, size_t row_size, cl_kernel kernel, cl_command_queue command_queue);

scoped_array<cl_kernel> f_mat_and_h_hat_mat_row_kernel; 
scoped_array<cl_kernel> upsweep_kernel;
scoped_array<cl_kernel> downsweep_kernel;
scoped_array<cl_kernel> h_mat_row_kernel;
scoped_array<cl_kernel> zero_kernel;

template <class T>
class Matrix {
public:
    Matrix(size_t num_rows, size_t num_cols, const T & value) : num_rows_(num_rows), num_cols_(num_cols) {
        mat_ = new T*[num_rows];
        for (size_t r = 0; r < num_rows; ++r) {
            mat_[r] = new T[num_cols];
            for (size_t c = 0; c < num_cols; ++c) {
                mat_[r][c] = value;
            }
        }
    }

    ~Matrix() {
        for (size_t r = 0; r < num_rows_; ++r) {
            delete [] mat_[r];
        }
        delete [] mat_;
    }

    T* operator[] (size_t row_index) { return mat_[row_index]; }

    size_t GetNumRows() { return num_rows_; }
    size_t GetNumCols() { return num_cols_; }

private:
    T** mat_;

    size_t num_rows_;
    size_t num_cols_;
    
    std::vector<T> vec_;
};


// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

// Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

    //using DataType = int32_t;

    int32_t match = 5;
    int32_t mismatch = -3;
    int32_t gap_start_penalty = -8;
    int32_t gap_extend_penalty = -1;

    std::string seq1 = "CAGCCTCGCTTAG";
    std::string seq2 = "AATGCCATTGCCGG";

    std::cout << "seq1.size(): " << seq1.size() << std::endl;
    std::cout << "seq2.size(): " << seq2.size() << std::endl;

    Matrix<int32_t> h_mat(seq2.size() + 1, seq1.size() + 1, 0);

    const size_t row_size = seq1.size() + 1;
    const size_t padded_row_size = GetPaddedRowSize(row_size);

    std::cout << "Padded row size: " << padded_row_size << std::endl;

    /*auto pow_of_2 = [](const size_t pow)
    {
        return 1 << pow;
    };


    auto log2 = [](size_t num) {
        size_t log = 0;

        while (num != 0) {
            num = num >> 1;
            ++log;
        }

        return log-1;
    };*/

  cl_int status=CL_SUCCESS;

    cl_mem f_mat_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem f_mat_prev_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem h_mat_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem h_mat_prev_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem h_hat_mat_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem padded_row_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t) * padded_row_size, NULL, &status); // This also doubles as e_mat row
    checkError(status);

    cl_mem a_subs_score_row_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem c_subs_score_row_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem g_subs_score_row_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    cl_mem t_subs_score_row_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int32_t) * row_size, NULL, &status);
    checkError(status);

    ZeroRow(f_mat_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(f_mat_prev_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(h_mat_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(h_mat_prev_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(h_hat_mat_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(padded_row_buffer, padded_row_size, zero_kernel, queue);
    ZeroRow(a_subs_score_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(c_subs_score_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(g_subs_score_row_buffer, row_size, zero_kernel, queue);
    ZeroRow(t_subs_score_row_buffer, row_size, zero_kernel, queue);

    clFinish(queue);

std::map<char, cl_mem> query_character_row_score_map;
    {
        // A
        std::vector<int32_t> a_vec(row_size, 0);
        for (int c = 1; c < row_size; ++c) {
            a_vec[c] = seq1[c-1] == 'A' ? match : mismatch;
        }

        clEnqueueWriteBuffer(queue, a_subs_score_row_buffer, CL_FALSE, 0, sizeof(int32_t) * row_size, a_vec.data(), 0, nullptr, nullptr);

        // C
        std::vector<int32_t> c_vec(row_size, 0);
        for (int c = 1; c < row_size; ++c) {
            c_vec[c] = seq1[c-1] == 'C' ? match : mismatch;
        }

        clEnqueueWriteBuffer(queue, c_subs_score_row_buffer, CL_FALSE, 0, sizeof(int32_t) * row_size, c_vec.data(), 0, nullptr, nullptr);

        // G
        std::vector<int32_t> g_vec(row_size, 0);
        for (int c = 1; c < row_size; ++c) {
            g_vec[c] = seq1[c-1] == 'G' ? match : mismatch;
        }

        clEnqueueWriteBuffer(queue, g_subs_score_row_buffer, CL_FALSE, 0, sizeof(int32_t) * row_size, g_vec.data(), 0, nullptr, nullptr);

        // T
        std::vector<int32_t> t_vec(row_size, 0);
        for (int c = 1; c < row_size; ++c) {
            t_vec[c] = seq1[c-1] == 'T' ? match : mismatch;
        }

        clEnqueueWriteBuffer(queue, t_subs_score_row_buffer, CL_FALSE, 0, sizeof(int32_t) * row_size, t_vec.data(), 0, nullptr, nullptr);

        query_character_row_score_map.insert('A', a_subs_score_row_buffer);
        query_character_row_score_map.insert('C', c_subs_score_row_buffer);
        query_character_row_score_map.insert('G', g_subs_score_row_buffer);
        query_character_row_score_map.insert('T', t_subs_score_row_buffer);

        clFinish(queue);
    }

//timer start

    for (size_t r = 1; r < h_mat.GetNumRows(); ++r) {
        cl_event f_mat_and_h_hat_mat_finished;
        // Calculate f_mat_row
        {
            status = 0;
            status = clSetKernelArg(f_mat_and_h_hat_mat_row_kernel, 0, sizeof(cl_mem), &f_mat_prev_row_buffer);
            status |= clSetKernelArg(f_mat_and_h_hat_mat_row_kernel, 1, sizeof(cl_mem), &h_mat_prev_row_buffer);
            status |= clSetKernelArg(f_mat_and_h_hat_mat_row_kernel, 2, sizeof(cl_mem), &f_mat_row_buffer);
            status |= clSetKernelArg(f_mat_and_h_hat_mat_row_kernel, 3, sizeof(cl_mem), &(query_character_row_score_map[seq2[r-1]]));
            status |= clSetKernelArg(f_mat_and_h_hat_mat_row_kernel, 4, sizeof(cl_mem), &h_hat_mat_row_buffer);

            checkError(status);

            size_t global = row_size;
            status = clEnqueueNDRangeKernel(queue, f_mat_and_h_hat_mat_row_kernel, 1, NULL, &global, nullptr, 0, nullptr, &f_mat_and_h_hat_mat_finished);
            checkError(status);
        }

        cl_event padded_row_buffer_load_finished;
        {
            status = clEnqueueCopyBuffer(queue, h_hat_mat_row_buffer, padded_row_buffer, 0, 0, row_size * sizeof(int32_t), 1, &f_mat_and_h_hat_mat_finished, &padded_row_buffer_load_finished);
            checkError(status);
            clReleaseEvent(f_mat_and_h_hat_mat_finished);
        }

        cl_event upsweep_finished;
        {
            // Upsweep
            std::vector<cl_event> upsweep_row_finished(log2(padded_row_size) - 1);
            for (size_t depth = 0; depth < log2(padded_row_size); ++depth) {
                status = 0;
                status = clSetKernelArg(upsweep_kernel, 0, sizeof(cl_mem), &padded_row_buffer);
                status |= clSetKernelArg(upsweep_kernel, 1, sizeof(cl_int), &depth);
                checkError(status);

                size_t global = padded_row_size / pow_of_2(depth+1);
                if (depth == log2(padded_row_size) - 1) {
                    // Last iteration
                    status = clEnqueueNDRangeKernel(queue, upsweep_kernel, 1, NULL, &global, nullptr, 1, &upsweep_row_finished[depth-1], &upsweep_finished);
                } else if (depth == 0) {
                    // First iteration
                    status = clEnqueueNDRangeKernel(queue, upsweep_kernel, 1, NULL, &global, nullptr, 1, &padded_row_buffer_load_finished, &upsweep_row_finished[depth]);
                    clReleaseEvent(padded_row_buffer_load_finished);
                } else {
                    status = clEnqueueNDRangeKernel(queue, upsweep_kernel, 1, NULL, &global, nullptr, 1, &upsweep_row_finished[depth-1], &upsweep_row_finished[depth]);
                }
                checkError(status);
            }

            for (auto & event : upsweep_row_finished) {
                clReleaseEvent(event);
            }
        }

        cl_event downsweep_initialization_finished;
        int32_t zero = 0;
        {
            status = clEnqueueWriteBuffer(queue, padded_row_buffer, CL_FALSE, (padded_row_size-1) * sizeof(DataType), sizeof(DataType), &zero, 1, &upsweep_finished, &downsweep_initialization_finished);
            checkError(status);
            clReleaseEvent(upsweep_finished);
        }

        cl_event downsweep_finished;
        {
        // Downsweep
        std::vector<cl_event> downsweep_row_finished(log2(padded_row_size) - 1);

            for (int64_t depth = log2(padded_row_size) - 1; depth >= 0; --depth) {
                status = 0;
                status = clSetKernelArg(downsweep_kernel, 0, sizeof(cl_mem), &padded_row_buffer);
                status |= clSetKernelArg(downsweep_kernel, 1, sizeof(cl_int), &depth);
                checkError(status);

                size_t global = padded_row_size / pow_of_2(depth+1);

                if (depth == log2(padded_row_size) - 1) {
                    // First iteration
                    status = clEnqueueNDRangeKernel(queue, downsweep_kernel, 1, NULL, &global, nullptr, 1, &downsweep_initialization_finished, &downsweep_row_finished[depth-1]);
                    clReleaseEvent(downsweep_initialization_finished);
                } else if (depth == 0) {
                    // Last iteration
                    status = clEnqueueNDRangeKernel(queue, downsweep_kernel, 1, NULL, &global, nullptr, 1, &downsweep_row_finished[depth], &downsweep_finished);
                } else {
                    status = clEnqueueNDRangeKernel(queue, downsweep_kernel, 1, NULL, &global, nullptr, 1, &downsweep_row_finished[depth], &downsweep_row_finished[depth-1]);
                }

                checkError(status);
            }

            for (auto & event : downsweep_row_finished) {
                clReleaseEvent(event);
            }
        }

        // Calculate h_mat_row
        cl_event h_mat_finished;
        {
            status = 0;
            status = clSetKernelArg(h_mat_row_kernel, 0, sizeof(cl_mem), &h_hat_mat_row_buffer);
            status |= clSetKernelArg(h_mat_row_kernel, 1, sizeof(cl_mem), &padded_row_buffer);
            status |= clSetKernelArg(h_mat_row_kernel, 2, sizeof(cl_mem), &h_mat_row_buffer);

            checkError(status);

            size_t global = row_size;
            status = clEnqueueNDRangeKernel(queue, h_mat_row_kernel, 1, NULL, &global, nullptr, 1, &downsweep_finished, &h_mat_finished);
            checkError(status);
            clReleaseEvent(downsweep_finished);
        }

//        // Copy to host
//        {
//            status = clEnqueueReadBuffer(queue, h_mat_row_buffer, CL_FALSE, 0, sizeof(DataType) * row_size, h_mat[r], 1, &h_mat_finished, nullptr);
//            checkError(status);
//            clReleaseEvent(h_mat_finished);
//        }

        {
            clFinish(queue);
        }

        std::swap(f_mat_row_buffer, f_mat_prev_row_buffer);
        std::swap(h_mat_row_buffer, h_mat_prev_row_buffer);
    }

//timer stop
   // auto SW_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

   // std::cout << "SW took: " << SW_time_milliseconds << " ms" << std::endl;
   // std::cout << "Estimated time to search entire genome: " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * (3000000000 / seq1.size()) / 1000000000.0 << " s" << std::endl;

//    for (int r = 0; r < h_mat.GetNumRows(); ++r) {
//        for (int c = 0; c < h_mat.GetNumCols(); ++c) {
//            std::cout << h_mat[r][c] << "\t";
//        }
//        std::cout << "\n";
//    }
//    std::cout << std::endl;

    clReleaseMemObject(f_mat_row_buffer);
    clReleaseMemObject(f_mat_prev_row_buffer);
    clReleaseMemObject(h_mat_row_buffer);
    clReleaseMemObject(h_mat_prev_row_buffer);
    clReleaseMemObject(h_hat_mat_row_buffer);
    clReleaseMemObject(padded_row_buffer);
    clReleaseMemObject(a_subs_score_row_buffer);
    clReleaseMemObject(c_subs_score_row_buffer);
    clReleaseMemObject(g_subs_score_row_buffer);
    clReleaseMemObject(t_subs_score_row_buffer);


    clReleaseKernel(f_mat_and_h_hat_mat_row_kernel);
    //clReleaseKernel(h_hat_mat_row_kernel);
    clReleaseKernel(upsweep_kernel);
    clReleaseKernel(downsweep_kernel);
    clReleaseKernel(h_mat_row_kernel);
    clReleaseKernel(zero_kernel);

    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext (context);


}

// Initializes the Common OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("SW_kernels", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");


    // Command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    f_mat_and_h_hat_mat_row_kernel = clCreateKernel(program, "f_mat_and_h_hat_mat_row_kernel", &status);
    checkError(status, "Failed to create f_mat_and_h_hat_mat_row_kernel");

    upsweep_kernel = clCreateKernel(program, "upsweep", &status);
    checkError(status, "Failed to create upsweep_kernel");

    downsweep_kernel = clCreateKernel(program, "downsweep", &status);
    checkError(status, "Failed to create downsweep_kernel");

    h_mat_row_kernel = clCreateKernel(program, "h_mat_row_kernel", &status);
    checkError(status, "Failed to create h_mat_row_kernel");

    zero_kernel = clCreateKernel(program, "zero", &status);
    checkError(status, "Failed to create zero_kernel");



  return true;
}

void ZeroRow(cl_mem row, size_t row_size, cl_kernel kernel, cl_command_queue command_queue) {
    cl_int error = CL_SUCCESS;
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &row);
    checkError(error);

    size_t global = row_size;
    error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, nullptr, 0, nullptr, nullptr);
    checkError(error);
}

size_t GetPaddedRowSize(size_t input_row_size) {
    --input_row_size;
    input_row_size |= input_row_size >> 1;
    input_row_size |= input_row_size >> 2;
    input_row_size |= input_row_size >> 4;
    input_row_size |= input_row_size >> 8;
    input_row_size |= input_row_size >> 16;
    input_row_size |= input_row_size >> 32;
    ++input_row_size;
    // std::cout << "Row size rounded up to next power of 2: " << padded_row_size << std::endl;

    return input_row_size;
}
