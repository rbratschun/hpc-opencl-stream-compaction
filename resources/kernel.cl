int upsweep(__local int * temp, int offset, int n)
{
    int lid = get_local_id(0);
    for (int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d) {
            int ai = offset * (2 * lid + 1) - 1;
            int bi = offset * (2 * lid + 2) - 1;

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    return offset;
}

void downsweep(__local int * temp, int offset, int n)
{
    int lid = get_local_id(0);
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        if (lid < d) {
            int ai = offset * (2 * lid + 1) - 1;
            int bi = offset * (2 * lid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
}

__kernel void prefixsum(__global const int* input,
                            __global int* output,
                            __local int * temp,
                            __global int * blocksums,
                            const int n)
{
    // WORKSPACE
    int local_size = get_local_size(0);
    int global_size = get_global_size(0);
    
    // ORIENTATION
    
    int group_id= get_group_id(0);
    int thread_id = get_local_id(0);
    
    int group_offset = group_id * local_size;
   
   // printf("group_id: %d", group_id);

    // STORE VALUES IN LOCAL MEMORY^
    temp[thread_id] = input[group_offset + thread_id];
    temp[thread_id + 1] = input[group_offset + thread_id + 1];

    // UPSWEEP (=reduce phase)
    int offset = upsweep(temp, 1, local_size);
    // printf("offset after reduce: %d", offset);
    
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (thread_id == 0) {
        // store last value of block in blocksums array!
        blocksums[group_id] = temp[local_size - 1];
        // exclusive scan ()=> last thread sets last index to zero
        temp[local_size - 1] = 0;
    }

    // DOWNSWEEP PHASE
    downsweep(temp, offset, local_size);
    
    // der letzte dreht das licht ab
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    output[group_offset + thread_id] = temp[thread_id];
    output[group_offset + thread_id + 1] = temp[thread_id + 1];
}

__kernel void addBlockSums(__global int * output, __global int * blockSumsScanned) {
    int globalID = get_global_id(0);
    int groupID = get_group_id(0);
    output[globalID] = output[globalID] + blockSumsScanned[groupID];
}

__kernel
void filterLess(__global const int * restrict input, __global int * output, const int comparable) {
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int offset = group_id * local_size;
    const int local_id = get_local_id(0);
    const int pos = offset + local_id;
    output[pos] = input[pos] < comparable ? 1 : 0;
}

__kernel
void filterEqual(__global const int * restrict input, __global int * output, const int comparable) {
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int offset = group_id * local_size;
    const int local_id = get_local_id(0);
    const int pos = offset + local_id;
    output[pos] = input[pos] == comparable ? 1 : 0;
}

__kernel
void filterGreater(__global const int * restrict input, __global int * output, const int comparable) {
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int offset = group_id * local_size;
    const int local_id = get_local_id(0);
    const int pos = offset + local_id;
    output[pos] = input[pos] > comparable ? 1 : 0;
}

__kernel 
void scatter(__global const int * restrict input, __global const int * restrict addresses, __global int * output) {
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int offset = group_id * local_size;
    const int local_id = get_local_id(0);
    const int pos = offset + local_id;
    if(temp[pos] == 1) {
        output[addresses[pos]] = input[pos];
    }
}
