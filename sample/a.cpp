#include <iostream>
#include <tbb/concurrent_hash_map.h>
#include <tbb/scalable_allocator.h>
#include <gmp.h>

int main() {
    std::cout << "Hello World!\n";
    tbb::scalable_allocator<int *> * allocator = new tbb::scalable_allocator<int *>();
    unsigned long a = 4;
    unsigned long b = 8;
    unsigned long c = 16;
    mpn_nand_n(&a, &b, &c, 3);
    std::cout << a << " " << b << " " << c << std::endl;
    return 0;
}
