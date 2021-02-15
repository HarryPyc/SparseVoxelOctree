#include "Morton.cuh"
// "Insert" two 0 bits after each of the 10 low bits of x
__host__ __device__ cuuint32_t Part1By2(cuuint32_t x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

// Inverse of Part1By2 - "delete" all bits not at positions divisible by 3
__host__ __device__ cuuint32_t Compact1By2(cuuint32_t x)
{
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

__host__ __device__ cuuint32_t EncodeMorton(cuuint32_t x, cuuint32_t y, cuuint32_t z) {
	return (Part1By2(z) << 2) + (Part1By2(y) << 1) + (Part1By2(x));
}
__host__ __device__ cuuint32_t DecodeMortonX(cuuint32_t code) {
	return Compact1By2(code);
}
__host__ __device__ cuuint32_t DecodeMortonY(cuuint32_t code) {
	return Compact1By2(code >> 1);
}
__host__ __device__ cuuint32_t DecodeMortonZ(cuuint32_t code) {
	return Compact1By2(code >> 2);
}
