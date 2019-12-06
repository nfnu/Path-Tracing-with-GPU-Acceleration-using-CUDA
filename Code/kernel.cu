#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include <stdlib.h>
#include "moving_sphere.h"
#include "box.h"
#include <fstream>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	vec3 cur_color = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_attenuation = emitted + cur_attenuation;
				cur_ray = scattered;
			}
			else {
				//cur_attenuation *= attenuation;
				//cur_attenuation = emitted + cur_attenuation;
				return cur_attenuation;
			}
		}
		else {
			return vec3(0.0, 0.0, 0.0);
			return cur_attenuation * vec3(0.005, 0.005, 0.005);
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	float f = 0.3; //Gamma correction
	col[0] = 1.85 * sqrt(col[0]) / f;
	col[1] = 1.85 * sqrt(col[1]) / f;
	col[2] = 1.85 * sqrt(col[2]) / f;
	if (col[0] > 1) {
		col[0] = 1;
	}
	if (col[1] > 1) {
		col[1] = 1;
	}
	if (col[2] > 1) {
		col[2] = 1;
	}

	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__device__ void scene_1(hitable **d_list, hitable **d_world, curandState *rand_state) {
	texture_val *checker = new checker_texture(
		new constant_texture(vec3(0.2, 0.3, 0.1)),
		new constant_texture(vec3(0.9, 0.9, 0.9))
	);
	curandState local_rand_state = *rand_state;
	*rand_state = local_rand_state;
	d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(checker));
	int i = 1;
	for (int a = -10; a < 10; a++) {
		for (int b = -10; b < 10; b++) {
			float choose_mat = RND;
			vec3 center(a + 0.9*RND, 0.2, b + 0.9*RND);
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8f) {
					d_list[i++] = new moving_sphere(
						center,
						center + vec3(0, 0.5*RND, 0),
						0.0, 1.0, 0.2,
						new lambertian(new constant_texture(
							vec3(RND*RND,
								RND*RND,
								RND*RND))));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}
	d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
	d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
	*d_world = new hitable_list(d_list, i);
}
__device__ void scene_2(hitable **d_list, hitable **d_world, curandState *rand_state) {
	texture_val *checker = new checker_texture(new constant_texture(vec3(0.2, 0.3, 0.1)), new constant_texture(vec3(0.9, 0.9, 0.9)));
	curandState local_rand_state = *rand_state;
	*rand_state = local_rand_state;
	d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(checker));
	d_list[1] = new sphere(vec3(0, 2, 0), 2, new lambertian(checker));
	d_list[2] = new sphere(vec3(0, 7, 0), 2,
		new diffuse_light_val(new constant_texture(vec3(4, 4, 4))));
	d_list[2] = new xy_rect(3, 5, 1, 3, -2,
		new diffuse_light_val(new constant_texture(vec3(4, 4, 4))));
	*d_world = new hitable_list(d_list, 3);
}

__device__ void scene_3(hitable **d_list, hitable **d_world, curandState *rand_state) {
	int i = 0;
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light_val(new constant_texture(vec3(15, 15, 15)));
	d_list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
	d_list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	d_list[i++] = new xz_rect(213, 343, 227, 332, 554, light);
	d_list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
	d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	d_list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
	d_list[i++] = new translate(
		new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18),
		vec3(130, 0, 65)
	);
	d_list[i++] = new translate(
		new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white), 15),
		vec3(265, 0, 295)
	);
	*d_world = new hitable_list(d_list, i);
}

__device__ void scene_4(hitable **d_list, hitable **d_world, curandState *rand_state) {
	int i = 0;
	d_list[i++] = new sphere(vec3(0, 0, -1), 0.5,
		new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5))));
	d_list[i++] = new sphere(vec3(0, -100.5, -1), 100,
		new lambertian(new constant_texture(vec3(0.8, 0.8, 0.0))));
	d_list[i++] = new sphere(vec3(1, 0, -1), 0.5,
		new metal(vec3(0.8, 0.6, 0.2), 0.0));
	d_list[i++] = new sphere(vec3(-1, 0, -1), 0.5,
		new dielectric(1.5));
	material *light = new diffuse_light_val(new constant_texture(vec3(15, 15, 15)));
	d_list[i++] = new xy_rect(-1, 1, 1, 3, -2,
		new diffuse_light_val(new constant_texture(vec3(4, 4, 4))));
	*d_world = new hitable_list(d_list, i);
}

__device__ void scene_5(hitable **d_list, hitable **d_world, curandState *rand_state) {
	int i = 0;
	curandState local_rand_state = *rand_state;
	*rand_state = local_rand_state;
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *white_metal = new metal(vec3(0.73, 0.73, 0.73), 0.0);
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light_val(new constant_texture(vec3(15, 15, 15)));
	d_list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
	d_list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	d_list[i++] = new xz_rect(213, 343, 227, 332, 554, light);
	d_list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
	d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	d_list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
	d_list[i++] = new sphere(vec3(280, 80, 170), 85,
		new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5))));
	d_list[i++] = new sphere(vec3(120, 85, 338), 85,
		new dielectric(1.5));
	d_list[i++] = new translate(
		new rotate_y(new box(vec3(0, 0, 0), vec3(165, 300, 165), white_metal), 15),
		vec3(265, 0, 295));
	*d_world = new hitable_list(d_list, i);
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		*rand_state = local_rand_state;
		//scene_1(d_list, d_world, rand_state);
		//scene_2(d_list, d_world, rand_state);
		scene_3(d_list, d_world, rand_state);
		//scene_4(d_list, d_world, rand_state);
		//scene_5(d_list, d_world, rand_state);
		//For scene1 and scene2
		/*
		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.0;
		float vfov = 50;
		*/
		//For scene3, scene4 and scene5
		
		vec3 lookfrom(278, 278, -800);
		vec3 lookat(278, 278, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.0;
		float vfov = 40;
		
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			vfov,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,1.0);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

int main() {
	int nx = 800;
	int ny = 800;
	int ns = 1000;
	int tx = 8;
	int ty = 8;
	std::ofstream myfile;
	myfile.open("output.ppm");
	std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// allocate random state
	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// make our world of hitables & the camera
	hitable **d_list;
	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables * sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	start = clock();
	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	// Output FB as Image
	myfile << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			int ir = int(255.99*fb[pixel_index].r());
			int ig = int(255.99*fb[pixel_index].g());
			int ib = int(255.99*fb[pixel_index].b());
			myfile << ir << " " << ig << " " << ib << "\n";
		}
	}
	myfile.close();
	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}