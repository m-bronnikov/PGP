#include <stdlib.h>
#include <stdio.h> 
#include <math.h>

typedef unsigned char uchar;

struct uchar4 {
	uchar x;
	uchar y;
	uchar z;
	uchar w;
};

struct vec3 {
	double x;
	double y;
	double z;
};

double dot(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 prod(vec3 a, vec3 b) {
	return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

vec3 norm(vec3 v) {
	double l = sqrt(dot(v, v));
	return {v.x / l, v.y / l, v.z / l};
}

vec3 diff(vec3 a, vec3 b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

vec3 add(vec3 a, vec3 b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
	return {a.x * v.x + b.x * v.y + c.x * v.z,
				a.y * v.x + b.y * v.y + c.y * v.z,
				a.z * v.x + b.z * v.y + c.z * v.z};
}

void print(vec3 v) {
	printf("%e %e %e\n", v.x, v.y, v.z);
}

struct trig {
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
};

trig trigs[6];

void build_space() {
	trigs[0] = {{-5, -5, 0}, {5, -5, 0}, {-5, 5, 0}, {0, 0, 255, 0}};
	trigs[1] = {{5, 5, 0}, {5, -5, 0}, {-5, 5, 0}, {0, 0, 255, 0}};
	trigs[2] = {{-2,-2, 4}, {2, -2, 4}, {0, 2, 4}, {128, 0, 128, 0}};
	trigs[3] = {{-2, -2, 4}, {2, -2, 4}, {0, 0, 7}, {255, 0, 0, 0}};
	trigs[4] = {{-2,-2, 4}, {0, 0, 7}, {0, 2, 4}, {255, 255, 0, 0}};
	trigs[5] = {{0, 0, 7}, {2, -2, 4}, {0, 2, 4}, {0, 255, 0, 0}};
/*	int i;
	for(i = 0; i < 6; i++) {
		print(trigs[i].a);
		print(trigs[i].b);
		print(trigs[i].c);
		print(trigs[i].a);
		printf("\n\n\n");
	}
	printf("\n\n\n"); */
}

uchar4 ray(vec3 pos, vec3 dir) {
	int k, k_min = -1;
	double ts_min;
	for(k = 0; k < 6; k++) {
		vec3 e1 = diff(trigs[k].b, trigs[k].a);
		vec3 e2 = diff(trigs[k].c, trigs[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		vec3 t = diff(pos, trigs[k].a);
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		double v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		double ts = dot(q, e2) / div; 	
		if (ts < 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
		}
	}
	if (k_min == -1)
		return {0, 0, 0, 0};

	return trigs[k_min].color;
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data) {
	int i, j;
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
	vec3 by = norm(prod(bx, bz));
	for(i = 0; i < w; i++)	
		for(j = 0; j < h; j++) {
			vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
			vec3 dir = mult(bx, by, bz, v);
			data[(h - 1 - j) * w + i] = ray(pc, norm(dir));
	//		print(pc);
	//		print(add(pc, dir));
	//		printf("\n\n\n");
		}
	//print(pc);
	//print(pv);
	//printf("\n\n\n");
}

int main() {
	int k, w = 640, h = 480;
	char buff[256];
	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	vec3 pc, pv;

	build_space();

	for(k = 0; k < 126; k++) { 
		pc = (vec3) {6.0 * sin(0.05 * k), 6.0 * cos(0.05 * k), 5.0 + 2.0 * sin(0.1 * k)};
		pv = (vec3) {3.0 * sin(0.05 * k + M_PI), 3.0 * cos(0.05 * k + M_PI), 0.0};
		render(pc, pv, w, h, 120.0, data);

		sprintf(buff, "res/%d.data", k);
		printf("%d: %s\n", k, buff);		

		FILE *out = fopen(buff, "wb");
		fwrite(&w, sizeof(int), 1, out);
		fwrite(&h, sizeof(int), 1, out);	
		fwrite(data, sizeof(uchar4), w * h, out);
		fclose(out);
	}
	free(data);	
	return 0;
}
