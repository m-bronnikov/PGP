**Project exist following files:**

1. main.cu - Main file in project. Reads the input data and creates scene with objects, camera and writer. (Not ready)

2. scene.cuh - Scene it class of scene. it contains all triangles, diods, lights of scene with camera and file writer. method render_scene launch recursive ray_tracing for all frames, which described by camera. (Not Ready)

3. camera.cuh - camera, which generate camera positions in earth coords for rendering frames. (Ready)

4. file_writer.cuh - class of writer. it writes image from GPU global mem to file, which describes by path modifier and num of frame. (Ready)

5. ray_cleaner.cuh - function for clean dead rays form recursion data of ray tracing. it use scan and radix binary sort. (Ready)

6. ray_tracing.cuh - Function provides main ray_tracing functionality for computing ray tracing itterations. this functions calls from render_scene function. (Not Ready)

7. structures.cuh - all abstract and computation structures for ray tracing with functions of work with them (Ready, may be updated).

8. figures.cuh - here classes for split 3d figures to triangles and lamp diods. (Not Ready)

9. material_table.cuh - class for define material in array of materials