from videosys.core.parallel_mgr import PgMesh


mesh = PgMesh(24,1,2, 3, 4)

print(mesh.get_ranks_along_axis(0))
print(mesh.get_ranks_along_axis(1))
print(mesh.get_ranks_along_axis(2))


mesh = PgMesh(8,0, 1, 1, 8)

print(mesh.get_ranks_along_axis(0))
print(mesh.get_ranks_along_axis(1))
print(mesh.get_ranks_along_axis(2))
