@echo off
set threads=1 2 4 8
set sizes=200 400 800 1200 1600 2000
for %%t in (%threads%) do (
    for %%n in (%sizes%) do (
        echo Running: threads=%%t, size=%%n
        matrix_mul.exe matrixA_%%n.txt matrixB_%%n.txt result_%%n_%%t.txt %%t
    )
)
echo Done.