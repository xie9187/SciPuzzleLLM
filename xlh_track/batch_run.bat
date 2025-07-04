@echo off

for /l %%i in (1, 1, 100) do (
    echo [run: %%i]
    python periodic_table.py
)