@echo off
for /l %%i in (1,1,94) do (
    echo [run: %%i]
    python periodic_table.py
)
echo.
echo === Running pythonw.exe processes ===
tasklist /FI "IMAGENAME eq pythonw"
echo.
echo 按任意键关闭窗口…
pause