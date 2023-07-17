cd C:/Users/MAIN
cmake -G "Visual Studio 16 2019" -S C:\Users\MAIN\source\repos\Pyfrost -B C:\Users\MAIN\source\builds\Pyfrost -DCMAKE_INSTALL_PREFIX=bifrost_packs
cmake --build C:\Users\MAIN\source\builds\Pyfrost --target install --config Release