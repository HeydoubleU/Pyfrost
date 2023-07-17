import os, shutil


def removeFolder(_dir, startswith):
    if not os.path.isdir(_dir):
        return

    for _folder in os.listdir(_dir):
        _path = os.path.join(_dir, _folder)
        if _folder.startswith(startswith) and os.path.isdir(_path):
            shutil.rmtree(_path)
            return


removeFolder("C:/Users/MAIN/bifrost_packs", "Pyfrost")
removeFolder("C:/Users/MAIN/source/builds", "Pyfrost")
removeFolder("C:/Users/MAIN/source/repos/Pyfrost/out", "build")

# run batch file
os.system("C:/Users/MAIN/source/repos/Pyfrost/build.bat")

# copy files
for folder in os.listdir("C:/Users/MAIN/bifrost_packs/Pyfrost-2.0.0"):
    path = os.path.join("C:/Users/MAIN/bifrost_packs/Pyfrost-2.0.0", folder)
    if folder.startswith("lib") and os.path.isdir(path):
        dst = os.path.join("A:/TeamEnvironment/Compounds/Pyfrost", folder)
        if os.path.isdir(dst):
            shutil.rmtree(os.path.join("A:/TeamEnvironment/Compounds/Pyfrost", folder))
        shutil.copytree(path, dst)
        break
