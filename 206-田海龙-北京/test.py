
import os

cur_dir=os.getcwd()
ens=os.listdir()

subDirs = [f for f in ens if os.path.isdir(os.path.join(cur_dir, f))]
subDirs.remove("206-田海龙-北京")

print(len(subDirs))
# print(subDir)

for dir in subDirs:
    print(f"{dir}/")