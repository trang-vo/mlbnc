import os
import shutil

if __name__ == "__main__":
    log_root = "../logs"
    folders = os.listdir(log_root)
    for folder in folders:
        if "Env" in folder:
            files = os.listdir(os.path.join(log_root, folder))
            if len(files) <= 3:
                is_removed = True
                for f in files:
                    if "model" in f:
                        is_removed = False
                        break
                if is_removed:
                    print(folder, files)
                    shutil.move(os.path.join(log_root, folder), "../trash")
