import hashlib, os, sys

def sha256_of_file(path, bufsize=1<<20):  # 1 MiB buffer
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_of_dir(path):
    for root, dirs, files in os.walk(path):
        for file in sorted(files):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, path)
            print(f"{sha256_of_file(full_path)}  {rel_path}")

if __name__ == "__main__" and len(sys.argv) == 2:
    sha256_of_dir(sys.argv[1])