import os
import re

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return
    
    new_content = content
    # Replace cuda
    new_content = re.sub(r'cuda', 'rtriton', new_content)
    new_content = re.sub(r'Cuda', 'Rtriton', new_content)
    new_content = re.sub(r'CUDA', 'RTRITON', new_content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

def rename_files_and_directories(start_path):
    for root, dirs, files in os.walk(start_path, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            if name.endswith('.py') or name.endswith('.md') or name.endswith('.cu') or name.endswith('.cuh') or name.endswith('.cpp') or name.endswith('.cc') or name.endswith('.toml'):
                process_file(filepath)
            
            # rename file
            if 'cuda' in name.lower():
                new_name = name.replace('cuda', 'rtriton').replace('CUDA', 'RTRITON')
                os.rename(filepath, os.path.join(root, new_name))
                print(f"Renamed file: {name} -> {new_name}")

        for name in dirs:
            if 'cuda' in name.lower():
                new_name = name.replace('cuda', 'rtriton').replace('CUDA', 'RTRITON')
                os.rename(os.path.join(root, name), os.path.join(root, new_name))
                print(f"Renamed dir: {name} -> {new_name}")

rename_files_and_directories('python/')
rename_files_and_directories('sgl-kernel/')
rename_files_and_directories('docs/')
rename_files_and_directories('test/')
