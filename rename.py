import os
import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    # Replace cutlass
    new_content = re.sub(r'cutlass', 'tilelang', new_content)
    new_content = re.sub(r'Cutlass', 'TileLang', new_content)
    new_content = re.sub(r'CUTLASS', 'TILELANG', new_content)
    
    # Replace cutedsl
    new_content = re.sub(r'cutedsl', 'tilelang', new_content)
    new_content = re.sub(r'CuteDSL', 'TileLang', new_content)
    new_content = re.sub(r'CUTEDSL', 'TILELANG', new_content)

    # Avoid recursive renames like tilelang_tilelang or flashinfer_tilelang_tilelang
    # (Since I already did some renames manually)
    
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
            if 'cutlass' in name.lower() or 'cutedsl' in name.lower():
                new_name = name.replace('cutlass', 'tilelang').replace('cutedsl', 'tilelang')
                os.rename(filepath, os.path.join(root, new_name))
                print(f"Renamed file: {name} -> {new_name}")

        for name in dirs:
            if 'cutlass' in name.lower() or 'cutedsl' in name.lower():
                new_name = name.replace('cutlass', 'tilelang').replace('cutedsl', 'tilelang')
                os.rename(os.path.join(root, name), os.path.join(root, new_name))
                print(f"Renamed dir: {name} -> {new_name}")

rename_files_and_directories('python/')
rename_files_and_directories('sgl-kernel/')
rename_files_and_directories('docs/')
rename_files_and_directories('test/')
