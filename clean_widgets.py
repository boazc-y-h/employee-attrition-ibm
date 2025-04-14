import os
import nbformat

def clean_widget_metadata(root_dir):
    count = 0
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.ipynb'):
                path = os.path.join(folder, file)
                with open(path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

                if 'widgets' in nb.metadata:
                    del nb.metadata['widgets']
                    with open(path, 'w', encoding='utf-8') as f:
                        nbformat.write(nb, f)
                    print(f"✔ Removed widgets metadata from: {path}")
                    count += 1

    if count == 0:
        print("🎉 No notebooks had 'widgets' metadata.")
    else:
        print(f"✅ Cleaned {count} notebook(s).")

# 🔧 Replace with your folder path
clean_widget_metadata("notebooks")