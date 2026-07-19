import zipfile
import os

files_to_pack = [
    "unsloth/np_engine.py",
    "unsloth/__init__.py",
    "test_unsloth_np_integration.py",
    "unsloth_np_comprehensive_benchmark.py",
    "UNSLOTH_NP_INTEGRATION_GUIDE.md"
]

zip_name = "Unsloth_NP_Engine_Upload.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for f in files_to_pack:
        if os.path.exists(f):
            zipf.write(f)
            print(f"Packed: {f}")

print(f"\n✅ Created archive: {os.path.abspath(zip_name)}")
