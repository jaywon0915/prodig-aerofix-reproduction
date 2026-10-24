import importlib
import sys

packages = ["torch", "gsplat", "pycolmap", "numpy", "PIL", "sklearn", "tqdm"]
failed = []
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"[FAIL] {name}: {exc}")

if failed:
    print("\nMissing/broken dependencies:")
    for name, err in failed:
        print(f"  - {name}: {err}")
    sys.exit(1)
print("\nEnvironment looks ready for ProDiG training.")
