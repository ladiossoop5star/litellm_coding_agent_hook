import runpy


namespace = runpy.run_path("/backport-tests/test_tool_result_images_backport.py")
tests = sorted(
    (name, function)
    for name, function in namespace.items()
    if name.startswith("test_") and callable(function)
)

for name, function in tests:
    function()
    print("PASS", name)

print("TOTAL", len(tests))
