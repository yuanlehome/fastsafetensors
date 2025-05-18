import paddle
from fastsafetensors import SafeTensorsFileLoader, SingleGroup

import sys
sys.path.insert(0, "/workspace/fastsafetensors/fastsafetensors")

device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
loader = SafeTensorsFileLoader(SingleGroup(), device, nogds=True, debug_log=True, framework="paddle")

loader.add_filenames({0: ["a_paddle.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
keys = list(fb.key_to_rank_lidx.keys())
for k in keys:
    t = fb.get_tensor(k)
    print(f' k, shape = {k, t.shape}\n')
fb.close()

loader.reset() # reset the loader for reusing with different set of files
loader.add_filenames({0: ["b_paddle.safetensors"]}) # {rank: files}
fb = loader.copy_files_to_device()
keys = list(fb.key_to_rank_lidx.keys())
for k in keys:
    t = fb.get_tensor(k)
    print(f' k, shape = {k, t.shape}\n')
fb.close()
loader.close()
