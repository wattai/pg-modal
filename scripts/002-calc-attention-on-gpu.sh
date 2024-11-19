modal run scripts/002-calc-attention-on-gpu.py

# ❯ ./scripts/002-calc-attention-on-gpu.sh
# ✓ Initialized. View run at https://modal.com/apps/wattai/main/ap-XXX
# ✓ Created objects.
# ├── 🔨 Created mount /home/wattai/dev/pg-modal/scripts/002-calc-attention-on-gpu.py
# └── 🔨 Created function run_on_modal.
# Output:  tensor([[[ 0.2948,  0.0937, -0.0547, -0.9989,  0.4071],
#          [ 0.3575, -0.1298,  0.1484, -1.0902,  0.4050],
#          [ 0.3430, -0.0678,  0.0975, -1.0700,  0.4060]]], device='cuda:0',
#        grad_fn=<ViewBackward0>)
# Attention Weights:  tensor([[[0.0484, 0.2293, 0.7223],
#          [0.4553, 0.3486, 0.1961],
#          [0.3713, 0.3055, 0.3232]]], device='cuda:0')
# Stopping app - local entrypoint completed.
# ✓ App completed. View run at https://modal.com/apps/wattai/main/ap-XXX
