from .conv_block import Conv2dBlock, SeparableConv2dBlock

# from .mamba_block import build_mamba_block
# from .vit_block import build_vit_block


# def build_block(name, **kwargs):
#     cls = globals().get(name, None)
#     if cls is None:
#         raise ValueError(f'Block named {name} is not support.')
#     return cls(**kwargs)


# def list_blocks(filter=''):
#     block_list = [k for k in globals().keys() if 'Block' in k]
#     if len(filter):
#         return fnmatch.filter(block_list, filter)  # include these blocks
#     else:
#         return block_list
