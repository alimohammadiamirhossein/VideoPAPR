def transfer_unets(pipe_zero123, pipe_svd):
    # Assign transformer_blocks and spatial_res_block from pipe_zero123 to pipe_svd
    for i in range(4):
        # Down blocks
        if i < 3:
            for j in range(2):
                pipe_svd.unet.down_blocks[i].attentions[j].transformer_blocks = pipe_zero123.unet.down_blocks[i].attentions[j].transformer_blocks
        for j in range(2):
            pipe_svd.unet.down_blocks[i].resnets[j].spatial_res_block = pipe_zero123.unet.down_blocks[i].resnets[j]

        # Up blocks
        if i > 0:
            for j in range(3):
                pipe_svd.unet.up_blocks[i].attentions[j].transformer_blocks = pipe_zero123.unet.up_blocks[i].attentions[j].transformer_blocks
        for j in range(2):
            pipe_svd.unet.up_blocks[i].resnets[j].spatial_res_block = pipe_zero123.unet.up_blocks[i].resnets[j]

    return pipe_zero123, pipe_svd