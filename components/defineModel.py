def defineG(config):
    script_name = "components." + config["module_script_name"]
    print('model name: %s' % config["module_script_name"])
    class_name = config["class_name"]
    package = __import__(script_name, fromlist=True)
    network_class = getattr(package, class_name)

    model_name = config["module_script_name"]
    if model_name == 'OmniSR':
        # TODO replace below lines to define the model framework
        network = network_class(1,
                                1,
                                config["feature_num"],
                                **config["module_params"]
                                )
        return network
    elif model_name == 'swinIR':
        lr_patch_size = config['dataset_params']['lr_patch_size']
        # model = network_class(upscale=config['module_params']['upsampling'], in_chans=1, img_size=(lr_patch_size,lr_patch_size), window_size=8,
        #             img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        #             mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        model = network_class(upscale=config['module_params']['upsampling'], in_chans=1, img_size=(lr_patch_size, lr_patch_size), window_size=8,
                              img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                              mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')

        return model
    elif model_name == 'IMDN':
        model = network_class(in_nc=1, out_nc=1, upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'Ours':
        model = network_class(1,
                              1,
                              config["feature_num"],
                              up_scale=config['module_params']['upsampling'],
                              res_num=config['module_params']['res_num'],
                              block_num=config['module_params']['block_num'],
                              bias=config['module_params']['bias'],
                              window_size=config['module_params']['window_size'],
                              pe=True, ffn_bias=True
                              )
        return model
    elif model_name == 'OursMany':
        model = network_class(1,
                              1,
                              config["feature_num"],
                              up_scale=config['module_params']['upsampling'],
                              res_num=config['module_params']['res_num'],
                              block_num=config['module_params']['block_num'],
                              bias=config['module_params']['bias'],
                              window_size=config['module_params']['window_size'],
                              pe=True, ffn_bias=True
                              )
        return model
    elif model_name == 'CANM':
        model = network_class(scale=config['module_params']['upsampling'],)
        return model
    elif model_name == 'MGDUN':
        model = network_class(inchannel=1, n_feats=64,scale=config['module_params']['upsampling'] )
        return model

