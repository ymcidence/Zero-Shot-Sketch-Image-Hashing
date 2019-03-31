from util.layer.alex_net import alex_net


def net_image(tensor_in, with_fc=False):
    if with_fc:
        fc_7, _ = alex_net(tensor_in, with_fc=with_fc)
        return fc_7
    else:
        feat_map = alex_net(tensor_in, with_fc=with_fc)
        return feat_map


def net_sketch(tensor_in, with_fc=False):
    if with_fc:
        fc_7, _ = alex_net(tensor_in, with_fc=with_fc)
        return fc_7
    else:
        feat_map = alex_net(tensor_in, with_fc=with_fc)
        return feat_map
