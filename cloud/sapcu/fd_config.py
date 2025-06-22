import fd_coder as coder

def get_model(device):
    decoder = coder.pyramid_Decoder3()
    encoder = coder.DGCNN_cls(20, 1024)
    model = coder.OccupancyNetwork(decoder, encoder, device=device)
    
    return model