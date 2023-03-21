import struct
import torch

state_dict = torch.load("weights/ch_ptocr_v3_rec_infer.pth", map_location="cpu")

with open("weights/ch_ptocr_v3_rec_infer.wts", 'w') as f:
    f.write("{}\n".format(len(state_dict.keys())))

    for k, v in state_dict.items():
        print(k, v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
