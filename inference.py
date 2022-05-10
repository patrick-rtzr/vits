from scipy.io.wavfile import write
import torch
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence


device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default='만나서 반가워')
parser.add_argumnet("--config_path", type=str, default="config/kss_base.json")
parser.add_argument("--checkpoint_path", type=str, default="logs/results/G_98000.pth")
parser.add_argument("--output_path", type=str, default="test.wav")
args = parser.parse_args()


def get_text(text, hps):
    text_norm = text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file(args.config_path)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(args.checkpoint_path, net_g, None)

stn_tst = get_text(args.text, hps)
with torch.no_grad():
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    write(args.output_file, hps.data.sampling_rate, audio)
    print(args.output_file)
